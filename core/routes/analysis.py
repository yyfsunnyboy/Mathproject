# -*- coding: utf-8 -*-
"""
=============================================================================
模組名稱 (Module Name): core/routes/analysis.py

說明 (Description):
  練習與測驗相關之 AI 分析路由：手寫辨識／批改、Chat AI 對話、考卷影像分析等。

用法 (Usage):
  由 Flask 應用程式註冊 blueprint 後載入。

版本 (Version): V2.0
日期 (Date): 2026-01-13
維護者 (Maintainer): Math AI Project Team
=============================================================================
"""

from flask import request, jsonify, render_template, current_app, url_for, session
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

import os
import base64
import re
import tempfile
import uuid
import traceback

from . import core_bp, practice_bp
from core.session import get_current
from config import Config
from core.ai_analyzer import (
    build_chat_prompt,
    get_chat_response,
    analyze,
    diagnose_error,
    diversify_follow_up_prompts,
    build_dynamic_follow_up_prompts_variant,
    clean_and_parse_json,
    enforce_strict_mode,
)
from core.ai_client import call_ai, call_google_model
from core.ai_settings import get_effective_model_config
from core.adaptive.judge import (
    judge_answer_with_feedback,
    _as_symbolic_tolerant,
    _normalize_math_text,
    _parse_quotient_remainder,
    _symbolic_equal,
    _extract_divisor_from_question,
)
from core.exam_analyzer import analyze_exam_image, save_analysis_result
from core.diagnosis_analyzer import perform_weakness_analysis
from core.rag_engine import rag_search, rag_chat
from models import db, MistakeNotebookEntry, ExamAnalysis, SkillInfo


ALLOWED_EXAM_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def _handwriting_not_recognized_response():
    """缺圖、payload 無效、OCR 無法讀取或運算式不可用時的回應；不呼叫教學用 LLM。"""
    return jsonify({"success": False, "reason": "handwriting_not_recognized"})


def _handwriting_expression_from_parsed(parsed):
    if not isinstance(parsed, dict):
        return ""
    v = parsed.get("expression")
    if v is None:
        v = parsed.get("transcribed_expression") or parsed.get("text")
    if v is None:
        return ""
    return str(v).strip()


def _handwriting_expression_is_usable(expr):
    if not expr:
        return False
    low = expr.lower().strip()
    bogus_exact = {
        "n/a", "na", "none", "null", "empty",
        "?", "??", "???",
        "no text", "unreadable", "illegible",
    }
    if low in bogus_exact or low in {".", "-", "?"}:
        return False
    if len(expr) <= 16 and (
        "unreadable" in low or "illegible" in low or "no text" in low
    ):
        return False
    mathish = set("+-*/=^()[]0123456789")
    if any((c in mathish or c.isdigit()) for c in expr):
        return True
    if any("a" <= c.lower() <= "z" for c in expr if c.isalpha()):
        return True
    if "\\" in expr or "$" in expr:
        return True
    if len(expr) >= 4 and any("\u4e00" <= c <= "\u9fff" for c in expr):
        return True
    return len(expr) >= 8


def _handwriting_expected_answer(data, state, question_text, question_context=""):
    """Resolve expected answer with strict priority:
    0) POST expected_answer
    1) adaptive runtime current question correct_answer
    2) POST correct_answer
    2.5) session correct_answer
    3) answer parsed from question_context
    """
    # 0) POST expected_answer
    ca = str(data.get("expected_answer") or "").strip()
    if ca: return ca

    def _extract_from_question_context(ctx):
        raw = str(ctx or "").strip()
        if not raw:
            return ""
        try:
            import json as _json
            obj = _json.loads(raw)
            if isinstance(obj, dict):
                for k in ("correct_answer", "expected_answer", "answer"):
                    v = str(obj.get(k) or "").strip()
                    if v:
                        return v
        except Exception:
            pass
        m = re.search(
            r"(?:正確答案|答案|correct_answer|expected_answer)\s*[:：=]\s*([^\n\r;；]+)",
            raw,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().strip('"').strip("'")
        return ""

    sid = str(data.get("session_id") or state.get("session_id") or "").strip()
    qt = str(question_text or "").strip()
    store = session.get("adaptive_runtime") or {}

    # 1) session runtime current question correct_answer
    if isinstance(store, dict):
        if sid and isinstance(store.get(sid), dict):
            run = store.get(sid) or {}
            rq = str(run.get("question_text") or "").strip()
            ca = str(run.get("correct_answer") or "").strip()
            if ca and (not qt or not rq or rq == qt):
                return ca
        if qt:
            for _k, run in store.items():
                if not isinstance(run, dict):
                    continue
                rq = str(run.get("question_text") or "").strip()
                ca = str(run.get("correct_answer") or "").strip()
                if ca and rq == qt:
                    return ca

    # 2) POST correct_answer
    ca = str(data.get("correct_answer") or "").strip()
    if ca:
        return ca

    # 2.5) session correct_answer
    if isinstance(state, dict):
        ca = str(state.get("correct_answer") or state.get("expected_answer") or "").strip()
        if ca:
            return ca

    # 3) question_context answer
    return _extract_from_question_context(question_context)


def _handwriting_final_answer_candidates(expr):
    """Prefer last line / RHS of = as final answer when transcription has steps."""
    raw = (expr or "").strip()
    if not raw:
        return []
    chunks = []
    for part in re.split(r"[;\n\r]+", raw):
        p = part.strip()
        if not p:
            continue
        chunks.append(p)
        if "=" in p:
            rhs = p.split("=")[-1].strip()
            if rhs:
                chunks.append(rhs)
    if not chunks:
        return [raw]
    ordered = []
    seen = set()
    for c in reversed(chunks):
        key = c.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(c)
    if raw.lower() not in seen:
        ordered.append(raw)
    return ordered


def _handwriting_pick_final_expression(expr):
    raw = str(expr or "").strip()
    if not raw:
        return ""
    parts = [p.strip() for p in re.split(r"[;\n\r]+", raw) if p.strip()]
    if not parts:
        return raw
    last = parts[-1]
    if "=" in last:
        rhs = last.split("=")[-1].strip()
        if rhs:
            return rhs
    return last


def _handwriting_skill_focus(family_id):
    fid = str(family_id or "").strip()
    by_id = {
        "F1": "同類項合併與符號整理",
        "F2": "去括號與負號分配，再合併同類項",
        "F3": "單項式對每一項完整分配",
        "F4": "運算順序與等值化簡",
        "F5": "多項式展開時每一項都要相乘",
        "F6": "公式辨識與正確套用",
        "F7": "逐項相除並保持次方規則",
        "F8": "商與餘式關係判斷",
        "F9": "長除法步驟與對齊",
        "F10": "反推除式與被除式關係",
        "F11": "混合式化簡與結果整理",
        "F12": "幾何公式代入與化簡",
        "F13": "複合情境建式與計算",
    }
    by_en = {
        "poly_add_sub_flat": "同類項合併與符號整理",
        "poly_add_sub_nested": "去括號與負號分配，再合併同類項",
        "poly_add_sub_unknown": "符號與同類項整理",
        "poly_mul_monomial": "單項式對每一項完整分配",
        "poly_mul_poly": "多項式展開時每一項都要相乘",
        "poly_mul_special_identity": "公式辨識與正確套用",
        "poly_div_monomial_eval": "逐項相除並保持次方規則",
        "poly_div_monomial_qr": "商與餘式關係判斷",
        "poly_div_poly_qr": "長除法步驟與對齊",
        "poly_div_reverse": "反推除式與被除式關係",
        "poly_mixed_simplify": "混合式化簡與結果整理",
        "poly_geom_formula_direct": "幾何公式代入與化簡",
        "poly_geom_region_composite": "複合情境建式與計算",
    }
    return by_id.get(fid) or by_en.get(fid) or ""


def _handwriting_expr_partially_close(user_segment, correct_answer):
    """True if sympy parses both and difference is a non-zero constant (typical small slip)."""
    try:
        from sympy import simplify

        su = _as_symbolic_tolerant(user_segment)
        sc = _as_symbolic_tolerant(correct_answer)
        if su is None or sc is None:
            return False
        diff = simplify(su - sc)
        if diff == 0:
            return False
        return not diff.free_symbols
    except Exception:
        return False


ERROR_MECHANISMS = (
    "structure_error",
    "sign_error",
    "combine_error",
    "operation_error",
    "substitution_error",
    "notation_error",
    "unknown",
)

ERROR_MECHANISM_FEEDBACK = {
    "structure_error": "前一步的結構轉換不正確，請先確認去括號或展開是否做對。",
    "sign_error": "正負號處理有誤，請再檢查負號如何分配到各項。",
    "combine_error": "同類項或常數項合併錯誤，請再檢查最後整理。",
    "operation_error": "運算規則使用有誤，請先確認這一步應用的運算關係。",
    "substitution_error": "代值或抄值可能有誤，請先核對代入的數值與位置。",
    "notation_error": "式子格式或記號不完整，請先補齊括號、變數與表示方式。",
    "unknown": "最後整理結果不正確，請再檢查最終答案。",
}

# 白板 second-stage / rule-based fallback 用的「具體下一步」（不揭露數值答案）
ERROR_MECHANISM_ACTION_STEP = {
    "structure_error": "請先重寫去括號或展開後的式子，逐項核對符號是否正確，再重新合併同類項。",
    "sign_error": "請把負號或減號如何分配到括號內「每一項」寫清楚，再重新加減係數與常數項。",
    "combine_error": "請把同類項分組列出，確認次方相同的項才合併，再重算一次係數加總。",
    "operation_error": "請對照題目這一步應使用的運算規則，逐步寫出中間式再接到下一步。",
    "substitution_error": "請把代入或抄寫的數字與位置逐一核對，再重算代入後的運算。",
    "notation_error": "請先補齊括號、變數與等號兩側的式子結構，再依題意繼續化簡。",
    "unknown": "請從題目要求的最簡形式往回對照：括號、符號與同類項是否都處理完畢，再整理一次。",
}

FAMILY_ERROR_PRIORITY = {
    "F1": ["sign_error", "combine_error", "notation_error", "unknown"],
    "poly_add_sub_flat": ["sign_error", "combine_error", "notation_error", "unknown"],
    "F2": ["structure_error", "sign_error", "combine_error", "unknown"],
    "poly_add_sub_nested": ["structure_error", "sign_error", "combine_error", "unknown"],
    "F3": ["structure_error", "combine_error", "notation_error", "unknown"],
    "poly_add_sub_unknown": ["structure_error", "combine_error", "notation_error", "unknown"],
    "F4": ["operation_error", "sign_error", "combine_error", "unknown"],
    "poly_mul_monomial": ["operation_error", "sign_error", "combine_error", "unknown"],
    "F5": ["structure_error", "operation_error", "sign_error", "combine_error", "unknown"],
    "poly_mul_poly": ["structure_error", "operation_error", "sign_error", "combine_error", "unknown"],
    "F9": ["structure_error", "sign_error", "operation_error", "notation_error", "unknown"],
    "poly_div_poly_qr": ["structure_error", "sign_error", "operation_error", "notation_error", "unknown"],
}

F9_ERROR_MAIN_ISSUE = {
    "structure_error": "長除法各列的對齊與次序關係不清楚，請先釐清商、乘回、相減三種列。",
    "sign_error": "相減時整行變號有誤，請逐項檢查正負。",
    "operation_error": "乘回除式或上下相減的規則有誤，請對照每一步係數。",
    "notation_error": "長除法書寫不完整，請補齊商與餘數的標示。",
}

_F9_TUTOR_ROWS = (
    # (mechanism, step_focus or "" for default, hint_focus, guided_question, micro_step)
    ("structure_error", "subtract_row_1", "對齊相減後各列", "次方直欄是否對齊？", "缺項可補零再減"),
    ("structure_error", "quotient_term_1", "對齊商與被除式", "第一次商對到哪一階項？", "先寫第一次商"),
    ("structure_error", "final_remainder", "最後一列要對齊", "餘式與除式次方？", "確認最底行寫完"),
    ("structure_error", "", "長除法列要對齊", "商、乘回、相減順序？", "由上到下逐步核對"),
    ("sign_error", "final_remainder", "相減後餘項變號", "最底行每項符號？", "整行相減逐項改"),
    ("sign_error", "subtract_row_1", "相減要逐項變號", "這列減上列每項？", "先改符號再加"),
    ("sign_error", "", "相減逐項變號", "上下相減有漏負？", "整行取反再相加"),
    ("operation_error", "final_remainder", "最後餘數要檢查", "餘式還能再除？", "看餘式次方"),
    ("operation_error", "quotient_term_2", "下一項商核對", "降次後再除一次？", "對齊再取下一商"),
    ("operation_error", "subtract_row_2", "第二次相減核對", "乘回列是否完整？", "先乘回再相減"),
    ("operation_error", "quotient_term_1", "第一次相除核對", "首項除以除式首項？", "只決定第一項商"),
    ("operation_error", "", "乘回與相減規則", "商的每項都乘除式？", "一步一步寫乘回"),
    ("notation_error", "final_remainder", "餘數要寫清楚", "最後有沒有標餘？", "用商餘格式彙整"),
    ("notation_error", "", "長除法書寫", "商與餘數標示？", "依題目格式整理"),
)


def _f9_long_division_layout_heuristic(expr: object) -> bool:
    """True when OCR-flattened work still looks like polynomial long division (not a single line)."""
    raw = str(expr or "").strip()
    if not raw:
        return False
    parts = [p.strip() for p in re.split(r"[;\n\r]+", raw) if p.strip()]
    if len(parts) >= 3:
        return True
    if raw.count(";") >= 2:
        return True
    if ")" in raw and len(parts) >= 2:
        return True
    if "..." in raw or "…" in raw:
        return True
    return False


def _f9_find_student_quotient_remainder(parts: list[str]):
    """Best-effort QR parse from semicolon-separated OCR lines (single segment only)."""
    for p in reversed(parts):
        qr = _parse_quotient_remainder(p)
        if qr:
            return qr
    joined = " ".join(parts)
    return _parse_quotient_remainder(joined)


def _f9_remainder_is_sign_flip(ur: str, cr: str) -> bool:
    try:
        from sympy import simplify

        su = _as_symbolic_tolerant(ur)
        sc = _as_symbolic_tolerant(cr)
        if su is None or sc is None:
            return False
        return simplify(su + sc) == 0
    except Exception:
        return False


def _f9_default_step_focus(parts: list[str]) -> str:
    n = len(parts)
    if n >= 5:
        return "subtract_row_1"
    if n >= 3:
        return "quotient_term_1"
    return ""


def analyze_polynomial_long_division_layout(
    recognized_expression: object,
    expected_answer: object,
    question_text: object,
) -> dict:
    """
    Rule-based F9 / poly_div_poly_qr layout analysis. Does not call LLMs.
    Returns error_mechanism, main_issue, step_focus for merging into structured_analysis.
    """
    rec = str(recognized_expression or "").strip()
    exp = str(expected_answer or "").strip()
    parts = [p.strip() for p in re.split(r"[;\n\r]+", rec) if p.strip()]

    def pack(mech: str, step: str) -> dict:
        return {
            "error_mechanism": mech,
            "main_issue": F9_ERROR_MAIN_ISSUE.get(mech, ERROR_MECHANISM_FEEDBACK.get(mech, "")),
            "step_focus": step,
        }

    exp_qr = _parse_quotient_remainder(exp)
    st_qr = _f9_find_student_quotient_remainder(parts) if parts else _parse_quotient_remainder(rec)

    # No expected QR: still avoid blanket notation_error when multiline layout exists.
    if not exp_qr:
        if len(parts) >= 3:
            return pack("structure_error", _f9_default_step_focus(parts))
        if len(parts) >= 2:
            return pack("structure_error", "quotient_term_1")
        return pack("notation_error", "")

    cq, cr = exp_qr

    if st_qr:
        uq, ur = st_qr
        q_ok = _symbolic_equal(uq, cq)
        r_ok = _symbolic_equal(ur, cr)
        if q_ok and r_ok:
            return {
                "error_mechanism": "structure_error",
                "main_issue": "長除法演算與最後答案欄位對應不一致，請確認分號後最後一行是否為完整商餘格式。",
                "step_focus": "final_remainder",
            }

        q_step = "quotient_term_2" if len(parts) >= 5 else "quotient_term_1"
        if not q_ok and r_ok:
            return pack("operation_error", q_step)
        if q_ok and not r_ok:
            if _f9_remainder_is_sign_flip(ur, cr):
                return pack("sign_error", "final_remainder")
            return pack("operation_error", "final_remainder")
        # both differ: prefer structure when many rows (OCR layout noise), else operation
        if len(parts) >= 4:
            return pack("structure_error", "subtract_row_1")
        return pack("operation_error", q_step)

    # Multiline work but final answer not in QR — treat as layout / step trace issue first.
    if len(parts) >= 3:
        return pack("structure_error", _f9_default_step_focus(parts))
    if len(parts) == 2:
        return pack("structure_error", "quotient_term_1")

    # Single blob: try labeled divisor row hint
    div_hint = _extract_divisor_from_question(question_text)
    if div_hint and div_hint.replace(" ", "") and parts:
        row0 = parts[0].replace(" ", "")
        if div_hint.replace(" ", "") in row0:
            return pack("structure_error", "quotient_term_1")

    return pack("notation_error", "")


def _f9_tutor_guidance_pick(error_mechanism: str, step_focus: str):
    mech = str(error_mechanism or "").strip()
    step = str(step_focus or "").strip()
    for m, s, h, g, ms in _F9_TUTOR_ROWS:
        if m == mech and s == step:
            return h, g, ms
    for m, s, h, g, ms in _F9_TUTOR_ROWS:
        if m == mech and s == "":
            return h, g, ms
    return None


def _handwriting_has_unflipped_negative_parenthesis(question_text, user_segment):
    """Detect F2-specific mistake: '-(...)' kept original inner signs in student's expansion."""
    q = re.sub(r"\s+", "", str(question_text or "")).lower()
    u = re.sub(r"\s+", "", str(user_segment or "")).lower()
    if not q or not u:
        return False
    inners = re.findall(r"-\(([^()]+)\)", q)
    if not inners:
        return False
    for inner in inners:
        inner_clean = re.sub(r"\s+", "", inner).lower()
        if not inner_clean:
            continue
        # Student still keeps original inner term direction (e.g. -x+1) in work line.
        if inner_clean in u:
            return True
    return False


def _handwriting_infer_error_mechanism(user_segment, expected_answer, family_id, question_text=""):
    fid = str(family_id or "").strip()
    priorities = FAMILY_ERROR_PRIORITY.get(fid, ["sign_error", "combine_error", "unknown"])
    detected = set()

    # F2: highest-value deterministic structure/sign check.
    if fid in ("F2", "poly_add_sub_nested"):
        if _handwriting_has_unflipped_negative_parenthesis(question_text, user_segment):
            detected.add("structure_error")
            detected.add("sign_error")

    try:
        from sympy import Symbol, degree, expand, simplify, Poly

        x = Symbol("x")
        su = _as_symbolic_tolerant(user_segment)
        sc = _as_symbolic_tolerant(expected_answer)
        if su is None or sc is None:
            for mech in priorities:
                if mech in detected:
                    return mech
            return "notation_error"
        eu = expand(su)
        ec = expand(sc)
        diff = simplify(eu - ec)
        if diff == 0:
            return ""

        if ec.has(x):
            deg_c = int(degree(ec, x))
            deg_u = int(degree(eu, x)) if eu != 0 else -1
            if fid in ("F4", "poly_mul_monomial", "F5", "poly_mul_poly"):
                if deg_c > 0 and 0 <= deg_u < deg_c:
                    detected.add("operation_error")
                if deg_c > 0 and deg_u > deg_c:
                    detected.add("structure_error")

            try:
                pc = Poly(ec, x)
                pu = Poly(eu, x)
                exp_coeff = {int(k[0]): v for k, v in pc.as_dict().items()}
                usr_coeff = {int(k[0]): v for k, v in pu.as_dict().items()}
                if set(exp_coeff.keys()) != set(usr_coeff.keys()):
                    detected.add("structure_error")
                else:
                    # same term degrees but coefficients differ -> combine/sign
                    coeff_mismatch = any(exp_coeff[d] != usr_coeff[d] for d in exp_coeff.keys())
                    if coeff_mismatch:
                        detected.add("combine_error")
            except Exception:
                detected.add("notation_error")

        # Simple sign heuristic if expression contains opposite-signed counterpart.
        n_user = re.sub(r"\s+", "", str(user_segment or ""))
        n_exp = re.sub(r"\s+", "", str(expected_answer or ""))
        if ("-" in n_exp or "-(" in n_exp) and ("-" in n_user or "-(" in n_user):
            if n_user.startswith("-") != n_exp.startswith("-"):
                detected.add("sign_error")

        q = str(question_text or "")
        if re.search(r"[a-zA-Z]\s*=", q) and re.search(r"\d", str(user_segment or "")):
            detected.add("substitution_error")

        # F2/F5 additional structure preference when mechanism still ambiguous.
        if fid in ("F2", "poly_add_sub_nested") and "combine_error" in detected and "sign_error" in detected:
            detected.add("structure_error")
        if fid in ("F5", "poly_mul_poly") and "structure_error" in detected:
            detected.add("operation_error")
    except Exception:
        detected.add("unknown")

    for mech in priorities:
        if mech in detected:
            return mech
    return "unknown"


_POLY_REFINE_TARGET_FAMILIES = frozenset(
    {
        "F1",
        "F2",
        "F11",
        "poly_add_sub_flat",
        "poly_add_sub_nested",
        "poly_mixed_simplify",
    }
)


def _poly_refine_poly_in_x(expr: object):
    """Return sympy Poly in x, or None."""
    try:
        from sympy import Poly, Symbol, expand

        x = Symbol("x")
        su = _as_symbolic_tolerant(expr)
        if su is None:
            return None
        return Poly(expand(su), x)
    except Exception:
        return None


def _poly_refine_degree_coeff_dict(poly) -> dict | None:
    if poly is None:
        return None
    try:
        return {int(k[0]): v for k, v in poly.as_dict().items()}
    except Exception:
        return None


def _poly_refine_expr_has_x(expr: object) -> bool:
    try:
        from sympy import Symbol, expand

        x = Symbol("x")
        su = _as_symbolic_tolerant(expr)
        if su is not None:
            return expand(su).has(x)
    except Exception:
        pass
    return "x" in _normalize_math_text(expr)


def _poly_refine_expected_has_no_x(expected_answer: object) -> bool:
    try:
        from sympy import Symbol, expand

        x = Symbol("x")
        se = _as_symbolic_tolerant(expected_answer)
        if se is not None:
            return not expand(se).has(x)
    except Exception:
        pass
    return "x" not in _normalize_math_text(expected_answer)


def _question_has_opposite_linear_x_terms(question_text: object) -> bool:
    """True when question text shows at least one pair of x-terms that can cancel (e.g. 6x and -6x)."""
    t = _normalize_math_text(question_text)
    if "x" not in t:
        return False
    # _normalize_math_text inserts implicit '*' (6x -> 6*x); allow optional * before x
    pairs = re.findall(r"([+-]?)(\d*)\*?x", t)
    coeffs: list[int] = []
    for sign, digs in pairs:
        v = int(digs) if digs else 1
        if sign == "-":
            v = -v
        coeffs.append(v)
    if len(coeffs) < 2:
        return False
    for i in range(len(coeffs)):
        for j in range(i + 1, len(coeffs)):
            if coeffs[i] + coeffs[j] == 0:
                return True
    return False


def _poly_refine_global_sign_flip(user_txt: object, exp_txt: object) -> bool:
    try:
        from sympy import expand, simplify

        su = _as_symbolic_tolerant(user_txt)
        se = _as_symbolic_tolerant(exp_txt)
        if su is None or se is None:
            return False
        return simplify(expand(su) + expand(se)) == 0
    except Exception:
        return False


def _poly_refine_all_coeffs_negated(d_exp: dict, d_usr: dict) -> bool:
    if not d_exp or set(d_exp) != set(d_usr):
        return False
    try:
        from sympy import simplify

        for d in d_exp:
            if simplify(d_usr[d] + d_exp[d]) != 0:
                return False
        return True
    except Exception:
        return False


def _f11_question_suggests_expansion(question_text: object) -> bool:
    qt = str(question_text or "")
    if "(" in qt or ")" in qt:
        return True
    t = _normalize_math_text(qt)
    return "^2" in t or "²" in qt or "**2" in t


def _refine_polynomial_unknown_mechanism(
    family_id,
    recognized_expression,
    final_expression,
    expected_answer,
    question_text,
):
    """
    When generic handwriting inference yields unknown, narrow F1/F2/F11 errors using
    deterministic checks on expressions and question text (no LLM).
    Returns dict with error_mechanism, main_issue, step_focus or None.
    """
    fid = str(family_id or "").strip()
    if fid not in _POLY_REFINE_TARGET_FAMILIES:
        return None

    rec = str(recognized_expression or "").strip()
    fin = str(final_expression or "").strip()
    exp = str(expected_answer or "").strip()
    qt = str(question_text or "").strip()

    if not exp:
        return None

    d_exp = _poly_refine_degree_coeff_dict(_poly_refine_poly_in_x(exp))
    d_fin = _poly_refine_degree_coeff_dict(_poly_refine_poly_in_x(fin))
    d_rec = _poly_refine_degree_coeff_dict(_poly_refine_poly_in_x(rec))

    # Prefer final line for "student answer" structure checks; fall back to full OCR.
    d_usr = d_fin if d_fin is not None else d_rec

    if fid in ("F1", "poly_add_sub_flat"):
        # A1: constant expected but work still shows x, and question has cancellable x-terms
        if (
            _poly_refine_expected_has_no_x(exp)
            and (_poly_refine_expr_has_x(fin) or _poly_refine_expr_has_x(rec))
            and _question_has_opposite_linear_x_terms(qt)
        ):
            return {
                "error_mechanism": "combine_error",
                "step_focus": "like_terms",
                "main_issue": "同類項合併錯誤，請先檢查可互相抵消的項是否已正確消去。",
            }
        # A2: same set of term degrees but wrong coefficients -> combine
        if (
            d_exp is not None
            and d_usr is not None
            and set(d_exp) == set(d_usr)
            and d_exp
            and not _poly_refine_all_coeffs_negated(d_exp, d_usr)
            and any(d_exp[k] != d_usr[k] for k in d_exp)
        ):
            return {
                "error_mechanism": "combine_error",
                "step_focus": "like_terms",
                "main_issue": "同類項合併錯誤，請再檢查可合併項的係數加減。",
            }

    if fid in ("F2", "poly_add_sub_nested"):
        if "(" not in qt and ")" not in qt:
            return None
        # B2: missing term degrees after expansion vs expected
        if (
            d_exp is not None
            and d_usr is not None
            and set(d_usr) < set(d_exp)
            and set(d_exp)
        ):
            return {
                "error_mechanism": "structure_error",
                "step_focus": "bracket_scope",
                "main_issue": "去括號或重寫式子的結構有誤，請先確認每一項是否都有正確保留下來。",
            }
        # B1: sign distribution / global flip
        if _handwriting_has_unflipped_negative_parenthesis(qt, rec) or _handwriting_has_unflipped_negative_parenthesis(
            qt, fin
        ):
            return {
                "error_mechanism": "sign_error",
                "step_focus": "sign_distribution",
                "main_issue": "去括號時的正負號處理有誤，請檢查括號前符號是否正確分配到每一項。",
            }
        chk_u, chk_e = (fin or rec), exp
        if _poly_refine_global_sign_flip(chk_u, chk_e):
            return {
                "error_mechanism": "sign_error",
                "step_focus": "sign_distribution",
                "main_issue": "去括號時的正負號處理有誤，請檢查括號前符號是否正確分配到每一項。",
            }
        if (
            d_exp is not None
            and d_usr is not None
            and set(d_exp) == set(d_usr)
            and d_exp
            and _poly_refine_all_coeffs_negated(d_exp, d_usr)
        ):
            return {
                "error_mechanism": "sign_error",
                "step_focus": "sign_distribution",
                "main_issue": "去括號時的正負號處理有誤，請檢查括號前符號是否正確分配到每一項。",
            }

    if fid in ("F11", "poly_mixed_simplify"):
        # C2: expansion context but student polynomial is missing expected degrees
        if (
            _f11_question_suggests_expansion(qt)
            and d_exp is not None
            and d_usr is not None
            and set(d_usr) < set(d_exp)
            and set(d_exp)
        ):
            return {
                "error_mechanism": "structure_error",
                "step_focus": "expansion_structure",
                "main_issue": "展開或重寫式子的結構有誤，請先確認每個括號中的項是否都有正確處理。",
            }
        # C1: same term structure, coefficient merge wrong
        if (
            d_exp is not None
            and d_usr is not None
            and set(d_exp) == set(d_usr)
            and d_exp
            and not _poly_refine_all_coeffs_negated(d_exp, d_usr)
            and any(d_exp[k] != d_usr[k] for k in d_exp)
        ):
            return {
                "error_mechanism": "combine_error",
                "step_focus": "simplify_combine",
                "main_issue": "展開後的同類項整理有誤，請重新檢查哪些項可以合併。",
            }
        # C3: sign flip on matched terms
        if (
            d_exp is not None
            and d_usr is not None
            and set(d_exp) == set(d_usr)
            and d_exp
            and _poly_refine_all_coeffs_negated(d_exp, d_usr)
        ):
            return {
                "error_mechanism": "sign_error",
                "step_focus": "expansion_sign",
                "main_issue": "展開後的正負號處理有誤，請重新檢查負號或減號是否有正確套用到各項。",
            }
        if _poly_refine_global_sign_flip(fin or rec, exp):
            return {
                "error_mechanism": "sign_error",
                "step_focus": "expansion_sign",
                "main_issue": "展開後的正負號處理有誤，請重新檢查負號或減號是否有正確套用到各項。",
            }

    return None


def _clean_math_expr(expr: str) -> str:
    """清理常見格式干擾：空白、全半形、乘號替換"""
    if not expr: return ""
    import unicodedata
    # 轉半形
    cleaned = unicodedata.normalize('NFKC', str(expr))
    # 移除空白與轉換 x 的小寫和乘號
    cleaned = cleaned.replace(" ", "").replace("\t", "").replace("．", ".").replace("X", "x")
    return cleaned

def _handwriting_structured_analysis(recognized_expression, expected_answer, question_text, family_id):
    exp = (expected_answer or "").strip()
    qt = (question_text or "").strip()
    skill_focus = _handwriting_skill_focus(family_id)
    family_label_zh, family_description_zh = _handwriting_family_meta(family_id)
    
    final_expr = _handwriting_pick_final_expression(recognized_expression)
    base = {
        "recognized_expression": recognized_expression,
        "final_expression": final_expr,
        "expected_answer": exp,
        "is_correct": False,
        "status": "unknown",
        "main_issue": "",
        "skill_focus": skill_focus,
        "issue_tag": "",
        "error_mechanism": "unknown",
        "family_id": family_id or "",
        "family_label_zh": family_label_zh,
        "family_description_zh": family_description_zh,
        "step_focus": "",
        "analysis_source": "generic_handwriting_analysis",
    }
    
    if not recognized_expression or str(recognized_expression).strip() == "":
        base["main_issue"] = "系統未能辨識出您的手寫內容，請檢查筆跡是否清晰，或重新書寫一次。"
        base["status"] = "ocr_failed"
        return base
        
    if not exp:
        base["main_issue"] = "目前缺少可對照的標準答案，無法完成穩定批改。"
        base["skill_focus"] = skill_focus or "依題目重點進行化簡與整理"
        base["analysis_source"] = "generic_handwriting_analysis"
        return base

    candidates = [final_expr] if final_expr else []
    if not candidates:
        candidates = _handwriting_final_answer_candidates(recognized_expression)
    if not candidates:
        candidates = [str(recognized_expression).strip()]

    # 在進入複雜邏輯前，先進行穩定格式比較 (字串去白/全半形正規化)
    exp_clean = _clean_math_expr(exp)
    for cand in candidates:
        cand_clean = _clean_math_expr(cand)
        if cand_clean and cand_clean == exp_clean:
            base.update({
                "is_correct": True,
                "status": "correct",
                "main_issue": "",
                "skill_focus": skill_focus or "你已掌握本題重點",
                "issue_tag": "format_matched",
                "analysis_source": "generic_handwriting_analysis",
            })
            return base

    for cand in candidates:
        judged = judge_answer_with_feedback(
            cand, exp, question_text=qt, family_id=family_id
        )
        if judged.get("is_correct"):
            base.update(
                {
                    "is_correct": True,
                    "status": "correct",
                    "main_issue": "",
                    "skill_focus": skill_focus or "你已掌握本題重點",
                    "issue_tag": "",
                    "error_mechanism": "unknown",
                    "step_focus": "",
                    "analysis_source": "generic_handwriting_analysis",
                }
            )
            return base

    primary = candidates[0]
    status = "incorrect"
    main_issue = (
        "這題有錯，請先對照題目關鍵步驟再整理一次。"
    )
    if _handwriting_expr_partially_close(primary, exp):
        status = "partially_correct"
        main_issue = (
            "方向正確，但有一個關鍵小錯需要修正。"
        )

    mechanism = ""
    step_focus = ""
    analysis_src = "generic_handwriting_analysis"
    fid = str(family_id or "").strip()
    ua_full = str(recognized_expression or "").strip()
    if status in ("incorrect", "partially_correct"):
        if fid in ("F9", "poly_div_poly_qr") and _f9_long_division_layout_heuristic(ua_full):
            f9_res = analyze_polynomial_long_division_layout(ua_full, exp, qt)
            mechanism = f9_res.get("error_mechanism") or "notation_error"
            main_issue = f9_res.get("main_issue") or ERROR_MECHANISM_FEEDBACK.get(
                mechanism, ERROR_MECHANISM_FEEDBACK["unknown"]
            )
            step_focus = str(f9_res.get("step_focus") or "")
            analysis_src = "f9_layout_heuristic"
        else:
            mechanism = _handwriting_infer_error_mechanism(
                primary, exp, family_id, question_text=qt
            )
            main_issue = ERROR_MECHANISM_FEEDBACK.get(
                mechanism, ERROR_MECHANISM_FEEDBACK["unknown"]
            )

        if str(mechanism or "").strip() == "unknown" and status in (
            "incorrect",
            "partially_correct",
        ):
            refined = _refine_polynomial_unknown_mechanism(
                fid, ua_full, final_expr, exp, qt
            )
            if refined:
                mechanism = refined["error_mechanism"]
                main_issue = refined["main_issue"]
                step_focus = str(refined.get("step_focus") or "")

    base.update(
        {
            "is_correct": False,
            "status": status,
            "main_issue": main_issue,
            "skill_focus": skill_focus or "依題目重點進行化簡與整理",
            "issue_tag": mechanism,
            "error_mechanism": mechanism,
            "step_focus": step_focus,
            "analysis_source": analysis_src,
        }
    )
    return base


def _handwriting_family_meta(family_id):
    fid = str(family_id or "").strip()
    alias = {
        "poly_add_sub_basic": "F1",
        "poly_add_sub_nested": "F2",
        "poly_mul_monomial": "F3",
        "poly_mul_poly": "F5",
        "poly_div_monomial": "F11",
        "poly_div_poly_qr": "F9",
    }
    fid = alias.get(fid, fid)
    mapping = {
        "F1": ("多項式加減", "同類項合併與符號整理"),
        "F2": ("巢狀多項式加減", "先處理括號與符號，再合併同類項"),
        "F3": ("多項式乘以單項式", "分配律與係數次方正確分配"),
        "F5": ("多項式乘法展開", "每一項都要完整分配相乘"),
        "F9": ("多項式長除法（商餘）", "長除法對齊、乘回與相減"),
        "F11": ("多項式綜合化簡", "包含展開、乘法公式與同類項合併的混合型化簡"),
    }
    return mapping.get(fid, ("一般題型", "請根據題目內容判斷關鍵步驟與常見錯誤"))


_HW_SECOND_STAGE_ERROR_HINT = re.compile(
    r"(錯誤|有誤|不正確|不對|問題|偏差|少了|多了|漏掉|混淆|沒有對|不一致|不符|差在|"
    r"哪裡|這一步|這一列|括號|符號|同類項|合併|化簡|分配|係數|次方|展開|去括號|負號|正負|"
    r"方向正確|大致正確|尚未|需要修正|還沒|還差|不成立|算成|寫成|目前|關鍵小錯|"
    r"推理|化簡|整理|最後一步|前一步|中間|"
    r"代表|表示|意思是|也就是|因此|所以|因為|應該是|常見的誤解|這裡其實|這代表比例|比例應該|"
    r"不是.{0,35}而是)"
)
_HW_SECOND_STAGE_GUIDANCE = re.compile(
    r"(請|建議|下一步|試著|不妨|重新|再試|再算|再寫|再檢查|再核對|再確認|對照|核對|確認|"
    r"逐步|一步一步|逐項|試算|試試|想一下|👉|不妨先|可以先|試著先|先.{0,14}再|"
    r"可以|接著|然後|先看|先把|再把|檢查一下|想想看|重算|重整|化簡一次)"
)


def _hw_second_stage_reply_suggests_final_answer(text: str, expected: str) -> bool:
    """偵測是否像在公布標準答案（避免暴雷）；允許『你目前算成…』類 framing。"""
    exp = _clean_math_expr(str(expected or ""))
    if len(exp) < 5:
        return False
    compact = _clean_math_expr(str(text or ""))
    if exp not in compact:
        return False
    idx = compact.find(exp)
    prefix = compact[max(0, idx - 48) : idx]
    student_frame = any(
        x in prefix
        for x in (
            "你目前",
            "你寫",
            "你算",
            "辨識",
            "這裡寫",
            "寫成",
            "算成",
            "寫的是",
            "得到",
            "變成",
        )
    )
    leak_markers = (
        "答案是",
        "正確答案",
        "最終答案",
        "所以答案",
        "結果應該",
        "應該等於",
        "故得",
        "因此得",
        "正解為",
        "答為",
        "所以為",
        "因此答案",
    )
    window = compact[max(0, idx - 48) : idx + len(exp) + 36]
    if any(m in window for m in leak_markers):
        return True
    if student_frame and not any(m in window for m in leak_markers):
        return False
    if idx + len(exp) >= len(compact) - 2 and len(compact) <= len(exp) + 24:
        return True
    return False


def _handwriting_second_stage_compliance_flags(reply, status, expected_answer=""):
    """
    白板 second-stage：語意合規（不要求固定標題／段數）。
    須同時：指出錯誤或問題點、給出可操作的下一步、且未直接公布完整答案。
    """
    text = str(reply or "").strip()
    st = str(status or "unknown").strip()
    reject_reasons = []
    non_empty = bool(text)
    exp_ans = str(expected_answer or "").strip()

    if st not in ("partially_correct", "incorrect"):
        ok = non_empty
        if not non_empty:
            reject_reasons.append("empty_reply")
        return {
            "compliance_ok": ok,
            "non_empty_reply": non_empty,
            "has_error_explanation": ok,
            "has_guidance": ok,
            "gives_final_answer": False,
            "status": st,
            "reject_reasons": reject_reasons,
        }

    gives_final = _hw_second_stage_reply_suggests_final_answer(text, exp_ans)
    has_err = bool(_HW_SECOND_STAGE_ERROR_HINT.search(text)) and len(text) >= 12
    if (not has_err) and len(text) >= 8 and ("不是" in text and "而是" in text):
        has_err = True
    has_guide = bool(_HW_SECOND_STAGE_GUIDANCE.search(text))

    if not non_empty:
        reject_reasons.append("empty_reply")
    if non_empty and not has_err:
        reject_reasons.append("missing_error_explanation")
    if non_empty and not has_guide:
        reject_reasons.append("missing_guidance")
    if gives_final:
        reject_reasons.append("gave_final_answer")

    compliance_ok = non_empty and has_err and has_guide and not gives_final
    return {
        "compliance_ok": compliance_ok,
        "non_empty_reply": non_empty,
        "has_error_explanation": has_err,
        "has_guidance": has_guide,
        "gives_final_answer": gives_final,
        "status": st,
        "reject_reasons": reject_reasons,
    }


def _handwriting_feedback_second_prompt(
    analysis_result,
    question_text,
    question_context,
    prereq_text,
    family_id,
):
    status = str((analysis_result or {}).get("status") or "unknown")
    question = (question_text or question_context or "").strip()
    student_expression = (analysis_result or {}).get("recognized_expression") or ""
    expected_answer = (analysis_result or {}).get("expected_answer") or ""
    family_description_zh = (analysis_result or {}).get("family_description_zh") or (prereq_text or "")
    main_issue = (analysis_result or {}).get("main_issue") or ""
    error_mechanism = (analysis_result or {}).get("error_mechanism") or ""
    import logging
    logger = logging.getLogger(__name__)
    
    json_format_str = (
        "只輸出 JSON（不要 markdown、不要多餘文字），格式固定為："
        "{\"reply\":\"...您的教學回饋放入這裡...\",\"is_process_correct\":true/false,\"correct\":true/false,"
        "\"next_question\":\"\",\"follow_up_prompts\":[],"
        "\"error_type\":\"handwriting_ok|handwriting_partial|handwriting_wrong|handwriting_unknown\"}。"
    )
    
    try:
        from core.prompts.composer import compose_prompt
        full_prompt, source = compose_prompt(
            base_key=None,
            task_key="handwriting_feedback_prompt",
            extra_blocks=[json_format_str],
            question=question,
            student_expression=student_expression,
            expected_answer=expected_answer,
            status=status,
            family_description_zh=family_description_zh,
            error_mechanism=error_mechanism,
            main_issue=main_issue
        )
        
        logger.info(f"[handwriting second stage] prompt_key=handwriting_feedback_prompt source={source} has_expected_answer={bool(expected_answer)} fallback_to_hardcoded=False")
        return full_prompt
    except Exception as e:
        logger.warning(f"[handwriting second stage] render failed, fallback to hardcoded: {e}")
        logger.info(f"[handwriting second stage] prompt_key=handwriting_feedback_prompt source=fallback has_expected_answer={bool(expected_answer)} fallback_to_hardcoded=True")
        
        return (
            "你是國中數學評量系統的批改回饋模組，不是直接把整題算完的答案機。\n\n"
            "你只根據以下資訊撰寫 JSON 內的 reply 欄位（其餘欄位依語意填寫）：\n\n"
            f"題目：{question}\n"
            f"學生手寫結果：{student_expression}\n"
            f"正確答案：{expected_answer}\n"
            f"判定狀態：{status}\n"
            f"本題重點：{family_description_zh}\n"
            f"錯誤機制：{error_mechanism}\n"
            f"已知錯誤重點：{main_issue}\n\n"
            "請嚴格遵守（輸出僅能是下方指定的 JSON，不要 markdown、不要多餘文字）：\n"
            "- 若 status 為 partially_correct 或 incorrect：reply 必須指出「一個」最關鍵的具體錯誤或偏差"
            "（可說是哪一步、哪種觀念或式子哪一段），並給出「一個」可執行的具體下一步"
            "（例如先如何整理算式、先核對哪一類項），避免空泛只說『再想想』；"
            "優先銜接「已知錯誤重點」與錯誤機制。\n"
            "- 若 status 為 correct：reply 以簡短肯定為主，可鼓勵進入下一題，不須刻意找錯。\n"
            "- 不要直接寫出或暗示題目的最終數值／最終化簡答案；不要替學生把整題解完。\n"
            "- 不要規定固定套語或結尾句型（例如不必強迫寫「請再檢查一次」「想一下再試一次」等）。\n"
            "- 不要限制 reply 行數；語氣簡潔、像老師批改即可；不要聊天。\n\n"
            + json_format_str
        )


def _handwriting_rule_based_reply(analysis_result):
    """白板流程用的 rule-based 回饋（含 second-stage 失敗 fallback）：具體提示＋下一步，不直接給答案。"""
    ar = analysis_result or {}
    st = ar.get("status")
    mech = str(ar.get("error_mechanism") or "").strip()
    issue = ERROR_MECHANISM_FEEDBACK.get(
        mech, str(ar.get("main_issue") or "").strip() or ERROR_MECHANISM_FEEDBACK["unknown"]
    )
    action = ERROR_MECHANISM_ACTION_STEP.get(mech, ERROR_MECHANISM_ACTION_STEP["unknown"])
    expr_raw = str(ar.get("recognized_expression") or ar.get("final_expression") or "").strip()
    expr_show = expr_raw if len(expr_raw) <= 320 else expr_raw[:317] + "…"
    family_hint = str(ar.get("family_label_zh") or "").strip()

    if st == "correct":
        return "✔ 答案正確！\n👉 進入下一題"
    if st == "partially_correct":
        lines = ["⚠️ 前面的整理大致正確，但還有一個關鍵步驟需要修正"]
        if expr_show:
            lines.append(f"你目前辨識到的式子：{expr_show}")
        lines.append(f"錯誤提示：{issue}")
        if family_hint:
            lines.append(f"本題題型重點：{family_hint}")
        lines.append(f"👉 具體下一步：{action}")
        return "\n".join(lines)
    if st == "incorrect":
        lines = ["✘ 這題的推理或化簡仍有需要調整之處"]
        if expr_show:
            lines.append(f"你目前辨識到的式子：{expr_show}")
        lines.append(f"錯誤提示：{issue}")
        if family_hint:
            lines.append(f"本題題型重點：{family_hint}")
        lines.append(f"👉 具體下一步：{action}")
        return "\n".join(lines)
    return "目前我無法穩定判斷你的手寫結果，請確認筆跡清楚或再試一次。"


def _handwriting_issue_is_clear(analysis_result):
    mechanism = str((analysis_result or {}).get("error_mechanism") or "").strip()
    if mechanism in ERROR_MECHANISMS:
        return True
    issue_tag = str((analysis_result or {}).get("issue_tag") or "").strip()
    main_issue = str((analysis_result or {}).get("main_issue") or "").strip()
    if issue_tag:
        return True
    if len(main_issue) < 10:
        return False
    generic = (
        "請檢查",
        "再看看",
        "注意運算順序",
        "檢查係數",
        "檢查符號",
    )
    return not any(g in main_issue for g in generic)


TUTOR_FORBIDDEN_PHRASES = (
    "你錯了", "你對了", "有道理", "不對", "很棒", "沒關係", "再試試",
    "答案是", "我覺得", "我是AI",
)


def _tutor_extract_structured_analysis(data):
    """Prefer explicit structured_analysis; else handwriting_analysis; else flat request fields."""
    if not isinstance(data, dict):
        data = {}
    raw_sa = data.get("structured_analysis")
    raw_ha = data.get("handwriting_analysis")
    if isinstance(raw_sa, dict) and raw_sa:
        src = dict(raw_sa)
    elif isinstance(raw_ha, dict) and raw_ha:
        src = dict(raw_ha)
    else:
        src = {}
    return {
        "status": str(src.get("status") or data.get("status") or "unknown"),
        "main_issue": str(src.get("main_issue") or data.get("main_issue") or ""),
        "issue_tag": str(src.get("issue_tag") or data.get("issue_tag") or ""),
        "expected_answer": str(src.get("expected_answer") or data.get("expected_answer") or ""),
        "error_mechanism": str(src.get("error_mechanism") or data.get("error_mechanism") or "unknown"),
        "step_focus": str(src.get("step_focus") or data.get("step_focus") or ""),
        "family_id": str(src.get("family_id") or data.get("family_id") or ""),
        "analysis_source": str(src.get("analysis_source") or data.get("analysis_source") or ""),
    }


def _tutor_deterministic_fallback(structured_analysis):
    mech = str(structured_analysis.get("error_mechanism") or "unknown")
    issue = str(structured_analysis.get("main_issue") or "")
    fid = str(structured_analysis.get("family_id") or "").strip()
    step = str(structured_analysis.get("step_focus") or "").strip()
    if fid in ("F9", "poly_div_poly_qr"):
        picked = _f9_tutor_guidance_pick(mech, step)
        if picked:
            hint_focus, guided_question, micro_step = picked
            return {
                "hint_focus": hint_focus[:18],
                "guided_question": guided_question[:28],
                "micro_step": micro_step[:22],
                "forbidden": False,
            }
    by_mech = {
        "sign_error": (
            "先看括號前的符號",
            "去括號後每一項符號有變嗎？",
            "先把括號內每項重寫一次",
        ),
        "combine_error": (
            "先找同類項",
            "哪些項的字母和次方完全一樣？",
            "先把可合併的項圈出來",
        ),
        "structure_error": (
            "先確認式子結構",
            "這一步有沒有改變原本括號範圍？",
            "先逐行對照原式重寫",
        ),
        "operation_error": (
            "先確認運算規則",
            "這一步是加減還是分配展開？",
            "先只做一個運算動作",
        ),
        "substitution_error": (
            "先核對代入值",
            "每個代入值放到正確位置了嗎？",
            "先把代入位置標記出來",
        ),
        "notation_error": (
            "先補齊記號格式",
            "括號、變數、次方都寫完整了嗎？",
            "先修正式子書寫格式",
        ),
        "unknown": (
            "先回到題目關鍵條件",
            "哪一步開始與題目條件不一致？",
            "先重寫第一步再往下",
        ),
    }
    hint_focus, guided_question, micro_step = by_mech.get(mech, by_mech["unknown"])
    if issue and mech == "unknown":
        hint_focus = issue[:18]
    return {
        "hint_focus": hint_focus[:18],
        "guided_question": guided_question[:28],
        "micro_step": micro_step[:22],
        "forbidden": False,
    }


def _tutor_guidance_to_reply(g):
    return f"這題的關鍵是：{g['hint_focus']}\n\n想一下：\n{g['guided_question']}\n\n👉 {g['micro_step']}"


def _tutor_has_answer_leak(text):
    t = str(text or "")
    if "答案是" in t:
        return True
    # very simple leak check: direct assignment pattern in tutor text
    return bool(re.search(r"[a-zA-Z]\s*=\s*[-+]?\d", t))


def _tutor_guidance_is_compliant(g, structured_analysis):
    if not isinstance(g, dict):
        return False
    for k in ("hint_focus", "guided_question", "micro_step", "forbidden"):
        if k not in g:
            return False
    if bool(g.get("forbidden")):
        return False
    hint_focus = str(g.get("hint_focus") or "").strip()
    guided_question = str(g.get("guided_question") or "").strip()
    micro_step = str(g.get("micro_step") or "").strip()
    if (not hint_focus) or (not guided_question) or (not micro_step):
        return False
    if len(hint_focus) > 18 or len(guided_question) > 28 or len(micro_step) > 22:
        return False
    merged = f"{hint_focus} {guided_question} {micro_step}"
    if any(p in merged for p in TUTOR_FORBIDDEN_PHRASES):
        return False
    if _tutor_has_answer_leak(merged):
        return False
    status = str(structured_analysis.get("status") or "")
    if status in ("correct", "incorrect", "partially_correct"):
        banned_judge = ("答對", "答錯", "正確", "錯誤")
        if any(p in merged for p in banned_judge):
            return False
    main_issue = str(structured_analysis.get("main_issue") or "").strip()
    if main_issue and (main_issue[:4] not in merged) and (main_issue[:6] not in merged):
        # not strictly required but should stay aligned with truth source
        key_tokens = [w for w in re.findall(r"[\u4e00-\u9fff]{2,}", main_issue) if len(w) >= 2]
        if key_tokens and not any(tok in merged for tok in key_tokens[:2]):
            return False
    return True

def allowed_exam_file(filename):







    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXAM_EXTENSIONS















# ==========================================







# AI Chat & Handwriting（AI 對話與手寫）







# ==========================================















@practice_bp.route('/chat_ai', methods=['POST'])







def chat_ai():







    """AI chat API."""







    data = request.get_json()







    user_question = data.get('question', '').strip()







    context = data.get('context', '')







    question_text = data.get('question_text', '')
    correct_answer = data.get('correct_answer', '').strip()







    family_id = (data.get('family_id') or '').strip()







    question_context = (data.get('question_context') or context or '').strip()







    subskill_nodes = data.get('subskill_nodes') or []















    if not user_question:







        return jsonify({"reply": "缺少必要參數：question"}), 400















    current = get_current()







    skill_id = current.get("skill")







    prereq_skills = current.get('prereq_skills', [])







    if not family_id:







        family_id = (current.get('family_id') or current.get('skill') or skill_id or '').strip()







    







    full_question_context = question_text if question_text else (current.get("question") or question_context)
    structured_analysis = _tutor_extract_structured_analysis(data)
    _raw_ha = data.get("handwriting_analysis") if isinstance(data, dict) else None
    _has_handwriting = isinstance(_raw_ha, dict) and bool(_raw_ha)
    _sa_empty = (
        str(structured_analysis.get("status") or "") == "unknown"
        and not str(structured_analysis.get("main_issue") or "").strip()
        and str(structured_analysis.get("error_mechanism") or "") == "unknown"
    )
    payload_keys = list(data.keys()) if isinstance(data, dict) else []
    current_app.logger.info(f"[chat_ai] payload keys={payload_keys}")

    if not _has_handwriting and _sa_empty:
        current_app.logger.info("[chat_ai] structured_analysis empty, using fallback")
        structured_analysis = {
            "status": "unknown",
            "main_issue": user_question,
            "issue_tag": "",
            "expected_answer": "",
            "error_mechanism": "unknown",
            "step_focus": "",
            "family_id": family_id,
            "analysis_source": "fallback",
            "question_text": full_question_context,
            "student_message": user_question,
            "analysis": ""
        }
        current_app.logger.info("[chat_ai] calling AI with fallback context")
    structured_summary = (
        f"status={structured_analysis.get('status','unknown')}; "
        f"main_issue={structured_analysis.get('main_issue','')}; "
        f"issue_tag={structured_analysis.get('issue_tag','')}; "
        f"error_mechanism={structured_analysis.get('error_mechanism','unknown')}; "
        f"step_focus={structured_analysis.get('step_focus','')}; "
        f"family_id={structured_analysis.get('family_id','')}; "
        f"analysis_source={structured_analysis.get('analysis_source','')}; "
        f"expected_answer={structured_analysis.get('expected_answer','')}"
    )
    if structured_summary:
        full_question_context = f"{full_question_context}\n\n[structured_analysis]\n{structured_summary}"







    







    if not skill_id and context:







        # 依內文關鍵字推測 skill_id（Fallback）







        if '餘式' in context: skill_id = 'remainder'







        elif '因式' in context: skill_id = 'factor_theorem'















    full_prompt = build_chat_prompt(







        skill_id=skill_id,







        user_question=user_question,







        full_question_context=full_question_context,







        context=context,
        prereq_skills=prereq_skills,
        correct_answer=correct_answer
    )














    current_app.logger.info(
        "[chat_ai] analysis_source=%s family_id=%s error_mechanism=%s step_focus=%s",
        structured_analysis.get("analysis_source", ""),
        structured_analysis.get("family_id", ""),
        structured_analysis.get("error_mechanism", ""),
        structured_analysis.get("step_focus", ""),
    )
    current_app.logger.info(







        "[chat_ai] received question='%s' context_head='%s'",







        user_question[:120],







        (full_question_context or '')[:120]







    )







    







    result = get_chat_response(







        full_prompt,







        user_question=user_question,







        question_context=full_question_context







    )

    # Tutor compliance gate: bounded language layer only.
    structured_analysis = _tutor_extract_structured_analysis(data)
    
    if not isinstance(result, dict) or not result:
        current_app.logger.info("[chat_ai] tutor output non-compliant; fallback to deterministic guidance")
        guidance = _tutor_deterministic_fallback(structured_analysis)
    else:
        guidance = {
            "hint_focus": str(result.get("hint_focus") or "").strip(),
            "guided_question": str(result.get("guided_question") or "").strip(),
            "micro_step": str(result.get("micro_step") or "").strip(),
            "forbidden": result.get("forbidden") if "forbidden" in result else False
        }
        
        has_focus = bool(guidance["hint_focus"])
        has_question = bool(guidance["guided_question"])
        has_step = bool(guidance["micro_step"])
        
        if not has_focus and not has_question and not has_step:
            current_app.logger.info("[chat_ai] tutor output non-compliant; fallback to deterministic guidance")
            guidance = _tutor_deterministic_fallback(structured_analysis)
        else:
            current_app.logger.info("[chat_ai] tutor output partial but accepted")
            if not has_focus or not has_question or not has_step:
                current_app.logger.info("[chat_ai] tutor output missing fields auto-filled")
                fb = _tutor_deterministic_fallback(structured_analysis)
                if not has_focus: guidance["hint_focus"] = fb["hint_focus"]
                if not has_question: guidance["guided_question"] = fb["guided_question"]
                if not has_step: guidance["micro_step"] = fb["micro_step"]

    result = result if isinstance(result, dict) else {}
    result["hint_focus"] = guidance["hint_focus"]
    result["guided_question"] = guidance["guided_question"]
    result["micro_step"] = guidance["micro_step"]
    result["forbidden"] = False
    result["reply"] = _tutor_guidance_to_reply(guidance)















    # 讀寫 session 中的延伸問答狀態（與題目脈絡綁定）。







    chat_state = session.get('chat_followup_state', {}) if isinstance(session.get('chat_followup_state', {}), dict) else {}







    last_prompts = chat_state.get('last_prompts', [])







    turn_index = int(chat_state.get('turn_index', 0))







    last_context = chat_state.get('last_context', '')















    if last_context != full_question_context:







        last_prompts = []







        turn_index = 0















    # 題目脈絡變更時重置回合索引；依序產生本回合的後續引導題組。







    per_turn_prompts = build_dynamic_follow_up_prompts_variant(







        user_question=user_question,







        question_context=full_question_context,







        ai_reply=result.get('reply', ''),







        variant=turn_index







    )







    result['follow_up_prompts'] = diversify_follow_up_prompts(







        per_turn_prompts,







        last_prompts,







        user_question=user_question,







        question_context=full_question_context,







        ai_reply=result.get('reply', ''),







        turn_index=turn_index







    )















    current_app.logger.info(







        "[chat_ai] generated follow_up_prompts=%s",







        result.get('follow_up_prompts', [])







    )















    session['chat_followup_state'] = {







        'last_prompts': result.get('follow_up_prompts', [])[:3],







        'turn_index': turn_index + 1,







        'last_question': user_question[:120],







        'last_context': full_question_context







    }















    subskill_map = {







        'sign_handling': '符號處理',







        'add_sub': '加減運算',







        'mul_div': '乘除運算',







        'mixed_ops': '四則混合',







        'absolute_value': '絕對值',







        'parentheses': '括號運算',







        'divide_terms': '分項除法',







        'conjugate_rationalize': '有理化',







    }







    focus_points = []







    for item in subskill_nodes:







        label = subskill_map.get(str(item), str(item).replace('_', ' '))







        if label and label not in focus_points:







            focus_points.append(label)







    if not focus_points:







        focus_points.append('依題目關鍵步驟進行檢查')







    result['subskill_labels'] = focus_points















    result['question_text'] = question_text







    result['family_id'] = family_id







    result['question_context'] = question_context or question_text















    return jsonify(result)















@practice_bp.route('/analyze_handwriting', methods=['POST'])







@login_required







def analyze_handwriting():







    """Student diagnosis page."""







    data = request.get_json(silent=True) or {}







    print("payload keys:", list(data.keys()))
    print("has image:", bool(data.get("image_base64")))
    if data.get("image_base64"):
        print("image prefix:", str(data["image_base64"])[:80])
    img = (
        data.get('image_data_url')
        or data.get('image_base64')
        or data.get('handwriting_image')
        or data.get('scratchpad_image')
    )
    print("has any image field:", bool(img))
    if img:
        print("effective image prefix:", str(img)[:80])







    if not img:
        return _handwriting_not_recognized_response()







    







    state = get_current()







    prereq_skills = state.get('prereq_skills', [])
    ai_provider = (data.get('provider') or 'local').strip().lower()
    if ai_provider not in ('local', 'google'):
        ai_provider = 'local'
    question_text = (data.get('question_text') or state.get('question_text') or state.get('question') or "").strip()
    question_context = (data.get('question_context') or state.get('question') or "").strip()
    family_id = (data.get('family_id') or state.get('family_id') or state.get('skill') or "").strip()
    skill_labels = {
        'sign_handling': '符號處理',
        'add_sub': '加減運算',
        'mul_div': '乘除運算',
        'mixed_ops': '四則混合',
        'absolute_value': '絕對值',
        'parentheses': '括號運算',
        'divide_terms': '分項除法',
        'conjugate_rationalize': '有理化',
    }
    prereq_text = ", ".join(
        f"{skill_labels.get(item.get('id'), item.get('name', item.get('id', '')))}"
        for item in prereq_skills
        if isinstance(item, dict)
    )
    b64 = str(img)
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip()
    temp_path = None
    result = None
    try:
        try:
            img_data = base64.b64decode(b64)
        except Exception:
            return _handwriting_not_recognized_response()
        if not img_data:
            return _handwriting_not_recognized_response()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            f.write(img_data)
            temp_path = f.name
        print("calling handwriting analyzer (recognition then analysis)")
        print("question_text:", question_text)
        print("image length:", len(b64) if b64 else 0)
        recognition_prompt = (
            "Transcribe all visible handwritten mathematical content from this student whiteboard image. "
            "Return ONLY valid JSON, no markdown, no extra text. "
            "The JSON must be one object with key \"expression\" (string). "
            "Put equations, numbers, and step-by-step work in that string; separate lines with semicolons. "
            "If the image is blank, illegible, or has no mathematical writing, set \"expression\" to \"\". "
            "Do not invent symbols or steps that are not clearly visible."
        )
        vision_cfg = get_effective_model_config("vision_analyzer")
        vision_provider = str(vision_cfg.get("provider", "")).strip().lower()
        if ai_provider == 'google' and vision_provider in ('google', 'gemini'):
            rec_response = call_google_model(
                vision_cfg,
                recognition_prompt,
                image_path=temp_path,
                max_retries=2,
                retry_delay=1,
                verbose=False,
            )
        else:
            if ai_provider == 'google' and vision_provider not in ('google', 'gemini'):
                current_app.logger.info(
                    "analyze_handwriting: provider override=google ignored because vision_analyzer runtime provider=%s",
                    vision_provider or "unknown",
                )
            rec_response = call_ai(
                role="vision_analyzer",
                prompt=recognition_prompt,
                image_path=temp_path,
                max_retries=2,
                retry_delay=1,
                verbose=False,
            )
        rec_raw = (getattr(rec_response, 'text', '') or '').strip()
        rec_cleaned = re.sub(r'^```json\s*|\s*```$', '', rec_raw, flags=re.MULTILINE)
        rec_parsed = clean_and_parse_json(rec_cleaned)
        expr = _handwriting_expression_from_parsed(rec_parsed)
        if not _handwriting_expression_is_usable(expr):
            current_app.logger.info("analyze_handwriting: no usable expression after recognition pass")
            return _handwriting_not_recognized_response()

        expected_answer = _handwriting_expected_answer(data, state, question_text, question_context)
        
        current_app.logger.info(
            f"[Handwriting Grade Debug] \n"
            f" - current_question: {question_text}\n"
            f" - expected_answer: {expected_answer}\n"
            f" - recognized_expression: {expr}\n"
            f" - final grading input: (expr='{expr}', expected_answer='{expected_answer}', question_text='{question_text}', family_id='{family_id}')"
        )
        
        analysis_result = _handwriting_structured_analysis(
            expr, expected_answer, question_text, family_id
        )
        current_app.logger.info(
            "analyze_handwriting: analysis_source=%s family_id=%s status=%s",
            analysis_result.get("analysis_source"),
            family_id,
            analysis_result.get("status"),
        )
        _hw_err = {
            "correct": "handwriting_ok",
            "partially_correct": "handwriting_partial",
            "incorrect": "handwriting_wrong",
            "unknown": "handwriting_unknown",
        }

        hw_status = analysis_result.get("status")
        if hw_status in ("unknown", "correct"):
            reply_text = _handwriting_rule_based_reply(analysis_result)
            is_hw_correct = hw_status == "correct"
            result = {
                "reply": enforce_strict_mode(reply_text),
                "is_process_correct": hw_status in ("correct", "partially_correct"),
                "correct": is_hw_correct,
                # 與 /check_answer 語意對齊：答對時為 True，供前端與 textbox 相同流程判斷
                "next_question": is_hw_correct,
                "follow_up_prompts": [],
                "error_type": _hw_err.get(hw_status, "handwriting_unknown"),
                "success": True,
                "auto_next": is_hw_correct,
                "handwriting_analysis": analysis_result,
                "handwriting_status": hw_status,
            }
            print("analyzer result:", result)
        else:
            current_app.logger.info(f"[analyze_handwriting] routing_to_second_stage status={hw_status}")
            prompt = _handwriting_feedback_second_prompt(
                analysis_result, question_text, question_context, prereq_text, family_id
            )
            tutor_cfg = get_effective_model_config("tutor")
            tutor_provider = str(tutor_cfg.get("provider", "")).strip().lower()
            if ai_provider == 'google' and tutor_provider in ('google', 'gemini'):
                response = call_google_model(
                    tutor_cfg,
                    prompt,
                    image_path=None,
                    max_retries=2,
                    retry_delay=1,
                    verbose=False,
                )
            else:
                if ai_provider == 'google' and tutor_provider not in ('google', 'gemini'):
                    current_app.logger.info(
                        "analyze_handwriting: provider override=google ignored because tutor runtime provider=%s",
                        tutor_provider or "unknown",
                    )
                response = call_ai(
                    role="tutor",
                    prompt=prompt,
                    image_path=None,
                    max_retries=2,
                    retry_delay=1,
                    verbose=False,
                )
            raw_text = (getattr(response, "text", "") or "").strip()
            cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE)
            parsed = clean_and_parse_json(cleaned)
            reply_stripped = ""
            if isinstance(parsed, dict):
                reply_stripped = str(parsed.get("reply", "") or "").strip()

            current_app.logger.info(
                "[handwriting second stage] raw reply start\n%s\n[handwriting second stage] raw reply end",
                raw_text,
            )
            current_app.logger.info(
                "[handwriting second stage] meta prompt_key=handwriting_feedback_prompt status=%s "
                "raw_total_len=%s reply_field_len=%s",
                hw_status,
                len(raw_text or ""),
                len(reply_stripped or ""),
            )

            _hw_exp = str((analysis_result or {}).get("expected_answer") or "")
            _hw_flags = _handwriting_second_stage_compliance_flags(
                reply_stripped, analysis_result.get("status"), _hw_exp
            )
            _hw_ok = _hw_flags.get("compliance_ok")
            current_app.logger.info(
                "[handwriting second stage] compliance flags=%r",
                _hw_flags,
            )

            if (not reply_stripped) or (not _hw_ok):
                if _hw_flags.get("reject_reasons"):
                    current_app.logger.info(
                        "[handwriting second stage] rejected: %s; using rule-based reply",
                        "; ".join(_hw_flags["reject_reasons"]),
                    )
                else:
                    current_app.logger.info(
                        "[handwriting second stage] rejected: unknown; using rule-based reply"
                    )
                reply_stripped = _handwriting_rule_based_reply(analysis_result)
            else:
                current_app.logger.info(
                    "[handwriting second stage] accepted (semantic compliance)"
                )
            result = {
                "reply": "目前無法產生穩定批改回饋，請再試一次。",
                "is_process_correct": False,
                "correct": False,
                "next_question": "",
                "follow_up_prompts": [],
                "error_type": "handwriting_analyzer_failed",
            }
            if isinstance(parsed, dict):
                result.update(parsed)
            result["reply"] = enforce_strict_mode(reply_stripped)
            result["handwriting_analysis"] = analysis_result
            result["handwriting_status"] = analysis_result.get("status", "unknown")
            st = analysis_result.get("status")
            result["correct"] = st == "correct"
            result["is_process_correct"] = st in ("correct", "partially_correct")
            result["error_type"] = _hw_err.get(st, "handwriting_unknown")
            result["success"] = True
            if result.get("correct"):
                result["auto_next"] = True
                result["next_question"] = True
            else:
                result["auto_next"] = False
                result["next_question"] = False
            if not isinstance(result.get("follow_up_prompts"), list):
                result["follow_up_prompts"] = []
            print("analyzer result:", result)
    except Exception as analyzer_exc:
        current_app.logger.exception("analyze_handwriting analyzer failed: %s", analyzer_exc)
        return _handwriting_not_recognized_response()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)















    # [Start of modification] Diagnosis for handwriting







    if False and result.get("correct") is False and result.get("handwriting_status") == "incorrect":







        try:







            current_app.logger.info(f"[Handwriting Check] Incorrect answer detected, diagnosing...")







            question_text = state.get('question_text', '')







            correct_answer = state.get('correct_answer', '')







            student_answer = f"【手寫批改回覆】：{result.get('reply', '')}"







            







            # [Phase 6] Diagnosis







            diagnosis = diagnose_error(







                question_text=question_text,







                correct_answer=correct_answer,







                student_answer=student_answer,







                prerequisite_units=prereq_skills







            )







            







            if diagnosis.get('related_prerequisite_id'):







                prereq_id = diagnosis['related_prerequisite_id']







                # Find name







                prereq_name = "未知先備課題"







                if prereq_skills:







                    for p in prereq_skills:







                        # prereq_skills structure: [{'id':..., 'name':...}]







                        if str(p.get('id')) == str(prereq_id) or str(p.get('skill_id')) == str(prereq_id):







                            prereq_name = p.get('name') or p.get('skill_ch_name')







                            break







                







                result['suggested_prerequisite'] = {







                    'id': prereq_id,







                    'name': prereq_name,







                    'reason': diagnosis.get('prerequisite_explanation', '建議先複習相關先備概念。')







                }







        except Exception as e:







            current_app.logger.error(f"Handwriting diagnosis failed: {e}")







    # [End of modification]







    







    # 進度更新交由其他路由／helper；此處不直接呼叫 update_progress







    







    return jsonify(result)































# ==========================================







# Mistake Notebook & Diagnosis（錯題本與診斷）







# ==========================================















@core_bp.route('/mistake-notebook')







@login_required







def mistake_notebook():







    return render_template('mistake_notebook.html', username=current_user.username)















@core_bp.route('/api/mistake-notebook', methods=['GET'])







@login_required







def api_mistake_notebook():







    entries = db.session.query(MistakeNotebookEntry).filter_by(student_id=current_user.id).order_by(MistakeNotebookEntry.created_at.desc()).all()







    return jsonify([entry.to_dict() for entry in entries])















@core_bp.route('/mistake-notebook/add', methods=['POST'])







@login_required







def add_mistake_entry():







    try:







        data = request.get_json()







        db.session.add(MistakeNotebookEntry(







            student_id=current_user.id,







            exam_image_path=data.get('exam_image_path'),







            question_data=data.get('question_data'),







            notes=data.get('notes'),







            skill_id=data.get('skill_id')







        ))







        db.session.commit()







        return jsonify({'success': True, 'message': '已儲存'})







    except Exception as e:







        db.session.rollback()







        return jsonify({'success': False, 'message': str(e)}), 500















@core_bp.route('/student/analyze_weakness', methods=['POST'])







@login_required







def analyze_weakness():







    """Analyze student weakness data for radar chart."""







    try:







        force_refresh = request.json.get('force_refresh', False) if request.json else False







        result = perform_weakness_analysis(current_user.id, force_refresh)







        return jsonify(result)







    except Exception as e:







        current_app.logger.error(f"錯題本寫入失敗: {e}")







        return jsonify({'success': False, 'error': str(e)}), 500















# ==========================================







# Mistake Notebook 相關輔助函式







# ==========================================















@core_bp.route('/add_mistake_page')







@login_required







def add_mistake_page():







    """Render the mistake notebook page."""







    skills = db.session.query(SkillInfo).filter_by(is_active=True).order_by(SkillInfo.skill_ch_name).all()







    return render_template('add_mistake.html', skills=skills, username=current_user.username)















@core_bp.route('/mistake-notebook/upload-image', methods=['POST'])







@login_required







def upload_mistake_image():







    """Upload a mistake notebook image."""







    if 'file' not in request.files: return jsonify({'success': False, 'message': '未找到上傳檔案'}), 400







    file = request.files['file']







    if file.filename == '' or not allowed_exam_file(file.filename):







        return jsonify({'success': False, 'message': '不允許的檔案類型'}), 400















    try:







        upload_dir = os.path.join(current_app.static_folder, 'mistake_uploads', str(current_user.id))







        os.makedirs(upload_dir, exist_ok=True)







        







        filename = secure_filename(file.filename)







        unique_filename = f"{uuid.uuid4().hex}_{filename}"







        path = os.path.join(upload_dir, unique_filename)







        file.save(path)







        







        rel_path = os.path.join('mistake_uploads', str(current_user.id), unique_filename).replace('\\', '/')







        return jsonify({'success': True, 'path': rel_path})







    except Exception as e:







        return jsonify({'success': False, 'message': str(e)}), 500















# ==========================================







# Student Diagnosis（學生診斷頁）







# ==========================================















@core_bp.route('/student/diagnosis')







@login_required







def student_diagnosis():







    """顯示學生診斷（學習落點）頁面。"""







    return render_template('student_diagnosis.html', username=current_user.username)















# ==========================================







# Naive RAG（簡易檢索 + LLM 彙整）







# ==========================================







# RAG 查詢與進階 RAG







# ==========================================















@practice_bp.route('/api/rag_search', methods=['POST'])







def api_rag_search():







    """RAG search API."""







    data = request.get_json(silent=True) or {}







    query = (data.get('query') or '').strip()















    if not query:







        return jsonify({"results": [], "error": "查詢字串不可為空"}), 400















    results = rag_search(query, top_k=5)







    return jsonify({"results": results})























@practice_bp.route('/api/rag_chat', methods=['POST'])







def api_rag_chat():







    """RAG + LLM chat API."""







    data = request.get_json(silent=True) or {}







    query = (data.get('query') or '').strip()







    top_skill_id = (data.get('top_skill_id') or '').strip()















    if not query or not top_skill_id:







        return jsonify({"reply": "查詢或技能 ID 不可為空"}), 400















    try:
        from core.advanced_rag_engine import HAS_ADV_LIBS
        import logging
        logger = logging.getLogger(__name__)
        
        if HAS_ADV_LIBS:
            logger.info("[API RAG CHAT] Routing to Advanced RAG Engine.")
            from core.advanced_rag_engine import adv_rag_search, adv_rag_chat
            from core.ai_settings import get_effective_model_config
            
            cfg = get_effective_model_config(role="tutor")
            provider = cfg.get("provider", "local")
            
            skills = adv_rag_search(query, top_k=3)
            result = adv_rag_chat(query, retrieved_skills=skills, provider=provider, family_id=top_skill_id)
            return jsonify(result)
        else:
            logger.info("[API RAG CHAT] HAS_ADV_LIBS is False. Routing to Naive RAG.")
    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).warning(f"Error checking advanced RAG, fallback directly to naive: {e}\n{traceback.format_exc()}")

    result = rag_chat(query, top_skill_id)
    return jsonify(result)









