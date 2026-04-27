"""
═════════════════════════════════════════════════════════════════════════════
自適應複習模式 - Flask API 集成層
═════════════════════════════════════════════════════════════════════════════

提供 REST API 端點，將自適應複習功能整合到 Flask 應用中。

API 端點：
1. POST /api/adaptive-review/start
   - 開始新的複習會話（若前端 history 為空，從 DB 恢復）
2. GET  /api/adaptive-review/analyze/<student_id>
   - 分析學生弱項
3. POST /api/adaptive-review/feedback
   - 提交答題反饋（LLM 評分、NodeCompetency 更新、歷史持久化）
4. GET  /api/adaptive-review/question/<item_id>
   - 取得題目（包含單題截斷 + LLM 答案格式提示）
5. POST /api/adaptive-review/chat
   - AI 引導式提示
6. GET  /api/adaptive-review/health
   - 健康檢查
"""

from flask import Blueprint, request, jsonify
from functools import lru_cache
import json
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import sympy as sp
from sympy import latex as sympy_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from core.ai_wrapper import get_ai_client, call_ai_with_retry
from core.prompts.composer import compose_prompt

# ═══════════════════════════════════════════════════════════════════════════
# 集中式 Prompt 管理
# ═══════════════════════════════════════════════════════════════════════════

LLM_GUIDE_PROMPT = """你是一個耐心、鼓勵學生的數學 AI 家教。
這是一道關於【{skill}】的題目：
{question}

學生目前的疑問或回答：
{query}

請遵守：
1. 不可直接給出最終答案或完整解法。
2. 先點出一個核心觀念，再給 1 到 2 個可執行的小步驟。
3. 若學生卡住，可補一個很短的例子，但不能直接代入本題得出答案。
4. 最後一定要用一個引導問題收尾。
5. 回覆比一句話更完整，但仍保持精煉，約 4 到 8 行。
"""

LLM_ANSWER_FORMAT_PROMPT = """根據以下這道數學題，用少於 20 字的中文提示學生應該怎麼作答（重點說明需要填的提入內容形式，例如「填入一個整數」、「寫出一元二次方程式」）。不要告訴答案，只說明格式。

題目：{question}

答案格式提示："""

# ── 子題前綴清洗（⑴⑵①② (1)(2) 1. 等） ────────────────────────────────
_SUBPROBLEM_PREFIX_RE = re.compile(
    r'^[\s]*'
    r'(?:[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽①②③④⑤⑥⑦⑧⑨⑩]'
    r'|\([0-9]+\)'
    r'|（[0-9]+）'
    r'|[0-9]+[\u3002.、]'
    r')[\s]*'
)


def _strip_subproblem_prefix(text: str) -> str:
    """移除答案字串開頭的小題標記（⑴ ① (1)（1）1. 等），並 strip 結果。"""
    if not text:
        return text
    return _SUBPROBLEM_PREFIX_RE.sub('', text, count=1).strip()


# 規則式答案格式提示萃取（避免每次呼叫 LLM）
def _rule_based_format_hint(question_text: str) -> str:
    """從題目文字用規則萃取答案格式提示。無法判斷時回傳空字串。"""
    import re
    q = question_text.lower().strip()
    # 題目本身已含答案格式提示
    m = re.search(r'[（(]答案格式[：:]([^）)]{1,30})[）)]', question_text)
    if m:
        return f'請輸入：{m.group(1).strip()}'
    # 座標類
    if re.search(r'座標|坐標|交點|頂點|A\(|B\(', question_text):
        return '請輸入座標，如 (x, y)'
    # 聯立方程式 / 解
    if re.search(r'聯立|消去|代入法|elimination|substitution', q):
        return '請輸入 x 和 y 的值，如 x=2, y=3'
    # 不等式
    if re.search(r'不等式|inequality|解集', q):
        return '請輸入不等式解，如 x > 2'
    # 因式分解
    if re.search(r'因式分解|factori', q):
        return '請輸入因式分解結果，如 (x+1)(x-2)'
    # 多個數字
    if re.search(r'分別|各為|各是|列出所有', question_text):
        return '請輸入所有答案，用逗號分隔，如 1, 2, 3'
    # 根號 / 根式
    if re.search(r'根號|√|平方根|radical', q):
        return '請輸入根式或數字，如 2√3 或 6'
    # 方程式
    if re.search(r'方程|equation|求.*x', q):
        return '請輸入 x 的值（數字或分數，如 3 或 1/2）'
    # 機率
    if re.search(r'機率|probability', q):
        return '請輸入機率（分數或小數，如 1/4 或 0.25）'
    # 面積 / 體積
    if re.search(r'面積|體積|周長|area|volume|perimeter', q):
        return '請輸入數字（含單位如 cm²）'
    return '請輸入你的答案（若是數字請直接填寫整數、小數或分數）'  # 預設格式提示，避免回傳空字串觸發 fallback

# ── 規則式評分引擎（取代 LLM，速度從秒級 → 毫秒級） ──────────────────────
def _parse_number(s: str):
    """將字串解析為 float。支援整數、小數、分數（如 1/2）。失敗回傳 None。"""
    s = s.strip().replace('，', ',').replace(' ', '')
    try:
        return float(s)
    except ValueError:
        pass
    # 分數
    frac = re.match(r'^(-?\d+)/(-?\d+)$', s)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den != 0:
            return num / den
    return None

def _normalize_token(t: str) -> str:
    """把單一 token 正規化為可比較的字串。"""
    t = t.strip()
    # 去除座標包裝括號 A(3,5) / (3,5)
    t = re.sub(r'^[A-Za-z]*\((.+)\)$', r'\1', t)
    # 數字統一為小數字串（精度 6 位）
    n = _parse_number(t)
    if n is not None:
        return f'{n:.6g}'
    # 去除空格、大小寫統一
    return t.lower().replace(' ', '')

def _split_answers(s: str) -> list:
    """將多答案字串拆成 token 列表（支援逗號/分號/空格分隔）。"""
    # 先把全形逗號轉半形
    s = s.replace('，', ',').replace('；', ';')
    # 以逗號或分號分割
    parts = re.split(r'[,;]+', s)
    tokens = [p.strip() for p in parts if p.strip()]
    return tokens

def _rule_based_grade(correct_answer: str, student_answer: str) -> tuple:
    """
    規則式評分，回傳 (is_correct: bool, detail: str)。
    支援：
      - 精確字串比對（大小寫不敏感）
      - 數字 / 分數 / 小數等價（1/2 == 0.5）
      - 多答案逗號列表（順序不敏感）
      - 座標 (x, y) / A(x, y)
      - 不等式（x > 2, x<=3 等）
      - 因式分解括號表達式（簡單比對符號去除空格後）
      - 子題標記前綴自動去除（⑴⑵①② (1)(2) 等）
    """
    # 移除小題標記前綴（⑴ 0 → 0）
    ca = _strip_subproblem_prefix(correct_answer.strip())
    sa = _strip_subproblem_prefix(student_answer.strip())
    if not ca:
        return True, '✅ 已記錄（無標準答案）'
    if not sa:
        return False, '❌ 未輸入答案'

    # 1. 精確比對（大小寫不敏感）
    if ca.lower() == sa.lower():
        return True, '✅ 答對了！'

    # 2. 數字等價比對（支援分數↔小數）
    ca_num = _parse_number(ca)
    sa_num = _parse_number(sa)
    if ca_num is not None and sa_num is not None:
        if abs(ca_num - sa_num) < 1e-6:
            return True, '✅ 答對了！'
        else:
            return False, f'❌ 答案不正確（正確：{ca}）'

    # 3. 多答案列表比對（順序不敏感）
    ca_parts = _split_answers(ca)
    sa_parts = _split_answers(sa)
    if len(ca_parts) > 1 or len(sa_parts) > 1:
        ca_norm = sorted([_normalize_token(t) for t in ca_parts])
        sa_norm = sorted([_normalize_token(t) for t in sa_parts])
        if ca_norm == sa_norm:
            return True, '✅ 答對了！'
        # 嘗試數字等價（列表中每個元素）
        if len(ca_norm) == len(sa_norm):
            all_match = True
            for c, s in zip(sorted(ca_parts), sorted(sa_parts)):
                cn, sn = _parse_number(_normalize_token(c)), _parse_number(_normalize_token(s))
                if cn is None or sn is None or abs(cn - sn) > 1e-6:
                    all_match = False
                    break
            if all_match:
                return True, '✅ 答對了！'
        return False, f'❌ 答案不正確（正確：{ca}）'

    # 4. 去除所有空格後精確比對（因式分解、括號表達式）
    ca_compact = re.sub(r'\s+', '', ca).lower()
    sa_compact = re.sub(r'\s+', '', sa).lower()
    if ca_compact == sa_compact:
        return True, '✅ 答對了！'

    # 5. 不等式正規化比對（x>2 vs x > 2）
    def norm_ineq(s):
        return re.sub(r'\s*([><]=?|[≤≥])\s*', lambda m: m.group(1), s.lower()).replace(' ', '')
    if norm_ineq(ca) == norm_ineq(sa):
        return True, '✅ 答對了！'

    # 6. 無法確定 — 視為錯誤，提示學生注意格式
    return False, f'❌ 答案不正確（正確答案格式：{ca}）'


from adaptive_review_mode import (
    AdaptiveReviewEngine,
    analyze_weak_skills,
    generate_review_plan,
    REVIEW_CONFIG,
    BETA_THRESHOLD,
)

# ═══════════════════════════════════════════════════════════════════════════
# 初始化
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
adaptive_review_bp = Blueprint('adaptive_review', __name__, url_prefix='/api/adaptive-review')

_SYMPY_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_SYMPY_LOCAL_DICT = {
    name: sp.Symbol(name)
    for name in (
        "a", "b", "c", "d", "m", "n", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z"
    )
}
_SYMPY_LOCAL_DICT.update({
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "pi": sp.pi,
})

# 全局引擎實例（緩存）
_engine_instance = None

def get_engine() -> AdaptiveReviewEngine:
    """獲取或創建複習引擎實例（單例）"""
    global _engine_instance
    if _engine_instance is None:
        logger.info("初始化自適應複習引擎...")
        _engine_instance = AdaptiveReviewEngine()
        logger.info("✓ 複習引擎已初始化")
    return _engine_instance


# ═══════════════════════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════════════════════

def normalize_history(history: Dict) -> Dict:
    """標準化學習歷史格式"""
    if isinstance(history, list):
        items = [h.get('item_id', 0) for h in history]
        skills = [h.get('skill_id', 0) for h in history]
        responses = [h.get('correct', 0) for h in history]
        return {
            'item_history': items,
            'skill_history': skills,
            'resp_history': responses,
        }
    
    if 'items' in history and 'item_history' not in history:
        history['item_history'] = history.pop('items')
    if 'skills' in history and 'skill_history' not in history:
        history['skill_history'] = history.pop('skills')
    if 'responses' in history and 'resp_history' not in history:
        history['resp_history'] = history.pop('responses')
    
    return history


def validate_history(history: Dict) -> Tuple[bool, str]:
    """驗證歷史格式"""
    required = ['item_history', 'skill_history', 'resp_history']
    for key in required:
        if key not in history:
            return False, f"缺少必需欄位: {key}"
        if not isinstance(history[key], list):
            return False, f"{key} 必須是列表"
    
    if not (len(history['item_history']) == len(history['skill_history']) == len(history['resp_history'])):
        return False, "三個序列長度不一致"
    
    return True, "OK"


def _save_review_state(student_id: str, history: Dict, apr: float):
    """持久化學習歷史到 adaptive_review_state 表格"""
    try:
        from models import db
        from sqlalchemy import text as sql_text
        db.session.execute(sql_text("""
            INSERT INTO adaptive_review_state (user_id, history_json, last_apr, updated_at)
            VALUES (:uid, :hist, :apr, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                history_json = excluded.history_json,
                last_apr = excluded.last_apr,
                updated_at = excluded.updated_at
        """), {
            'uid': int(student_id),
            'hist': json.dumps(history),
            'apr': apr,
        })
        db.session.commit()
    except Exception as e:
        logger.warning(f"無法儲存學習狀態: {e}")


def _load_review_state(student_id: str) -> Optional[Dict]:
    """從 adaptive_review_state 表格讀取舊歷史"""
    try:
        from models import db
        from sqlalchemy import text as sql_text
        row = db.session.execute(sql_text("""
            SELECT history_json FROM adaptive_review_state WHERE user_id = :uid
        """), {'uid': int(student_id)}).fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        logger.warning(f"無法讀取學習狀態: {e}")
    return None


def _ensure_state_table():
    """確保 adaptive_review_state 表格存在"""
    try:
        from models import db
        from sqlalchemy import text as sql_text
        db.session.execute(sql_text("""
            CREATE TABLE IF NOT EXISTS adaptive_review_state (
                user_id INTEGER PRIMARY KEY,
                history_json TEXT NOT NULL DEFAULT '{}',
                last_apr REAL DEFAULT 0.5,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.session.commit()
    except Exception as e:
        logger.warning(f"無法建立 adaptive_review_state 表格: {e}")


def _truncate_first_subproblem(full_text: str, correct_answer: str):
    """截斷多小題，只取第一小題；同時清除第一子題的前綴標記（⑴⑵①②(1)(2)）"""
    # 匹配第二個子題標記 ⑵ ② 2.  (2) 等
    SUB2 = re.compile(r'[⑵②]|(?:^|\s|\d)[（\(]?2[）\)][\s.、]', re.MULTILINE)
    m = SUB2.search(full_text)
    if m and m.start() > 10:
        question_text = full_text[:m.start()].strip()
        # 嘗試同步截斷答案（取第⑴小題的答案）
        am = SUB2.search(correct_answer)
        if am:
            correct_answer = correct_answer[:am.start()].strip()
    else:
        question_text = full_text.strip()
    # 清除答案中殘留的第⑴小題標記前綴（例如 "⑴ 0" → "0"）
    correct_answer = _strip_subproblem_prefix(correct_answer)
    return question_text, correct_answer


def _build_subskill_text(subskill_nodes: List[str], rag_summary: Optional[str] = None) -> str:
    """將 subskill nodes 與 RAG 摘要整理成 rag_tutor_prompt 可用的知識文字。"""
    from core.rag_engine import _label_node

    labels = []
    for node in subskill_nodes or []:
        text = str(node or "").strip()
        if text:
            labels.append(_label_node(text))

    blocks = []
    if labels:
        blocks.append("重點子技能：" + "、".join(labels))
    if rag_summary:
        summary = str(rag_summary).strip()
        if summary:
            blocks.append("知識庫摘要：" + summary)

    return "\n".join(blocks) if blocks else "核心概念"


def _build_adaptive_tutor_prompt(
    *,
    question_text: str,
    skill_name: str,
    family_id: str,
    subskill_nodes: List[str],
    user_query: str,
    rag_summary: Optional[str] = None,
) -> Tuple[str, str]:
    """優先使用後台可編輯的 rag_tutor_prompt，失敗時退回較詳細的本地 fallback。"""
    family_name_block = f"（{family_id}）" if family_id else ""
    prompt_query = f"[目前題目]\n{question_text}\n\n[學生提問]\n{user_query}"
    subskill_text = _build_subskill_text(subskill_nodes, rag_summary=rag_summary)

    try:
        prompt, source = compose_prompt(
            task_key="rag_tutor_prompt",
            query=prompt_query,
            ch_name=skill_name,
            family_name_block=family_name_block,
            subskill_text=subskill_text,
            route_label="Adaptive-Review-RAG",
        )
        if prompt.strip():
            return prompt, source
    except Exception as e:
        logger.warning(f"組裝 rag_tutor_prompt 失敗，改用 fallback prompt: {e}")

    fallback_prompt = LLM_GUIDE_PROMPT.format(
        skill=skill_name,
        question=question_text,
        query=user_query,
    )
    fallback_prompt += f"\n\n【技能主題】\n{family_name_block or '未提供'}\n\n【可用知識】\n{subskill_text}"
    return fallback_prompt, "adaptive_review_fallback"


def _fetch_seed_question_by_item_id(item_id: int, engine: AdaptiveReviewEngine) -> Dict[str, Any]:
    """依照既有固定映射邏輯取出 RL 選中的資料庫題目，作為 seed question。"""
    skill_id = engine.akt_inference.problem_to_skill_id.get(item_id, 0)
    if hasattr(engine.akt_inference, 'skills_list') and skill_id < len(engine.akt_inference.skills_list):
        skill_name = engine.akt_inference.skills_list[skill_id]
    else:
        skill_name = f"skill_{skill_id}"

    predicted_difficulty = 0.5
    if hasattr(engine, 'item_properties'):
        predicted_difficulty = engine.item_properties.get(item_id, {}).get('difficulty', 0.5)

    from models import db
    from sqlalchemy import text as sql_text

    count_res = db.session.execute(sql_text("""
        SELECT COUNT(*) FROM textbook_examples WHERE skill_id = :skill_id
    """), {'skill_id': skill_name}).scalar()

    seed_row = None
    if count_res and count_res > 0:
        offset = item_id % count_res
        seed_row = db.session.execute(sql_text("""
            SELECT id, problem_text, correct_answer, detailed_solution, difficulty_level, problem_type
            FROM textbook_examples
            WHERE skill_id = :skill_id
            ORDER BY id LIMIT 1 OFFSET :offset
        """), {'skill_id': skill_name, 'offset': offset}).fetchone()

    if seed_row and seed_row[1]:
        question_text, correct_answer = _truncate_first_subproblem(
            seed_row[1], seed_row[2] or ''
        )
        detailed_solution = seed_row[3] or ''
        source_question_id = seed_row[0]
        difficulty_level = seed_row[4] if seed_row[4] is not None else None
        problem_type = seed_row[5] or ''
    else:
        question_text = f"題目 {item_id}（技能: {skill_name}）"
        correct_answer = ''
        detailed_solution = ''
        source_question_id = None
        difficulty_level = None
        problem_type = ''

    rag_meta = db.session.execute(sql_text("""
        SELECT family_id, subskill_nodes
        FROM skill_family_bridge
        WHERE skill_id = :skill_id LIMIT 1
    """), {'skill_id': skill_name}).fetchone()

    family_id = rag_meta[0] if rag_meta else ''
    subskill_nodes = []
    if rag_meta and rag_meta[1]:
        try:
            if str(rag_meta[1]).startswith('['):
                subskill_nodes = json.loads(rag_meta[1])
            else:
                subskill_nodes = [s.strip() for s in str(rag_meta[1]).split(';') if s.strip()]
        except Exception:
            subskill_nodes = []

    answer_format_hint = _rule_based_format_hint(question_text)
    seed_question = {
        'item_id': item_id,
        'skill_id': skill_id,
        'skill_name': skill_name,
        'question_text': question_text,
        'correct_answer': correct_answer,
        'predicted_difficulty': float(predicted_difficulty),
        'answer_format_hint': answer_format_hint,
        'family_id': family_id,
        'subskill_nodes': subskill_nodes,
        'detailed_solution': detailed_solution,
        'difficulty_level': difficulty_level,
        'problem_type': problem_type,
        'source_question_id': source_question_id,
        'source': 'rl_fixed_question',
    }
    seed_question['expected_answer'] = correct_answer
    seed_question['acceptable_answers'] = [correct_answer] if correct_answer else []
    return seed_question


def _coerce_json_object(raw_text: str) -> Dict[str, Any]:
    """最佳努力解析 AI 回傳 JSON。"""
    from core.ai_analyzer import clean_and_parse_json

    cleaned = re.sub(r'^\s*```json\s*|\s*```\s*$', '', str(raw_text or '').strip(), flags=re.IGNORECASE)
    data = clean_and_parse_json(cleaned)
    if not isinstance(data, dict):
        raise ValueError("AI response is not a JSON object")
    return data


def _normalize_answer_expression(answer_expression: str) -> Tuple[str, str]:
    """用 SymPy 正規化答案表達式，回傳 (plain_text, latex_text)。"""
    raw = str(answer_expression or '').strip()
    if not raw:
        raise ValueError("answer_expression is empty")

    expr_text = raw.replace('×', '*').replace('÷', '/').replace('−', '-')
    expr_text = expr_text.replace('^', '**')
    expr_text = re.sub(r'(?<![A-Za-z])√\s*([A-Za-z0-9(])', r'sqrt(\1', expr_text)
    if expr_text.count('sqrt(') > expr_text.count(')'):
        expr_text += ')' * (expr_text.count('sqrt(') - expr_text.count(')'))
    expr = parse_expr(expr_text, transformations=_SYMPY_TRANSFORMATIONS, local_dict=_SYMPY_LOCAL_DICT)
    simplified = sp.simplify(expr)
    return str(simplified), sympy_latex(simplified)


def _build_ai_variant_prompt(seed_question: Dict[str, Any], student_state: Optional[Dict[str, Any]] = None) -> str:
    seed_question_text = str(seed_question.get("question_text") or "").strip()
    seed_answer = str(seed_question.get("correct_answer") or "").strip()
    difficulty_level = seed_question.get("difficulty_level")

    return (
        "你是一個數學題目變體生成器。\n"
        "任務：根據 seed 題，產生 1 題同型新題。\n"
        "只做出題，不要解釋，不要教學，不要詳解，不要 Markdown。\n"
        "保持相同概念、相同解題結構、相近難度，只替換數字或情境。\n"
        "數字要友善，避免巨大分數、複雜根式、超出課綱。\n\n"
        f"seed_question: {seed_question_text}\n"
        f"seed_answer: {seed_answer}\n"
        f"seed_difficulty: {difficulty_level}\n\n"
        "只輸出 JSON，格式如下：\n"
        '{"question_text":"...","answer_expression":"...","answer_type":"integer|fraction|sympy_expression|polynomial|radical"}'
    )


def generate_ai_variant_from_rl_selected_question(question_id, student_state=None):
    """
    RL 仍直接選 question_id/item_id；這裡只負責：
    1. 讀 seed 題
    2. 讓 AI 生成同構變體
    3. 用 SymPy/Python 驗證答案
    4. 失敗時 fallback 回固定資料庫題目
    """
    started_at = time.perf_counter()
    seed_question = None
    fallback_used = False
    ai_variant_success = False
    sympy_validation_status = "not_started"

    try:
        engine = get_engine()
        seed_question = _fetch_seed_question_by_item_id(int(question_id), engine)
        payload = dict(seed_question)
        payload['rl_selected_question_id'] = int(question_id)

        required_seed_fields = (
            seed_question.get('question_text'),
            seed_question.get('skill_name'),
            seed_question.get('family_id'),
            seed_question.get('correct_answer'),
        )
        if not all(v is not None for v in required_seed_fields):
            raise ValueError("seed question missing required fields")

        prompt = _build_ai_variant_prompt(seed_question, student_state=student_state)
        client = get_ai_client('default')
        response = call_ai_with_retry(client, prompt, max_retries=1, retry_delay=0, timeout=10)
        raw_text = getattr(response, 'text', str(response or ''))
        ai_data = _coerce_json_object(raw_text)

        for key in (
            "question_text", "answer_expression", "answer_type"
        ):
            if key not in ai_data or ai_data[key] in (None, ""):
                raise ValueError(f"missing required ai field: {key}")

        normalized_answer, normalized_latex = _normalize_answer_expression(ai_data["answer_expression"])
        sympy_validation_status = "parsed_ok"

        acceptable_answers = []
        for candidate in (
            normalized_answer,
            normalized_latex,
            str(ai_data["answer_expression"]).strip(),
            seed_question.get("correct_answer", ""),
        ):
            text = str(candidate or '').strip()
            if text and text not in acceptable_answers:
                acceptable_answers.append(text)

        payload.update({
            'question_text': str(ai_data['question_text']).strip(),
            'latex': str(ai_data.get('latex') or normalized_latex).strip(),
            'correct_answer': normalized_answer,
            'expected_answer': normalized_answer,
            'acceptable_answers': acceptable_answers,
            'answer_expression': normalized_answer,
            'answer_type': str(ai_data['answer_type']).strip(),
            'difficulty_level': ai_data.get('difficulty_level', seed_question.get('difficulty_level')),
            'problem_type': seed_question.get('problem_type', ''),
            'source_question_id': seed_question.get('source_question_id'),
            'variant_notes': str(ai_data.get('variant_notes') or '').strip(),
            'source': 'ai_variant_from_rl_seed',
            'answer_format_hint': _rule_based_format_hint(str(ai_data['question_text']).strip()),
            'seed_question_text': seed_question.get('question_text', ''),
            'seed_correct_answer': seed_question.get('correct_answer', ''),
            'skill_catalog_id': seed_question.get('skill_name', ''),
            'ai_raw_answer_expression': str(ai_data['answer_expression']).strip(),
        })
        ai_variant_success = True
    except Exception as exc:
        fallback_used = True
        payload = dict(seed_question or {})
        if payload:
            payload.setdefault('rl_selected_question_id', int(question_id))
            payload['source'] = 'rl_fixed_question_fallback'
        else:
            payload = {
                'item_id': int(question_id),
                'rl_selected_question_id': int(question_id),
                'question_text': f'題目 {question_id}',
                'correct_answer': '',
                'expected_answer': '',
                'acceptable_answers': [],
                'source': 'rl_fixed_question_fallback',
            }
        sympy_validation_status = sympy_validation_status if sympy_validation_status != "not_started" else f"fallback:{type(exc).__name__}"
        logger.warning(f"AI variant generation failed for rl_selected_question_id={question_id}: {exc}")

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    payload['runtime_log'] = {
        'rl_selected_question_id': int(question_id),
        'ai_variant_success': ai_variant_success,
        'fallback_used': fallback_used,
        'sympy_validation_status': sympy_validation_status,
        'generation_latency_ms': latency_ms,
    }
    logger.info(
        "[adaptive_review.variant] rl_selected_question_id=%s ai_variant_success=%s "
        "fallback_used=%s sympy_validation_status=%s generation_latency_ms=%s",
        question_id,
        ai_variant_success,
        fallback_used,
        sympy_validation_status,
        latency_ms,
    )
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# API 端點
# ═══════════════════════════════════════════════════════════════════════════

@adaptive_review_bp.route('/start', methods=['POST'])
def start_review_session():
    """
    開始新的自適應複習會話。
    若前端傳來的 history 為空陣列，則嘗試從資料庫恢復上次的學習歷程。
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '無效的 JSON'}), 400
        
        student_id = data.get('student_id', 'unknown')
        history = data.get('history', {})
        n_recommendations = data.get('n_recommendations', 1)
        use_rl = data.get('use_rl', True)
        
        history = normalize_history(history)

        # 若 history 為空，嘗試從 DB 恢復
        if (not history.get('item_history') and student_id not in ('unknown', 'guest')):
            _ensure_state_table()
            saved = _load_review_state(student_id)
            if saved:
                history = normalize_history(saved)
                logger.info(f"[RESTORE] 已恢復用戶 {student_id} 的學習歷程，共 {len(history['item_history'])} 題")

        is_valid, error_msg = validate_history(history)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400
        
        engine = get_engine()
        
        s_t = engine.get_knowledge_state(
            history['item_history'],
            history['skill_history'],
            history['resp_history']
        )
        current_apr = engine.get_apr(s_t)
        
        recommendations = engine.recommend_next_items(
            history['item_history'],
            history['skill_history'],
            history['resp_history'],
            n_items=n_recommendations,
            use_rl=use_rl
        )
        
        session_id = f"sess_{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return jsonify({
            'status': 'success',
            'data': {
                'student_id': student_id,
                'current_apr': current_apr,
                'apr_target': BETA_THRESHOLD,
                'recommendations': recommendations,
                'session_id': session_id,
                'restored_history': history,  # 讓前端更新 learningHistory
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'min_session_length': REVIEW_CONFIG['min_session_length'],
                    'max_session_length': REVIEW_CONFIG['max_session_length'],
                }
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 複習會話啟動失敗: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'伺服器錯誤: {str(e)}'}), 500


@adaptive_review_bp.route('/analyze/<student_id>', methods=['GET'])
def analyze_student(student_id: str):
    """分析學生的弱項和強項"""
    try:
        history_json = request.args.get('history_json', '{}')
        history = json.loads(history_json)
        
        history = normalize_history(history)
        is_valid, error_msg = validate_history(history)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400
        
        engine = get_engine()
        
        analysis = analyze_weak_skills(
            engine,
            history['item_history'],
            history['skill_history'],
            history['resp_history']
        )
        
        recommendations = []
        overall_apr = analysis['overall_apr']
        
        if overall_apr < 0.5:
            recommendations.append("⚠️ 整體熟練度較低，建議從基礎概念開始複習")
        
        for skill in analysis['weakest_skills'][:3]:
            if skill['mastery'] < 0.4:
                recommendations.append(f"🔴 {skill['skill_name']}: 熟練度 {skill['mastery']:.2%}，需要重點加強")
            elif skill['mastery'] < 0.65:
                recommendations.append(f"🟡 {skill['skill_name']}: 熟練度 {skill['mastery']:.2%}，建議強化練習")
        
        if overall_apr >= BETA_THRESHOLD:
            recommendations.append(f"✅ 已達到目標熟練度 ({overall_apr:.2%})，可考慮進階內容")
        
        return jsonify({
            'status': 'success',
            'data': {
                'student_id': student_id,
                'overall_apr': analysis['overall_apr'],
                'target_apr': BETA_THRESHOLD,
                'weakest_skills': analysis['weakest_skills'][:5],
                'strongest_skills': analysis['strongest_skills'][:5],
                'all_skills': analysis['all_skills'],
                'recommendations': recommendations,
            }
        }), 200
        
    except json.JSONDecodeError:
        return jsonify({'status': 'error', 'message': '無效的歷史 JSON'}), 400
    except Exception as e:
        logger.error(f"❌ 分析失敗: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'伺服器錯誤: {str(e)}'}), 500


@adaptive_review_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    提交答題反饋：
    1. 用 LLM 評分學生答案
    2. 更新 AKT 知識狀態與 APR
    3. 將 APR 寫入 NodeCompetency（連動知識圖譜）
    4. 持久化學習歷史
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': '無效的 JSON'}), 400
        
        student_id = data.get('student_id', 'unknown')
        history = normalize_history(data.get('history', {}))
        item_id = data.get('item_id')
        user_answer = data.get('user_answer', '').strip()  # 學生的真實作答
        question_text = data.get('question_text', '')
        correct_answer = data.get('correct_answer', '')
        
        if item_id is None:
            return jsonify({'status': 'error', 'message': '缺少 item_id'}), 400
        
        is_valid, error_msg = validate_history(history)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400
        
        # ── 規則式評分（毫秒級，取代 LLM）────────────────────────────────
        correct = 0
        grade_detail = '未能評分'
        if not user_answer:
            correct = 0
            grade_detail = '❌ 未輸入答案'
        elif correct_answer:
            is_correct, grade_detail = _rule_based_grade(correct_answer, user_answer)
            correct = 1 if is_correct else 0
        else:
            # 無正確答案可比，保守當作對
            correct = 1
            grade_detail = '✅ 已記錄（無標準答案）'
        
        # ── AKT 更新 ──────────────────────────────────────────────────────
        engine = get_engine()
        
        s_t_before = engine.get_knowledge_state(
            history['item_history'],
            history['skill_history'],
            history['resp_history']
        )
        apr_before = engine.get_apr(s_t_before)
        
        skill_id = engine.akt_inference.problem_to_skill_id.get(item_id, 0)
        updated_history = {
            'item_history': history['item_history'] + [item_id],
            'skill_history': history['skill_history'] + [skill_id],
            'resp_history': history['resp_history'] + [correct],
        }
        
        s_t_after = engine.get_knowledge_state(
            updated_history['item_history'],
            updated_history['skill_history'],
            updated_history['resp_history']
        )
        apr_after = engine.get_apr(s_t_after)
        apr_change = apr_after - apr_before
        
        next_recommendations = engine.recommend_next_items(
            updated_history['item_history'],
            updated_history['skill_history'],
            updated_history['resp_history'],
            n_items=1,
            use_rl=True
        )
        
        # ── NodeCompetency 更新（連動知識圖譜）────────────────────────────
        try:
            from models import db, NodeCompetency
            skill_name = "unknown"
            if hasattr(engine.akt_inference, 'skills_list') and skill_id < len(engine.akt_inference.skills_list):
                skill_name = engine.akt_inference.skills_list[skill_id]
                
            if skill_name != "unknown" and student_id not in ('unknown', 'guest'):
                comp_score = float(apr_after * 100.0)
                nc = db.session.query(NodeCompetency).filter_by(user_id=int(student_id), node_id=skill_name).first()
                if nc:
                    nc.competency_score = comp_score
                    nc.competency_theta = apr_after
                else:
                    nc = NodeCompetency(
                        user_id=int(student_id),
                        node_id=skill_name,
                        competency_score=comp_score,
                        competency_theta=apr_after
                    )
                    db.session.add(nc)
                db.session.commit()
        except Exception as db_err:
            logger.warning(f"無法更新 NodeCompetency: {db_err}")
        
        # ── 持久化學習歷史 ────────────────────────────────────────────────
        if student_id not in ('unknown', 'guest'):
            _ensure_state_table()
            _save_review_state(student_id, updated_history, apr_after)
        
        # ── 組合回饋訊息 ──────────────────────────────────────────────────
        if apr_change > 0.05:
            apr_msg = " 熟練度顯著提升！"
        elif apr_change > 0:
            apr_msg = " 熟練度略有提升"
        elif apr_after >= BETA_THRESHOLD:
            apr_msg = f" 🎉 已達到目標熟練度！"
        else:
            apr_msg = " 繼續加油！"
        
        feedback_msg = grade_detail + apr_msg
        
        return jsonify({
            'status': 'success',
            'data': {
                'student_id': student_id,
                'item_id': item_id,
                'correct': correct,
                'grade_detail': grade_detail,
                'apr_before': apr_before,
                'apr_after': apr_after,
                'apr_change': apr_change,
                'target_apr': BETA_THRESHOLD,
                'reached_target': apr_after >= BETA_THRESHOLD,
                'next_recommendations': next_recommendations,
                'feedback': feedback_msg,
                'updated_history': updated_history,
                'timestamp': datetime.now().isoformat(),
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ 反饋提交失敗: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'伺服器錯誤: {str(e)}'}), 500


@adaptive_review_bp.route('/question/<int:item_id>', methods=['GET'])
def get_question(item_id: int):
    """
    獲取單個題目：
    - RL 仍直接指定 item_id
    - 優先把該固定資料庫題目當成 seed，生成 AI 同構變體
    - 若 AI/JSON/SymPy 驗證失敗，fallback 回既有固定題目流程
    - correct_answer 回傳給後端評分用，不在前端顯示
    """
    try:
        student_state = request.args.get('student_state')
        if student_state:
            try:
                student_state = json.loads(student_state)
            except Exception:
                student_state = {'raw': str(student_state)}
        data = generate_ai_variant_from_rl_selected_question(item_id, student_state=student_state)

        return jsonify({
            'status': 'success',
            'data': {
                'item_id': data.get('item_id', item_id),
                'skill_id': data.get('skill_id', 0),
                'skill_name': data.get('skill_name', ''),
                'question_text': data.get('question_text', ''),
                'correct_answer': data.get('correct_answer', ''),   # 後端評分用，前端不顯示
                'predicted_difficulty': float(data.get('predicted_difficulty', 0.5)),
                'answer_format_hint': data.get('answer_format_hint', ''),
                'family_id': data.get('family_id', ''),
                'subskill_nodes': data.get('subskill_nodes', []),
                'expected_answer': data.get('expected_answer', data.get('correct_answer', '')),
                'acceptable_answers': data.get('acceptable_answers', []),
                'difficulty_level': data.get('difficulty_level'),
                'problem_type': data.get('problem_type', ''),
                'source_question_id': data.get('source_question_id'),
                'source': data.get('source', 'rl_fixed_question'),
                'runtime_log': data.get('runtime_log', {}),
                'latex': data.get('latex', ''),
                'variant_notes': data.get('variant_notes', ''),
                'skill_catalog_id': data.get('skill_catalog_id', data.get('skill_name', '')),
                'rl_selected_question_id': data.get('rl_selected_question_id', item_id),
            }
        }), 200

    except Exception as e:
        logger.error(f"❌ 獲取題目失敗: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'無法獲取題目: {str(e)}'}), 500


@adaptive_review_bp.route('/check-handwriting', methods=['POST'])
def check_handwriting():
    """
    接收前端手寫板的 base64 圖片，轉交共用手寫分析流程。
    這樣 RL 複習模式會和 ai_prompt_settings 的 runtime 角色模型設定保持一致。
    """
    try:
        from core.routes.analysis import analyze_handwriting as shared_analyze_handwriting

        return shared_analyze_handwriting()

    except Exception as e:
        logger.error(f"❌ 手寫辨識失敗: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'AI 分析錯誤: {str(e)}'}), 500


@adaptive_review_bp.route('/chat', methods=['POST'])
def chat_with_tutor():
    """
    提供 LLM 引導式提示服務（不可直接洩漏答案）。
    當學生請求提示時，先嘗試 RAG（rag_hint_engine）組成結構化提示 HTML；
    若 RAG 無可用節點，退而使用 LLM 純文字引導。

    Request JSON:
    {
        "question_text": "題目內容...",
        "skill_name": "技能名稱...",
        "skill_id": "jh_數學1上_FourArithmeticOperationsOfIntegers",  # optional
        "family_id": "I1",                                            # optional
        "subskill_nodes": ["sign_handling", "mixed_ops"],             # optional
        "query": "學生的發言"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': '無效的 JSON'}), 400

    question_text = data.get('question_text', '未提供題目')
    skill_name = data.get('skill_name', '一般數學觀念')
    skill_id = (data.get('skill_id') or '').strip()
    family_id = (data.get('family_id') or '').strip()
    subskill_nodes = data.get('subskill_nodes') or []
    user_query = data.get('query', '請給我一個提示')
    rag_mode = data.get('rag_mode', 'naive')

    # ── RAG 層：嘗試從 skill_family_bridge 取得結構化提示 ───────────────────
    rag_hint_html = None
    rag_summary = None
    rag_error = None
    if rag_mode != 'none' and (subskill_nodes or skill_id or family_id):
        try:
            from core.adaptive.rag_hint_engine import build_rag_hint
            # 若未傳入 subskill_nodes，使用空清單（build_rag_hint 需至少有一個 node）
            nodes_to_use = subskill_nodes if subskill_nodes else []
            if nodes_to_use:
                rag_result = build_rag_hint(
                    subskill_nodes=nodes_to_use,
                    skill_id=skill_id,
                    family_id=family_id,
                    question_text=question_text[:240],
                )
                rag_hint_html = rag_result.get('hint_html')
                rag_summary = rag_result.get('hint_summary')
                logger.info(f"[RAG] 複習模式 hint 產生成功 summary={rag_summary}")
        except Exception as rag_err:
            logger.warning(f"[RAG] 複習模式 RAG 提示產生失敗（退回 LLM）: {rag_err}")
            rag_error = str(rag_err)

    # ── LLM 層：產生文字引導回覆（永遠執行，補足 RAG 無文字的部分）──────────
    llm_reply = None
    prompt, prompt_source = _build_adaptive_tutor_prompt(
        question_text=question_text,
        skill_name=skill_name,
        family_id=family_id,
        subskill_nodes=subskill_nodes,
        user_query=user_query,
        rag_summary=rag_summary,
    )
    try:
        client = get_ai_client('tutor')
        response = call_ai_with_retry(client, prompt)
        llm_reply = response.text if response and hasattr(response, 'text') else None
        logger.info(
            f"[Adaptive Review Tutor] prompt_source={prompt_source} "
            f"rag_used={rag_hint_html is not None} family_id={family_id or '-'}"
        )
    except Exception as e:
        logger.warning(f"LLM 提示產生失敗（RAG 優先模式）: {e}")

    # ── 合併回覆 ────────────────────────────────────────────────────────────
    if rag_hint_html and llm_reply:
        # 結構化 RAG 提示 + LLM 文字引導
        reply = f"{rag_hint_html}\n\n💬 **AI 引導**：{llm_reply}"
    elif rag_hint_html:
        reply = rag_hint_html
    elif llm_reply:
        reply = llm_reply
    else:
        reply = '目前無法取得提示，請稍後再試或洽老師。'

    return jsonify({
        'status': 'success',
        'data': {
            'reply': reply,
            'rag_used': rag_hint_html is not None,
            'rag_summary': rag_summary,
        }
    }), 200


@adaptive_review_bp.route('/health', methods=['GET'])
def health_check():
    """健康檢查端點"""
    try:
        engine = get_engine()
        return jsonify({
            'status': 'success',
            'message': '複習引擎就緒',
            'n_skills': engine.n_skills,
            'n_items': engine.n_items,
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'複習引擎異常: {e}'}), 500


if __name__ == "__main__":
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(adaptive_review_bp)
    logging.basicConfig(level=logging.INFO)
    print("自適應複習模式 API 伺服器")
    print("http://127.0.0.1:5000/api/adaptive-review/health")
    app.run(debug=True, port=5000)
