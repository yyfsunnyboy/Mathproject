# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from typing import Any, List, Dict
from config import Config
from core.prompts.registry import render_prompt

HAS_ADV_LIBS = True
try: import chromadb
except ImportError: HAS_ADV_LIBS = False
try: from sentence_transformers import SentenceTransformer
except ImportError: HAS_ADV_LIBS = False
try: from rank_bm25 import BM25Okapi
except ImportError: HAS_ADV_LIBS = False
try: import jieba
except ImportError: HAS_ADV_LIBS = False

from core.rag_engine import _load_bridge_rows, _build_document_text, _parse_subskill_nodes, rag_search, _label_node

logger = logging.getLogger(__name__)

_adv_chroma_client = None
_adv_collection = None
_bm25 = None
_embedding_model = None
_documents: List[str] = []
_ids: List[str] = []
_metadatas: List[Dict[str, Any]] = []
_skill_map: Dict[str, str] = {}
_bridge_rows: List[Dict[str, Any]] = []

def _to_chroma_metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def init_adv_rag(app):
    global _adv_chroma_client, _adv_collection, _bm25, _embedding_model
    global _documents, _ids, _metadatas, _skill_map, _bridge_rows

    missing = []
    try: import chromadb
    except ImportError: missing.append("chromadb")
    try: from sentence_transformers import SentenceTransformer
    except ImportError: missing.append("sentence_transformers")
    try: from rank_bm25 import BM25Okapi
    except ImportError: missing.append("rank_bm25")
    try: import jieba
    except ImportError: missing.append("jieba")

    if missing:
        global HAS_ADV_LIBS
        HAS_ADV_LIBS = False
        logger.warning(f"[Advanced RAG] Missing dependencies: {', '.join(missing)}. Advanced mode disabled. Fallback to Naive RAG.")
        return
    else:
        logger.info("[Advanced RAG] All dependencies met. Initializing Advanced RAG Engine.")

    try:
        from core.ai_analyzer import gemini_model
    except ImportError:
        pass

    with app.app_context():
        rows = _load_bridge_rows(app)
        
    if not rows:
        logger.info("[Advanced RAG] No skill data found.")
        return

    _bridge_rows = list(rows)
    _embedding_model = SentenceTransformer('shibing624/text2vec-base-chinese')
    _adv_chroma_client = chromadb.Client()

    try:
        _adv_chroma_client.delete_collection("math_skills_adv")
    except Exception:
        pass

    _adv_collection = _adv_chroma_client.create_collection(name="math_skills_adv")

    _documents = []
    _ids = []
    _metadatas = []
    _skill_map = {}

    for row in rows:
        skill_id = str(row.get("skill_id", "") or "").strip()
        family_id = str(row.get("family_id", "") or "").strip()
        if not skill_id:
            continue
        
        display_name = str(row.get("skill_ch_name") or row.get("skill_name") or skill_id).strip()
        _skill_map[skill_id] = display_name
        doc_text = _build_document_text(row)
        if not doc_text:
            continue
            
        doc_id = f"{skill_id}:{family_id}" if family_id else skill_id
        if doc_id in _ids:
            continue
            
        _documents.append(doc_text)
        _ids.append(doc_id)
        
        metadata_raw = {
            "skill_id": skill_id,
            "family_id": family_id,
            "family_name": row.get("family_name"),
            "subskill_nodes": row.get("subskill_nodes"),
            "curriculum": row.get("curriculum"),
            "grade": row.get("grade"),
            "chapter": row.get("chapter"),
            "section": row.get("section"),
            "skill_ch_name": row.get("skill_ch_name"),
        }
        _metadatas.append({k: _to_chroma_metadata_value(v) for k, v in metadata_raw.items()})

    if not _documents:
        logger.info("[Advanced RAG] No documents to index.")
        return

    # Embed and add to DB
    logger.info(f"[Advanced RAG] Building index for {_documents.__len__()} items (this may take a bit)...")
    
    import os, pickle, hashlib
    docs_string = "".join(_documents)
    docs_hash = hashlib.md5(docs_string.encode('utf-8')).hexdigest()
    cache_file = os.path.join(app.root_path, '..', 'configs', 'adv_embeddings_cache.pkl')
    
    embeddings = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                if cache_data.get('hash') == docs_hash:
                    embeddings = cache_data.get('embeddings')
                    logger.info("[Advanced RAG] Loaded embeddings from cache!")
        except Exception:
            pass
            
    if embeddings is None:
        embeddings = _embedding_model.encode(_documents).tolist()
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({'hash': docs_hash, 'embeddings': embeddings}, f)
        except Exception:
            pass

    _adv_collection.add(
        documents=_documents,
        ids=_ids,
        metadatas=_metadatas,
        embeddings=embeddings
    )

    # Initialize BM25
    tokenized_docs = [list(jieba.cut(doc)) for doc in _documents]
    _bm25 = BM25Okapi(tokenized_docs)
    logger.info("[Advanced RAG] Initialization complete!")


def get_llm_keywords(query: str) -> List[str]:
    """Extracts key search words from the student's problem using jieba (rule-based).
    LLM keyword extraction is skipped by default to conserve API quota.
    """
    try:
        import jieba
        tokenized = list(jieba.cut(query))
        keywords = [w for w in tokenized if len(w) >= 2][:5]
        if keywords:
            logger.info("[Advanced RAG Keyword Extraction] strategy=jieba_rule_based")
            return keywords
    except Exception as e:
        logger.warning(f"[Advanced RAG Keyword Extraction] jieba failed: {e}")
    return []

def extract_advanced_rag(query: str, K_VALUE=42, top_k=5):
    """Executes the dense + sparse hybrid RRF retrieval as per experiment script."""
    keywords = get_llm_keywords(query)
    q_dense = query + " " + " ".join(keywords) if keywords else query
    q_sparse = " ".join(keywords) if keywords else query

    vec_dense = _embedding_model.encode([q_dense]).tolist()
    res_dense = _adv_collection.query(query_embeddings=vec_dense, n_results=50)
    dense_ids = res_dense['ids'][0] if res_dense and res_dense['ids'] else []

    tokenized_query = list(jieba.cut(q_sparse))
    bm25_scores = _bm25.get_scores(tokenized_query)
    # Get top 50 indices
    top_n_idx = np.argsort(bm25_scores)[::-1][:50]
    sparse_ids = [_ids[i] for i in top_n_idx]

    rrf_scores = {}
    rrf_k = 60
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rrf_k + rank + 1)

    sorted_hybrid_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ans_list_advanced = [x[0] for x in sorted_hybrid_ids][:K_VALUE]

    # Map the retrieved IDs back to the desired output format
    output = []
    added = set()
    for doc_id in ans_list_advanced:
        if doc_id in added:
            continue
        try:
            idx = _ids.index(doc_id)
            metadata = _metadatas[idx]
            output.append({
                "skill_id": metadata.get("skill_id") or doc_id.split(":", 1)[0],
                "family_id": metadata.get("family_id"),
                "family_name": metadata.get("family_name"),
                "chinese_name": metadata.get("skill_ch_name") or _skill_map.get(metadata.get("skill_id"), metadata.get("skill_id")),
                "subskill_nodes": _parse_subskill_nodes(metadata.get("subskill_nodes")),
                "curriculum": metadata.get("curriculum"),
                "grade": metadata.get("grade"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "distance": 0, # Since we use RRF score, distance is not exact space distance
                "rrf_score": rrf_scores[doc_id]
            })
            added.add(doc_id)
            if len(output) >= top_k:
                break
        except ValueError:
            pass

    return output


def adv_rag_search(query: str, top_k=5):
    """
    Hybrid RAG Search controller with Configured Threshold
    """
    if not HAS_ADV_LIBS or _adv_collection is None:
        logger.warning("[Advanced RAG] Not ready. Fallback to Naive RAG.")
        return rag_search(query, top_k)

    # 1. First run Naive RAG to test distance
    naive_results = rag_search(query, top_k=top_k)
    naive_dist = naive_results[0]['distance'] if naive_results and 'distance' in naive_results[0] else float('inf')

    # 2. Check if we can route to Fast Path (Naive)
    try:
        from flask import current_app
        threshold = current_app.config.get('ADVANCED_RAG_NAIVE_THRESHOLD', getattr(Config, 'ADVANCED_RAG_NAIVE_THRESHOLD', 0.85))
    except RuntimeError:
        threshold = getattr(Config, 'ADVANCED_RAG_NAIVE_THRESHOLD', 0.85)
    
    # 距離分數越低代表越接近。
    if naive_dist <= threshold:
        logger.info(f"[Hybrid Routing] Naive distance {naive_dist:.3f} <= {threshold:.3f}. Using Fast Path (Naive).")
        # Ensure we attach a flag so frontend can see
        for res in naive_results:
            res['routing'] = 'Naive Fast-Path'
        return naive_results

    # 3. Else run Advanced RAG 
    logger.info(f"[Hybrid Routing] Naive distance {naive_dist:.3f} > {threshold:.3f}. Using Advanced RAG Recovery Path.")
    adv_results = extract_advanced_rag(query, K_VALUE=42, top_k=top_k)
    for res in adv_results:
        res['routing'] = 'Advanced RAG'
    
    return adv_results


def _build_adv_rag_prompt(
    query: str,
    retrieved_skills: list,
    question_text: str = "",
    family_id: str = "",
) -> str:
    units_text = ""
    for i, skill in enumerate(retrieved_skills):
        ch_name = skill.get("chinese_name", "")
        fam_name = skill.get("family_name", "")
        subs = skill.get("subskill_nodes", [])
        subs_str = "、".join(_label_node(n) for n in subs) if subs else "核心基礎"
        units_text += f"{i+1}. 單元: {ch_name} (類別: {fam_name}) - 重點: {subs_str}\n"

    current_question_block = ""
    if question_text:
        current_question_block = f"""
目前正在作答的題目：
{question_text}
""" 
        if family_id:
            current_question_block += f"\n目前題型 family：{family_id}\n"

    return f"""
你是「數學引導型助教（Scaffolding Tutor）」。

【任務】
只提供「下一步操作提示」。

【限制】
- 不要開場白
- 不要長篇解釋
- 不要完整解法
- 不要講核心觀念
- 不要提到檢索內容
- 控制在 1～2 句

{current_question_block}

學生問題：
{query}

檢索出的相關知識點：
{units_text}

【輸出】
直接給提示，例如：
- 先把有 x 的項和常數項分開
- 試著整理成兩個括號
"""


def _render_adv_rag_prompt_via_registry(
    query: str,
    retrieved_skills: list,
    question_text: str = "",
    family_id: str = "",
) -> str:
    ch_name = "多個相關單元"
    family_name_block = ""
    if retrieved_skills and len(retrieved_skills) == 1:
        ch_name = retrieved_skills[0].get("chinese_name", "")
        family_name = retrieved_skills[0].get('family_name', '')
        family_name_block = f"（{family_name}）" if family_name else ""
    elif family_id:
        family_name_block = f"（{family_id}）"

    units = []
    for idx, skill in enumerate(retrieved_skills or []):
        name = skill.get("chinese_name", "")
        fam = skill.get("family_name", "")
        subs = skill.get("subskill_nodes", [])
        subs_str = "、".join(_label_node(n) for n in subs) if subs else "未分類"
        units.append(f"{idx + 1}. {name}（{fam}）: {subs_str}")
        
    subskill_text = "\n".join(units) if units else "無對應知識"
    
    prompt_query = f"[目前題目]\n{question_text}\n\n[學生提問]\n{query}" if question_text else query

    route_label = "Advanced RAG"
    if retrieved_skills and "routing" in retrieved_skills[0]:
        route_label = retrieved_skills[0]["routing"]

    # Primary path: DB-configurable tutor prompt for RAG
    from core.prompts.composer import compose_prompt
    prompt, source = compose_prompt(
        task_key="rag_tutor_prompt",
        query=prompt_query,
        ch_name=ch_name,
        family_name_block=family_name_block,
        subskill_text=subskill_text,
        route_label=route_label,
    )
    return prompt, source


def adv_rag_chat(
    query: str,
    retrieved_skills: list,
    provider: str = "auto",
    question_text: str = "",
    family_id: str = "",
):
    """
    Generates Answer utilizing multiple retrieved contents from Advanced RAG.
    If no skills are retrieved, falls back to a direct LLM call using the question context.
    """
    # Auto-detect provider from ai_settings when not explicitly specified
    if not provider or provider in ("auto", ""):
        try:
            from core.ai_settings import get_effective_model_config
            cfg = get_effective_model_config("tutor")
            provider = str(cfg.get("provider", "local")).strip().lower()
        except Exception:
            provider = "local"

    # When RAG finds no relevant skills, fall back to direct LLM call
    if not retrieved_skills:
        logger.warning("[Adv RAG Chat] No retrieved skills — falling back to direct LLM call.")
        fallback_prompt = f"你是一位國中數學引導型助教。\n"
        if question_text:
            fallback_prompt += f"學生正在作答的題目：\n{question_text}\n\n"
        fallback_prompt += f"學生的問題：{query}\n\n只給 1-2 句提示，不要直接給答案。"
        try:
            if provider in ("google", "gemini", "cloud"):
                from core.ai_analyzer import gemini_model
                if gemini_model:
                    import google.generativeai as genai
                    resp = gemini_model.generate_content(
                        fallback_prompt,
                        generation_config=genai.types.GenerationConfig(temperature=0.4)
                    )
                    return {"reply": resp.text, "provider": "google", "rag_fallback": True}
            from core.ai_client import call_ai
            resp = call_ai("tutor", fallback_prompt)
            return {"reply": getattr(resp, "text", str(resp)), "provider": "local", "rag_fallback": True}
        except Exception as e:
            logger.error(f"[Adv RAG Chat] Fallback LLM error: {e}")
            return {"reply": f"AI 回覆發生錯誤：{str(e)}"}

    provider = (provider or "auto").strip().lower()
    try:
        prompt, source = _render_adv_rag_prompt_via_registry(
            query,
            retrieved_skills,
            question_text=question_text,
            family_id=family_id,
        )
        logger.info(f"[Prompt Trace] route='/api/adaptive/adv_rag_chat' task_type='adv_rag_chat' prompt_key='tutor_hint_prompt' source='{source}' model_role='tutor'")
    except Exception:
        # Keep legacy builder as fallback to avoid runtime interruption.
        prompt = _build_adv_rag_prompt(
            query,
            retrieved_skills,
            question_text=question_text,
            family_id=family_id,
        )

    try:
        if provider == "local":
            from core.ai_client import call_ai
            response = call_ai("tutor", prompt)
            return {"reply": getattr(response, "text", str(response)), "provider": "local"}

        from core.ai_analyzer import gemini_model
        if not gemini_model:
            return {"reply": "目前 Google API 助教尚未啟用，請改用本地模型。", "provider": "google"}

        import google.generativeai as genai
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
            ),
        )
        return {"reply": response.text, "provider": "google"}

    except Exception as e:
        logger.error(f"[Adv RAG Chat] Error: {e}")
        return {"reply": f"AI 回覆發生錯誤：{str(e)}"}
