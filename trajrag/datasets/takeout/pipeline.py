from __future__ import annotations
import duckdb
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Any
from constants import OPENAI_KEY, OPENAI_BASE_URL, EMBED_MODEL, USE_LOCAL_EMBED, LOCAL_EMBED_MODEL, EMBED_DIM, CANDIDATE_TOPK_TOTAL, USE_LLM_RERANK, LLM_RERANK_TOP_K
from parser import parse_intent
from evidence import retrieve_evidence, unified_retrieve, retrieve_point, retrieve_poi_candidates_via_sql
from scoring import score_candidates
from llm_rerank import llm_rerank
from utils import traj_to_dataframe

_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()


def embed_query(query: str) -> np.ndarray:
    if USE_LOCAL_EMBED:
        global _LOCAL_EMBED
        try:
            _LOCAL_EMBED
        except NameError:
            from sentence_transformers import SentenceTransformer
            _LOCAL_EMBED = SentenceTransformer(LOCAL_EMBED_MODEL)
        vec = _LOCAL_EMBED.encode("query: " + query, normalize_embeddings=True)
        return vec.astype(np.float32)
    resp = _client.embeddings.create(model=EMBED_MODEL, input=query)
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _populate_raw_traj_cache(uid, traj):
    try:
        import evidence as _ev
        _ev._RAW_TRAJ_CACHE[str(uid)] = traj
    except Exception:
        pass


def build_conn_from_traj(uid: str, traj: list,
                          poi_df: pd.DataFrame | None = None,
                          admin_df: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    _populate_raw_traj_cache(uid, traj)
    df = traj_to_dataframe(traj, uid)
    if len(df) == 0:
        df = pd.DataFrame(columns=[
            "uid", "event_start", "event_end", "poi_id", "poi_name", "poi_type",
            "lat", "lon", "o_lat", "o_lon", "d_lat", "d_lon", "status",
        ])
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")
    except Exception as _e:
        print(f"[build_conn] spatial extension load failed: {str(_e)[:80]}")
    conn.execute("CREATE TABLE trajectory AS SELECT * FROM df")
    if poi_df is not None and len(poi_df) > 0:
        pdf = poi_df
        conn.execute("CREATE TABLE pois AS SELECT * FROM pdf")
        conn.execute("CREATE INDEX idx_pois_type ON pois(primary_type)")
    if admin_df is not None and len(admin_df) > 0:
        adf = admin_df
        conn.execute("CREATE TABLE admin_regions AS SELECT * FROM adf")
        conn.execute("CREATE INDEX idx_admin_name ON admin_regions(name_en_norm)")
        conn.execute("CREATE INDEX idx_admin_level ON admin_regions(admin_level)")
    return conn


def score_query(
    query: str,
    uid: str,
    traj: list,
    poi_pool: List[Dict[str, Any]],
    poi_emb_cache: dict | None = None,
    poi_emb_matrix: np.ndarray | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    candidate_top_k: int = CANDIDATE_TOPK_TOTAL,
    use_llm_rerank: bool = USE_LLM_RERANK,
    llm_rerank_top_k: int = LLM_RERANK_TOP_K,
    verbose: bool = False,
    profile_cache: dict | None = None,
) -> List[Dict[str, Any]]:
    intent = parse_intent(query)
    intent['query'] = query
    if verbose:
        print(f"\n[Step 1] spatial.mode={intent.get('spatial',{}).get('mode')} "
              f"cat={intent.get('target',{}).get('category')} "
              f"zone_name={intent.get('spatial',{}).get('zone_name')}")

    from profile import build_profile
    from profile_compact import compact as compact_profile
    if profile_cache is not None and uid in profile_cache:
        profile = profile_cache[uid]
    else:
        profile = build_profile(uid, traj)
        if profile_cache is not None:
            profile_cache[uid] = profile
    profile_text = compact_profile(profile)

    if conn is None:
        conn = build_conn_from_traj(uid, traj, poi_df=None, admin_df=None)

    R_tau, A_q = unified_retrieve(intent, str(uid), conn,
                                   top_k_total=candidate_top_k,
                                   profile_text=profile_text)

    if intent.get("spatial", {}).get("mode") == "route" and R_tau.get("route") is None:
        intent["spatial"]["mode"] = "point"
        intent["spatial"]["anchor"] = "current"
        pt = retrieve_point(str(uid), {"anchor": "current"}, conn, profile_text)
        if pt:
            R_tau["point"] = pt
        if verbose:
            print("[Step 2] route evidence missing → downgraded to mode=point+current")
        A_q = retrieve_poi_candidates_via_sql(intent, R_tau, conn, top_k=candidate_top_k)

    if verbose:
        print(f"[Step 2] R_tau keys: {[k for k, v in R_tau.items() if v]} | candidates: {len(A_q)}")

    q_emb = embed_query(query)
    if poi_emb_cache is None:
        poi_emb_cache = {}

    ranked = score_candidates(query, intent, R_tau, A_q, q_emb, poi_emb_cache)

    intent_tem = intent.get("temporal") or {}
    time_bucket = intent_tem.get("time_bucket") or intent_tem.get("hour_bucket")
    day_hint = intent_tem.get("day_hint")
    try:
        from candidate import filter_by_opening_hours
        from utils import brand_dedupe
        if time_bucket:
            filt_idx = filter_by_opening_hours(list(range(len(ranked))), ranked, time_bucket, day_hint)
            ranked = [ranked[i] for i in filt_idx]
        ranked = brand_dedupe(ranked, k=200)
    except Exception as e:
        if verbose:
            print(f"[post-filter] failed: {e}")

    if use_llm_rerank:
        ranked = llm_rerank(query=query, R_tau=R_tau, ranked=ranked, top_k=llm_rerank_top_k, verbose=verbose)

    return ranked
