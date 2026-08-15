from __future__ import annotations
from typing import Any, Dict, List, Sequence
import numpy as np

from constants import (
    CANDIDATE_TOPK_TOTAL,
    CANDIDATE_TOPK_SPATIAL,
    CANDIDATE_TOPK_SEMANTIC,
    CANDIDATE_TOPK_PREFERENCE,
    ROUTE_PREFILTER_MULTIPLIER,
    R_ROUTE_KM,
    R_POINT_KM,
    ZONE_ALPHA,
    CATEGORY_ALIASES,
)
from utils import haversine_vec, cos_sim_vec, topk_indices, dist_to_polyline, valid_latlon


def _pool_arrays(pool: List[Dict[str, Any]]):
    lats = np.array([float(p.get("lat")) if valid_latlon(p.get("lat"), p.get("lon")) else np.nan for p in pool], dtype=float)
    lons = np.array([float(p.get("lon")) if valid_latlon(p.get("lat"), p.get("lon")) else np.nan for p in pool], dtype=float)
    valid = np.isfinite(lats) & np.isfinite(lons)
    return lats, lons, valid


def _emb_matrix_from_cache(pool: List[Dict[str, Any]], poi_emb_cache: dict):
    embs = []
    ids = []
    for p in pool:
        pid = str(p.get("poi_id") or p.get("id") or "")
        emb = poi_emb_cache.get(pid)
        if emb is None:
            emb = np.zeros(0, dtype=np.float32)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.size == 0:
            embs.append(None)
        else:
            embs.append(emb)
        ids.append(pid)
    dim = next((e.shape[0] for e in embs if e is not None), 1536)
    M = np.zeros((len(pool), dim), dtype=np.float32)
    for i, e in enumerate(embs):
        if e is not None and e.shape[0] == dim:
            M[i] = e
    return M, ids


def _indices_to_candidates(pool: List[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for idx in indices:
        if idx is None:
            continue
        i = int(idx)
        if i < 0 or i >= len(pool) or i in seen:
            continue
        seen.add(i)
        out.append(pool[i])
    return out


def _load_valid_types():
    global _VALID_TYPES, _VALID_TYPE_EMBS, _VALID_TYPE_NORMS
    if _VALID_TYPES is not None:
        return _VALID_TYPES, _VALID_TYPE_EMBS, _VALID_TYPE_NORMS
    import pickle, os
    p = os.path.join(os.path.dirname(__file__), "valid_type_embs.pkl")
    if not os.path.exists(p):
        _VALID_TYPES, _VALID_TYPE_EMBS, _VALID_TYPE_NORMS = [], None, None
        return [], None, None
    d = pickle.load(open(p, "rb"))
    _VALID_TYPES = d["types"]
    _VALID_TYPE_EMBS = d["embeddings"].astype(np.float32)
    n = np.linalg.norm(_VALID_TYPE_EMBS, axis=1)
    n[n < 1e-8] = 1.0
    _VALID_TYPE_NORMS = n
    return _VALID_TYPES, _VALID_TYPE_EMBS, _VALID_TYPE_NORMS

_VALID_TYPES = None
_VALID_TYPE_EMBS = None
_VALID_TYPE_NORMS = None


def _resolve_category_to_valid_types(category: str, top_k: int = 5) -> list[str]:
    if not category:
        return []
    cat_lc = category.lower().strip()
    valid_types, embs, norms = _load_valid_types()

    if cat_lc in [t.lower() for t in valid_types]:
        return [t for t in valid_types if t.lower() == cat_lc]

    aliases = CATEGORY_ALIASES.get(cat_lc, [])
    matched = [t for t in valid_types if t.lower() in {a.lower() for a in aliases}]
    if matched:
        return matched[:top_k]

    if embs is None:
        return [cat_lc]
    try:
        from openai import OpenAI
        import os
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small",
                                        input=[category.replace("_", " ")])
        q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_emb) or 1.0
        sims = (embs @ q_emb) / (norms * q_norm)
        top_idx = np.argsort(-sims)[:top_k]
        return [valid_types[i] for i in top_idx]
    except Exception:
        return [cat_lc]


def _category_bonus_indices(pool: List[Dict[str, Any]], category: str | None, k: int) -> np.ndarray:
    if not category or k <= 0:
        return np.array([], dtype=int)
    matched_types = _resolve_category_to_valid_types(category, top_k=5)
    if not matched_types:
        return np.array([], dtype=int)
    matched_set = {t.lower() for t in matched_types}
    legacy = {a.lower() for a in CATEGORY_ALIASES.get(category.lower(), [])}
    matched_set |= legacy
    idx = []
    for i, p in enumerate(pool):
        cat = str(p.get("category") or p.get("primaryType") or "").lower()
        if cat in matched_set:
            idx.append(i)
            if len(idx) >= k:
                break
    return np.asarray(idx, dtype=int)


def select_spatial_candidates(
    pool: List[Dict[str, Any]],
    intent: dict,
    R_tau: dict,
    top_k: int = CANDIDATE_TOPK_SPATIAL,
) -> List[int]:
    mode = intent.get("spatial", {}).get("mode", "none")
    lats, lons, valid = _pool_arrays(pool)
    scores = np.zeros(len(pool), dtype=np.float32)

    if mode == "point" and R_tau.get("point"):
        pt = R_tau["point"]
        d = haversine_vec(lats, lons, pt.get("lat"), pt.get("lon"))
        scores = np.maximum(0.0, 1.0 - d / R_POINT_KM).astype(np.float32)
        scores[~valid] = 0.0
        return topk_indices(scores, top_k).tolist()

    if mode == "zone" and R_tau.get("zone"):
        z = R_tau["zone"]
        denom = ZONE_ALPHA * max(float(z.get("radius_km") or 0.0), 0.5)
        d = haversine_vec(lats, lons, z.get("center_lat"), z.get("center_lon"))
        scores = np.maximum(0.0, 1.0 - d / denom).astype(np.float32)
        scores[~valid] = 0.0
        return topk_indices(scores, top_k).tolist()

    if mode == "route" and R_tau.get("route") and R_tau["route"].get("points"):
        route = R_tau["route"]["points"]
        approx = np.full(len(pool), np.inf, dtype=np.float32)
        for rp in route:
            if not valid_latlon(rp.get("lat"), rp.get("lon")):
                continue
            approx = np.minimum(approx, haversine_vec(lats, lons, rp["lat"], rp["lon"]).astype(np.float32))
        approx[~valid] = np.inf
        pre_k = min(len(pool), max(top_k * 4, 10000))
        pre_idx = topk_indices(-approx, pre_k)

        exact_scores = []
        for i in pre_idx:
            p = pool[int(i)]
            d = dist_to_polyline(p.get("lat"), p.get("lon"), route)
            if not np.isfinite(d):
                exact_scores.append(0.0)
            else:
                s = 1.0 / (1.0 + np.exp((d - R_ROUTE_KM) / (R_ROUTE_KM * 0.5)))
                exact_scores.append(float(s))
        exact_scores = np.asarray(exact_scores, dtype=np.float32)
        keep_local = topk_indices(exact_scores, min(top_k, len(pre_idx)))
        return [int(pre_idx[j]) for j in keep_local]

    return []


def select_semantic_candidates(
    pool: List[Dict[str, Any]],
    q_emb: np.ndarray,
    poi_emb_cache: dict | None = None,
    poi_emb_matrix: np.ndarray | None = None,
    top_k: int = CANDIDATE_TOPK_SEMANTIC,
) -> List[int]:
    if poi_emb_matrix is None:
        if poi_emb_cache is None:
            return []
        poi_emb_matrix, _ = _emb_matrix_from_cache(pool, poi_emb_cache)
    sims = cos_sim_vec(q_emb, poi_emb_matrix, clip01=True)
    return topk_indices(sims, top_k).tolist()


def select_preference_candidates(
    pool: List[Dict[str, Any]],
    R_tau: dict,
    poi_emb_cache: dict | None = None,
    poi_emb_matrix: np.ndarray | None = None,
    top_k: int = CANDIDATE_TOPK_PREFERENCE,
) -> List[int]:
    pref = R_tau.get("preference")
    if not pref or pref.get("embedding") is None:
        return []
    pref_emb = np.asarray(pref["embedding"], dtype=np.float32)
    if np.linalg.norm(pref_emb) < 1e-8:
        return []
    if poi_emb_matrix is None:
        if poi_emb_cache is None:
            return []
        poi_emb_matrix, _ = _emb_matrix_from_cache(pool, poi_emb_cache)
    sims = cos_sim_vec(pref_emb, poi_emb_matrix, clip01=True)
    return topk_indices(sims, top_k).tolist()


def select_candidates(
    pool: List[Dict[str, Any]],
    intent: dict,
    R_tau: dict,
    q_emb: np.ndarray,
    poi_emb_cache: dict | None = None,
    poi_emb_matrix: np.ndarray | None = None,
    top_k_total: int = CANDIDATE_TOPK_TOTAL,
    top_k_spatial: int = CANDIDATE_TOPK_SPATIAL,
    top_k_semantic: int = CANDIDATE_TOPK_SEMANTIC,
    top_k_preference: int = CANDIDATE_TOPK_PREFERENCE,
    include_category_bonus: bool = True,
) -> List[Dict[str, Any]]:
    if not pool:
        return []

    idx_spa = select_spatial_candidates(pool, intent, R_tau, top_k=top_k_spatial)
    idx_sem = select_semantic_candidates(pool, q_emb, poi_emb_cache, poi_emb_matrix, top_k=top_k_semantic)
    idx_pref = select_preference_candidates(pool, R_tau, poi_emb_cache, poi_emb_matrix, top_k=top_k_preference)

    idx_cat = []
    if include_category_bonus:
        cat = intent.get("target", {}).get("category")
        idx_cat = _category_bonus_indices(pool, cat, k=100000).tolist()

    seen = set()
    indices = []

    anchor_lat = anchor_lon = None
    if R_tau.get("point"):
        anchor_lat = R_tau["point"].get("lat"); anchor_lon = R_tau["point"].get("lon")
    elif R_tau.get("zone"):
        anchor_lat = R_tau["zone"].get("center_lat"); anchor_lon = R_tau["zone"].get("center_lon")
    elif R_tau.get("route") and R_tau["route"].get("points"):
        anchor_lat = R_tau["route"]["points"][0].get("lat"); anchor_lon = R_tau["route"]["points"][0].get("lon")

    if anchor_lat is not None and idx_cat:
        from utils import haversine_vec
        cat_arr = np.asarray(idx_cat, dtype=int)
        cat_lats = np.array([float(pool[i].get("lat")) if valid_latlon(pool[i].get("lat"), pool[i].get("lon")) else np.nan for i in cat_arr])
        cat_lons = np.array([float(pool[i].get("lon")) if valid_latlon(pool[i].get("lat"), pool[i].get("lon")) else np.nan for i in cat_arr])
        cat_dists = haversine_vec(cat_lats, cat_lons, anchor_lat, anchor_lon)
        cat_dists = np.where(np.isfinite(cat_dists), cat_dists, np.inf)
        order = np.argsort(cat_dists)
        idx_cat_sorted = cat_arr[order].tolist()
        for i in idx_cat_sorted[:500]:
            i = int(i)
            if i not in seen:
                seen.add(i); indices.append(i)

    for idxs in (idx_spa, idx_sem, idx_pref, idx_cat):
        for i in idxs:
            i = int(i)
            if i not in seen:
                seen.add(i); indices.append(i)
            if len(indices) >= top_k_total:
                break
        if len(indices) >= top_k_total:
            break

    if not indices:
        indices = list(range(min(top_k_total, len(pool))))

    return _indices_to_candidates(pool, indices[:top_k_total])


import re as _re_b2
_DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
_BUCKET_HOURS = {"breakfast":(6,10),"lunch":(11,14),"afternoon":(14,18),"dinner":(17,21),
                 "evening":(18,22),"morning":(6,12),"late_night":(22,30),"open_now":None}

def _parse_hours_text(txt: str, day_idx: int) -> list[tuple[int,int]] | None:
    if not txt or txt.lower() in {"nan","none",""}:
        return None
    day_name = _DAYS[day_idx]
    lines = [l for l in str(txt).split("\n") if day_name in l.lower()]
    if not lines:
        return None
    out = []
    for line in lines:
        for m in _re_b2.finditer(r"(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*[\u2013\u2014\-~to]+\s*(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?", line):
            h1, _, m1, h2, _, m2 = m.groups()
            h1=int(h1); h2=int(h2)
            if m1 and m1.lower()=="pm" and h1!=12: h1+=12
            if m1 and m1.lower()=="am" and h1==12: h1=0
            if m2 and m2.lower()=="pm" and h2!=12: h2+=12
            if m2 and m2.lower()=="am" and h2==12: h2=0
            out.append((h1,h2))
    return out or None

def filter_by_opening_hours(pool_indices: list[int], pool, time_bucket: str|None, day_hint: str|None) -> list[int]:
    if not time_bucket or time_bucket not in _BUCKET_HOURS or _BUCKET_HOURS[time_bucket] is None:
        return pool_indices
    bs, be = _BUCKET_HOURS[time_bucket]
    day_idx = 0
    if day_hint:
        dh = day_hint.lower()
        for i,d in enumerate(_DAYS):
            if d in dh:
                day_idx = i; break
    kept = []
    dropped = 0
    for idx in pool_indices:
        p = pool[idx]
        hrs = p.get("opening_hours") or p.get("regularOpeningHours_text")
        intervals = _parse_hours_text(hrs, day_idx)
        if intervals is None:
            kept.append(idx); continue
        if any((s <= be and e >= bs) for s,e in intervals):
            kept.append(idx)
        else:
            dropped += 1
    return kept
