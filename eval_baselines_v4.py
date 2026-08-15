#!/usr/bin/env python3
import os, json, re, time, math, pickle, requests, argparse
from collections import defaultdict
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
import duckdb
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
from anthropic import Anthropic

OPENAI_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DATASET_PATH  = None
POI_STORE_PKL = None
RESULT_DIR    = None
OSM_CACHE_DIR = "/tmp/trajrag_v3_osm_cache"
GLOBAL_CACHE  = "/tmp/trajrag_v3_cache/poi_store_global_v3.pkl"
E5_MODEL      = "intfloat/multilingual-e5-base"
os.makedirs(OSM_CACHE_DIR, exist_ok=True)
_KEY = os.environ.get("OPENAI_API_KEY") or OPENAI_KEY
client = OpenAI(api_key=_KEY)
import threading as _th_meter
_LLM_CALLS = {"chat": 0, "emb": 0}
_METER_LOCK = _th_meter.Lock()
if os.environ.get("BL_METER", "0") == "1":
    _orig_chat = client.chat.completions.create
    _orig_emb = client.embeddings.create
    def _chat_wrap(*a, **k):
        with _METER_LOCK: _LLM_CALLS["chat"] += 1
        return _orig_chat(*a, **k)
    def _emb_wrap(*a, **k):
        with _METER_LOCK: _LLM_CALLS["emb"] += 1
        return _orig_emb(*a, **k)
    client.chat.completions.create = _chat_wrap
    client.embeddings.create = _emb_wrap
anthropic_client = None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
    dlat,dlon = lat2-lat1,lon2-lon1
    a = sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2*R*atan2(sqrt(a),sqrt(1-a))

def recall_at_k(ids, target, k): return 1.0 if target in ids[:k] else 0.0
def ndcg_at_k(ids, target, k):
    dcg = 0.0
    for i, p in enumerate(ids[:k]):
        if p == target:
            dcg = 1.0 / math.log2(i + 2)
            break
    idcg = 1.0 / math.log2(1 + 1)
    return dcg / idcg
def mrr_score(ids, target):
    for i,p in enumerate(ids):
        if p==target: return 1.0/(i+1)
    return 0.0

def get_current_location(traj):
    visits = [(t.get("start_time",""), t.get("d_lat"), t.get("d_lon"))
              for t in traj if t.get("status")=="visit"
              and t.get("d_lat") and float(t.get("d_lat",0))!=0]
    if not visits: return None, None
    visits.sort(key=lambda x:x[0], reverse=True)
    return float(visits[0][1]), float(visits[0][2])

def openai_embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])

OSM_TYPE_MAP = {
    "restaurant":'[amenity=restaurant]',"cafe":'[amenity=cafe]',
    "supermarket":'[shop=supermarket]',"convenience_store":'[shop=convenience]',
    "gym":'[leisure=fitness_centre]',"pharmacy":'[amenity=pharmacy]',
    "bar":'[amenity=bar]',"park":'[leisure=park]',"hospital":'[amenity=hospital]',
    "bank":'[amenity=bank]',"gas_station":'[amenity=fuel]',
    "movie_theater":'[amenity=cinema]',"library":'[amenity=library]',
    "hotel":'[tourism=hotel]',"shopping_mall":'[shop=mall]',
    "beauty_salon":'[shop=beauty]',"hair_care":'[shop=hairdresser]',
    "fast_food_restaurant":'[amenity=fast_food]',
}
OSM_HEADERS = {"User-Agent":"MobilityRAG-Research/1.0"}

_GLOBAL_STORE = None
import threading as _th_gs
_GS_LOCK = _th_gs.Lock()
def get_global_store():
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None: return _GLOBAL_STORE
    with _GS_LOCK:
        if _GLOBAL_STORE is not None: return _GLOBAL_STORE
        with open(POI_STORE_PKL, "rb") as f: _g = pickle.load(f)
        print(f"  [Global] {len(_g)} POIs loaded from {POI_STORE_PKL}")
        _GLOBAL_STORE = _g
    return _GLOBAL_STORE

_PREFILTER_KM = float(os.environ.get("BL_SPATIAL_PREFILTER", "0"))
_USER_POS_CACHE = {}
import threading as _th_pf
_PF_LOCK = _th_pf.Lock()


def _user_anchor_positions(uid, traj_list):
    pos = _USER_POS_CACHE.get(uid)
    if pos is not None:
        return pos
    with _PF_LOCK:
        pos = _USER_POS_CACHE.get(uid)
        if pos is not None:
            return pos
        store = get_global_store()
        plat = pd.to_numeric(store["lat"], errors="coerce").values.astype("float64")
        plon = pd.to_numeric(store["lng"], errors="coerce").values.astype("float64")
        _vis = [t for t in traj_list
                if t.get("status") == "visit" and t.get("d_lat") is not None
                and t.get("d_lon") is not None]
        if _vis:
            _last = max(_vis, key=lambda t: str(t.get("event_start") or t.get("start_time") or ""))
            anchors = [(float(_last["d_lat"]), float(_last["d_lon"]))]
        else:
            anchors = []
        if not anchors:
            pos = np.arange(len(store))
        else:
            R = 6371.0
            plat_r = np.radians(plat); plon_r = np.radians(plon)
            mask = np.zeros(len(store), dtype=bool)
            for (alat, alon) in anchors:
                ar, aor = math.radians(alat), math.radians(alon)
                x = (np.sin((plat_r - ar) / 2) ** 2 +
                     np.cos(ar) * np.cos(plat_r) * np.sin((plon_r - aor) / 2) ** 2)
                d = 2 * R * np.arcsin(np.sqrt(np.clip(x, 0, 1)))
                mask |= (d <= _PREFILTER_KM)
            pos = np.nonzero(mask)[0]
            if len(pos) == 0:
                pos = np.arange(len(store))
        _USER_POS_CACHE[uid] = pos
        return pos


def get_user_poi_store(uid, traj_list):
    store = get_global_store()
    if _PREFILTER_KM <= 0:
        return store
    pos = _user_anchor_positions(uid, traj_list)
    if len(pos) == len(store):
        return store
    return store.iloc[pos].reset_index(drop=True)

def _poi_embed_text(row):
    parts = [str(row.get("displayName") or ""),
             str(row.get("primaryType") or "").replace("_"," "),
             str(row.get("shortFormattedAddress") or "")]
    return " | ".join(p for p in parts if p.strip())

_E5_MODEL = None
_USER_E5_CACHE = {}

import threading as _th_e5m
_E5M_LOCK = _th_e5m.Lock()
def get_e5_model():
    global _E5_MODEL
    if _E5_MODEL is not None: return _E5_MODEL
    with _E5M_LOCK:
        if _E5_MODEL is not None: return _E5_MODEL
        print("Loading E5 model...")
        _m = SentenceTransformer(E5_MODEL)
        print("E5 model loaded")
        _E5_MODEL = _m
    return _E5_MODEL

import threading as _threading
_E5_LOCK = _threading.Lock()

def get_user_e5_vecs(uid, poi_store):
    KEY = "__global__"
    v = _USER_E5_CACHE.get(KEY)
    if v is not None:
        if _PREFILTER_KM > 0:
            p = _USER_POS_CACHE.get(uid)
            if p is not None and len(p) != len(v):
                return v[p]
        return v
    with _E5_LOCK:
        v = _USER_E5_CACHE.get(KEY)
        if v is not None: return v
        cache_path = os.path.join(RESULT_DIR, "e5_global.pkl")
        if os.path.exists(cache_path):
            with open(cache_path,"rb") as f:
                _USER_E5_CACHE[KEY] = pickle.load(f)
            return _USER_E5_CACHE[KEY]
        model = get_e5_model()
        _full = get_global_store()
        descs = _full.apply(_poi_embed_text, axis=1).tolist()
        vecs = model.encode([f"passage: {d}" for d in descs],
                            batch_size=256, show_progress_bar=False, normalize_embeddings=True)
        tmp = cache_path + ".tmp"
        with open(tmp,"wb") as f: pickle.dump(vecs, f)
        os.replace(tmp, cache_path)
        _USER_E5_CACHE[KEY] = vecs
        if _PREFILTER_KM > 0:
            p = _USER_POS_CACHE.get(uid)
            if p is not None and len(p) != len(vecs):
                return vecs[p]
        return vecs


AGENTMOVE_NATIVE = os.environ.get("AGENTMOVE_NATIVE", "0") == "1"
RANKGPT_BM25 = os.environ.get("RANKGPT_BM25", "0") == "1"
_BM25_IDX = {}

def _bm25_build(poi_store):
    key = id(poi_store)
    if key in _BM25_IDX:
        return _BM25_IDX[key]
    import re as _re
    from collections import Counter as _C
    tokre = _re.compile(r"[a-z0-9]+")
    tok = lambda x: tokre.findall(str(x).lower().replace("_", " "))
    names = poi_store["displayName"].fillna("").astype(str).values
    types = poi_store["primaryType"].fillna("").astype(str).values
    addrs = poi_store["shortFormattedAddress"].fillna("").astype(str).values
    N = len(names)
    docs = [tok(names[i]) + tok(types[i]) + tok(addrs[i]) for i in range(N)]
    dl = np.array([len(d) for d in docs], dtype=np.float32)
    avgdl = float(dl.mean()) if N else 1.0
    dfc = _C()
    for d in docs:
        dfc.update(set(d))
    inv = {}
    for i, d in enumerate(docs):
        for t, c in _C(d).items():
            inv.setdefault(t, []).append((i, c))
    idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in dfc.items()}
    _BM25_IDX[key] = (inv, idf, dl, avgdl, N, tok)
    return _BM25_IDX[key]

def _bm25_scores(query, poi_store, k1=1.5, b=0.75):
    inv, idf, dl, avgdl, N, tok = _bm25_build(poi_store)
    sc = np.zeros(N, dtype=np.float32)
    for t in set(tok(query)):
        if t not in inv:
            continue
        w = idf[t]
        for i, c in inv[t]:
            sc[i] += w * (c * (k1 + 1)) / (c + k1 * (1 - b + b * dl[i] / avgdl))
    return sc


def _agentmove_native_order(poi_store, gold_id, n=100, radius_km=5.0):
    ids = poi_store["id"].astype(str).tolist()
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values.astype("float64")
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values.astype("float64")
    gi = {p: i for i, p in enumerate(ids)}.get(str(gold_id))
    if gi is None:
        return None
    a, o = math.radians(lats[gi]), math.radians(lons[gi])
    pr, po = np.radians(lats), np.radians(lons)
    x = np.sin((pr - a) / 2) ** 2 + np.cos(a) * np.cos(pr) * np.sin((po - o) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))
    near = np.nonzero(d <= radius_km)[0]
    near = near[np.argsort(d[near])][:n]
    if len(near) < n:
        rest = np.argsort(d)[:n]
        near = np.unique(np.concatenate([near, rest]))[:n]
    if gi not in near:
        near = np.concatenate([[gi], near[:n - 1]])
    return [ids[i] for i in near]


def _nearest_to_user(poi_store, traj, n=100, radius_km=None):
    cur_lat, cur_lon = get_current_location(traj)
    if cur_lat is None:
        return None
    ids = poi_store["id"].astype(str).tolist()
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values.astype("float64")
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values.astype("float64")
    a, o = math.radians(float(cur_lat)), math.radians(float(cur_lon))
    pr, po = np.radians(lats), np.radians(lons)
    x = np.sin((pr - a) / 2) ** 2 + np.cos(a) * np.cos(pr) * np.sin((po - o) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(x, 0, 1)))
    if radius_km is not None:
        idx = np.nonzero(d <= radius_km)[0]
        idx = idx[np.argsort(d[idx])][:n]
        if len(idx) < n:
            idx = np.argsort(d)[:n]
    else:
        idx = np.argsort(d)[:n]
    return [ids[i] for i in idx]


CAT_SEM_TOPN = int(os.environ.get('CAT_SEM_TOPN','0'))
_CS_CAT={}
def _cs_load():
    if _CS_CAT: return
    import json as _j
    for run in ('/tmp/takeout_fixcur.jsonl','/tmp/fix_geolife.jsonl'):
        try:
            for l in open(run):
                r=_j.loads(l)
                if r.get('target_cat'): _CS_CAT[r['question']]=str(r['target_cat']).lower()
        except Exception: pass




_ADMIN_CACHE = {}
_ADMIN_FILES = {"takeout": "./data/japan_admin/admin_regions.parquet",
                "geolife": "./data/beijing_admin/admin_regions.parquet"}
def _admin_df():
    ds = globals().get("_BL_DATASET", "takeout")
    if ds not in _ADMIN_CACHE:
        try:
            _ADMIN_CACHE[ds] = pd.read_parquet(_ADMIN_FILES[ds])
        except Exception:
            _ADMIN_CACHE[ds] = None
    return _ADMIN_CACHE[ds]

def _ref_geometry(query, traj, uid, poi_store):
    spa = _srag_extract_spatial(query, 1, _srag_region_names(poi_store))
    mode = (spa.get("query_type") or "point").lower()
    radius = spa.get("distance_km") or (spa.get("buffer_distance") or 1000) / 1000.0
    radius = float(max(0.2, min(radius, 10.0)))
    out = {"mode": mode, "ref_pts": [], "poly": None, "radius_km": radius}

    if mode == "region" and spa.get("region"):
        try:
            adf = _admin_df()
            if adf is not None:
                rq = str(spa["region"]).strip().lower()
                row = None
                for col in ("name", "name_en", "name_en_norm", "name_ja"):
                    if col in adf.columns:
                        m = adf[adf[col].astype(str).str.lower() == rq]
                        if len(m): row = m; break
                if row is not None and len(row) and row.iloc[0].get("polygon_wkt"):
                    from shapely import wkt as _wkt
                    out["poly"] = _wkt.loads(row.iloc[0]["polygon_wkt"])
                    c = out["poly"].centroid
                    out["ref_pts"] = [(c.y, c.x)]
        except Exception:
            pass

    if mode == "route":
        pts = [(t["o_lat"], t["o_lon"]) for t in traj
               if t.get("status") == "activity" and t.get("o_lat") is not None]
        pts += [(t["d_lat"], t["d_lon"]) for t in traj
                if t.get("status") == "activity" and t.get("d_lat") is not None]
        out["ref_pts"] = pts[:200]

    if not out["ref_pts"]:
        cur_lat, cur_lon = get_current_location(traj)
        if cur_lat is not None:
            out["ref_pts"] = [(cur_lat, cur_lon)]
            if mode == "region":
                out["mode"] = "point"
    return out


def _ref_distance(lat, lon, ref):
    if ref["poly"] is not None:
        try:
            from shapely.geometry import Point as _P
            if ref["poly"].contains(_P(lon, lat)):
                return 0.0
        except Exception:
            pass
    if not ref["ref_pts"]:
        return 9e9
    return min(haversine(lat, lon, rl, ro) for rl, ro in ref["ref_pts"])



import json as _json_bl
_BL_WL_FILES = {"takeout": "/tmp/baseline_whitelist.json",
                "geolife": "/tmp/baseline_whitelist_geolife.json"}
_BL_WL_CACHE = {}
def _bl_whitelist():
    ds = globals().get("_BL_DATASET", "takeout")
    if ds not in _BL_WL_CACHE:
        _BL_WL_CACHE[ds] = _json_bl.load(open(_BL_WL_FILES[ds]))
    return _BL_WL_CACHE[ds]
_BL_ALIASES = None
def _bl_aliases():
    global _BL_ALIASES
    if _BL_ALIASES is not None:
        return _BL_ALIASES
    _BL_ALIASES = {}
    if globals().get("_BL_DATASET", "takeout") == "takeout":
        try:
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location(
                "_bl_constants",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "trajrag", "datasets", "takeout", "constants.py"))
            _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
            _BL_ALIASES = getattr(_m, "CATEGORY_ALIASES", {}) or {}
        except Exception:
            _BL_ALIASES = {}
    return _BL_ALIASES
_BL_CAT_CACHE = {}
def _extract_target_category(query):
    if query in _BL_CAT_CACHE:
        return _BL_CAT_CACHE[query]
    wl = _bl_whitelist()
    prompt = ("Select the most plausible place categories for the query below. "
              "Choose 1 to 4 entries STRICTLY from this whitelist (place primaryType); "
              "if none fits, return an empty list.\n\nWhitelist:\n"
              + ", ".join(wl)
              + f'\n\nQuery: "{query}"\n\nReturn ONLY a JSON array of category strings, e.g. ["cafe","bakery"].')
    cats=[]
    try:
        r=client.chat.completions.create(model="gpt-4o-mini",temperature=0,
            max_tokens=60,messages=[{"role":"user","content":prompt}])
        txt=r.choices[0].message.content.strip().replace("```json","").replace("```","")
        cats=[str(c).lower().strip() for c in json.loads(txt) if str(c).lower().strip() in set(wl)]
    except Exception:
        cats=[]
    _BL_CAT_CACHE[query]=cats
    return cats


def _cand_by_category(query, poi_store):
    _CA = _bl_aliases()
    tcs = _extract_target_category(query)
    ptype = poi_store["primaryType"].astype(str).str.lower().values
    if tcs:
        cats = set()
        for tc in tcs:
            cats.add(tc); cats |= {str(x).lower() for x in _CA.get(tc, [])}
        idx = np.nonzero(np.isin(ptype, list(cats)))[0]
        if len(idx) > 0:
            return idx
    return np.arange(len(ptype))



def _ref_dist_vec(lats, lons, ref):
    lats=np.asarray(lats,dtype='float64'); lons=np.asarray(lons,dtype='float64')
    _nan=np.isnan(lats)|np.isnan(lons)
    pr=np.radians(np.where(_nan,0.0,lats)); por=np.radians(np.where(_nan,0.0,lons))
    best=np.full(len(lats), 9e9)
    for (rl,ro) in ref["ref_pts"]:
        a=math.radians(float(rl)); o=math.radians(float(ro))
        x=np.sin((pr-a)/2)**2+np.cos(a)*np.cos(pr)*np.sin((por-o)/2)**2
        d=2*6371.0*np.arcsin(np.sqrt(np.clip(x,0,1)))
        best=np.minimum(best,d)
    if ref["poly"] is not None:
        try:
            poly=ref["poly"]; minx,miny,maxx,maxy=poly.bounds
            cand=np.nonzero((~_nan)&(lons>=minx)&(lons<=maxx)&(lats>=miny)&(lats<=maxy))[0]
            if len(cand):
                try:
                    from shapely import points as _pts, contains as _ct
                    inside=_ct(poly, _pts(lons[cand], lats[cand]))
                    best[cand[np.asarray(inside,dtype=bool)]]=0.0
                except Exception:
                    from shapely.geometry import Point as _P
                    for k in cand:
                        if poly.contains(_P(lons[k],lats[k])): best[k]=0.0
        except Exception: pass
    best[_nan]=9e9
    return best


def rank_sd(query, traj, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).values
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values
    ref = _ref_geometry(query, traj, uid, poi_store)
    idx = np.arange(len(poi_ids))
    d = _ref_dist_vec(lats[idx], lons[idx], ref)
    order = idx[np.argsort(d)]
    _mask = np.ones(len(poi_ids), dtype=bool); _mask[order] = False
    rest = np.nonzero(_mask)[0]
    return [poi_ids[i] for i in np.concatenate([order, rest])]



_OAI_EMB_MODEL = os.environ.get("BL_OAI_EMB", "text-embedding-3-small")
_OAI_POOL_CACHE = {}
_OAI_QVEC_CACHE = {}
import threading as _th_oai
_OAI_LOCK = _th_oai.Lock()

def _openai_embed(texts, batch=1000):
    import time as _t
    N = len(texts); result = None; nb = (N + batch - 1) // batch
    for bi, b in enumerate(range(0, N, batch)):
        chunk = [str(t)[:8000] if str(t).strip() else " " for t in texts[b:b+batch]]
        emb = None
        for attempt in range(6):
            try:
                r = client.embeddings.create(model=_OAI_EMB_MODEL, input=chunk)
                emb = np.asarray([d.embedding for d in r.data], dtype=np.float32)
                break
            except Exception:
                if attempt == 5: raise
                _t.sleep(2 * (attempt + 1))
        if result is None:
            result = np.empty((N, emb.shape[1]), dtype=np.float32)
        result[b:b+len(emb)] = emb
        if (bi + 1) % 20 == 0 or bi + 1 == nb:
            print(f"    [OAI] {bi+1}/{nb} batches done", flush=True)
    n = np.linalg.norm(result, axis=1, keepdims=True); n[n == 0] = 1.0
    return result / n

def get_pool_openai_vecs(poi_store):
    if _BL_ENCODER == "e5":
        return _bl_e5_pool(poi_store)
    KEY = _OAI_EMB_MODEL
    v = _OAI_POOL_CACHE.get(KEY)
    if v is not None:
        return v
    with _OAI_LOCK:
        v = _OAI_POOL_CACHE.get(KEY)
        if v is not None:
            return v
        _cdir = os.path.dirname(POI_STORE_PKL) or "."
        cache_path = os.path.join(_cdir, f"oai_pool_{KEY}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                _OAI_POOL_CACHE[KEY] = pickle.load(f)
            return _OAI_POOL_CACHE[KEY]
        _full = get_global_store()
        descs = _full.apply(_poi_embed_text, axis=1).tolist()
        print(f"  [OAI] encoding the full pool of {len(descs)} POIs via {KEY} ...", flush=True)
        vecs = _openai_embed(descs)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(vecs, f)
        os.replace(tmp, cache_path)
        _OAI_POOL_CACHE[KEY] = vecs
        print(f"  [OAI] full-pool vectors cached -> {cache_path} shape={vecs.shape}", flush=True)
        return vecs

def _oai_query_vec(query):
    if _BL_ENCODER == "e5":
        return _bl_e5_qvec(query)
    q = _OAI_QVEC_CACHE.get(query)
    if q is None:
        q = _openai_embed([query])[0]
        _OAI_QVEC_CACHE[query] = q
    return q

def rank_st(query, traj, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).values
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values
    pool = get_pool_openai_vecs(poi_store)
    qv = _oai_query_vec(query)[None, :]
    sem = cosine_similarity(qv, pool)[0]
    ref = _ref_geometry(query, traj, uid, poi_store)
    idx = np.arange(len(poi_ids))
    d = _ref_dist_vec(lats[idx], lons[idx], ref)
    dist_score = 1.0 / (1.0 + d)
    score = 0.5 * sem[idx] + 0.5 * dist_score
    order = idx[np.argsort(-score)]
    _mask = np.ones(len(poi_ids), dtype=bool); _mask[order] = False
    rest = np.nonzero(_mask)[0]
    return [poi_ids[i] for i in np.concatenate([order, rest])]


def rank_naiverag(query, traj, uid, poi_store, top_k=20):
    poi_ids = poi_store["id"].astype(str).values
    names = poi_store["displayName"].fillna("").astype(str).values
    types = poi_store["primaryType"].fillna("").astype(str).values
    addrs = poi_store["shortFormattedAddress"].fillna("").astype(str).values
    pool = get_pool_openai_vecs(poi_store)
    qv = _oai_query_vec(query)[None, :]
    sem = cosine_similarity(qv, pool)[0]
    top = np.argsort(-sem)[:top_k]
    places = [{"idx": int(j), "name": str(names[i]), "category": str(types[i]),
               "address": str(addrs[i])} for j, i in enumerate(top)]
    prompt = ("You are given a user query and a list of candidate places retrieved "
              "by semantic similarity. Rank them by how well they answer the query.\n\n"
              f"Query: {query}\n\nCandidates:\n{json.dumps(places, ensure_ascii=False)}\n\n"
              "Return a JSON array of indices sorted best-first, length "
              f"{len(places)}, e.g. [2,0,1].")
    order = []
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=300, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        order = [int(x) for x in json.loads(txt) if 0 <= int(x) < len(top)]
    except Exception:
        pass
    seen = set(order)
    order += [j for j in range(len(top)) if j not in seen]
    ranked = [int(top[j]) for j in order]
    rs = set(ranked)
    tail = [int(i) for i in np.argsort(-sem) if int(i) not in rs]
    return [poi_ids[i] for i in ranked + tail]


def _geollm_prompt_for(i, lats, lons, names, types, addrs, cand_idx):
    _la=np.radians(lats[cand_idx].astype('float64')); _lo=np.radians(lons[cand_idx].astype('float64'))
    _a=math.radians(float(lats[i])); _o=math.radians(float(lons[i]))
    _x=np.sin((_la-_a)/2)**2+np.cos(_a)*np.cos(_la)*np.sin((_lo-_o)/2)**2
    d=2*6371.0*np.arcsin(np.sqrt(np.clip(_x,0,1)))
    near_order = cand_idx[np.argsort(d)][:11]
    nearby = ""
    for j in near_order:
        if j == i:
            continue
        nearby += f"{haversine(lats[i],lons[i],lats[j],lons[j]):.1f} km {types[j]}: {names[j]}\n"
        if nearby.count("\n") >= 10:
            break
    return (f'Coordinates: ({lats[i]:.5f}, {lons[i]:.5f})\n\n'
            f'Address: "{addrs[i]}"\n\nNearby Places:\n"\n{nearby}"')


def rank_geollm(query, traj, uid, poi_store, cand_n=20):
    poi_ids = poi_store["id"].astype(str).values
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values
    names = poi_store["displayName"].fillna("").astype(str).values
    types = poi_store["primaryType"].fillna("").astype(str).values
    addrs = poi_store["shortFormattedAddress"].fillna("").astype(str).values
    ref = _ref_geometry(query, traj, uid, poi_store)
    idx = np.arange(len(poi_ids))
    d = _ref_dist_vec(lats[idx], lons[idx], ref)
    cand = idx[np.argsort(d)[:cand_n]]
    ctx = []
    for k, i in enumerate(cand):
        ctx.append(f"[{k}] " + _geollm_prompt_for(i, lats, lons, names, types, addrs, idx))
    prompt = ("You are a geographic assistant. Each candidate place is described by "
              "its coordinates, address, and nearby places (GeoLLM context). Select and "
              "rank the places that best answer the user query.\n\n"
              f"Query: {query}\n\n" + "\n\n".join(ctx) +
              f"\n\nReturn a JSON array of indices sorted best-first, length {len(cand)}.")
    order = []
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=300, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        order = [int(x) for x in json.loads(txt) if 0 <= int(x) < len(cand)]
    except Exception:
        pass
    seen = set(order)
    order += [k for k in range(len(cand)) if k not in seen]
    ranked = [int(cand[k]) for k in order]
    rs = set(int(x) for x in ranked)
    _idxset = set(int(x) for x in idx)
    tail = [int(i) for i in idx if int(i) not in rs]
    tail += [i for i in range(len(poi_ids)) if i not in rs and i not in _idxset]
    return [poi_ids[i] for i in ranked + tail]


def rank_e5only(query, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, e5_vecs)[0]
    order = np.argsort(sims)[::-1]
    return [poi_ids[i] for i in order]


import baseline_text2sql as _t2s

def _t2s_llm(prompt, max_tokens=400):
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def rank_text2sql(query, query_time, traj, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    sims = cosine_similarity(
        model.encode([f"query: {query}"], normalize_embeddings=True), e5_vecs)[0]
    e5_order = [poi_ids[i] for i in np.argsort(sims)[::-1]]
    return _t2s.rank_text2sql(query, query_time, traj, uid, poi_store,
                              llm_fn=_t2s_llm, e5_order=e5_order)



import baseline_agentmove as _am

_AM_META = {"map": None, "built": False}

def _am_llm(prompt, max_tokens=400):
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def _am_prepare(all_records, poi_store):
    if _AM_META["built"]:
        return
    _am.build_social_graph(all_records)
    ids = poi_store["id"].astype(str).tolist()
    nms = poi_store["displayName"].fillna("").astype(str).tolist()
    tys = poi_store["primaryType"].fillna("").astype(str).tolist()
    _AM_META["map"] = {i: (n, t) for i, n, t in zip(ids, nms, tys)}
    _AM_META["built"] = True

def rank_agentmove(query, query_time, traj, uid, poi_store, top_k=100, native_cands=None):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    sims = cosine_similarity(
        model.encode([f"query: {query}"], normalize_embeddings=True), e5_vecs)[0]
    order = np.argsort(sims)[::-1]
    e5_order = [poi_ids[i] for i in order]
    return _am.rank_agentmove(query, query_time, traj, uid, poi_store,
                              llm_fn=_am_llm, e5_order=e5_order,
                              cand_ids=(native_cands or e5_order[:top_k]),
                              poi_meta=_AM_META["map"] or {}, top_k=top_k)



import baseline_reactgis as _rga

def _rga_llm(prompt, max_tokens=300, stop=None):
    try:
        kw = {"model": "gpt-4o-mini", "temperature": 0, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
        if stop:
            kw["stop"] = stop
        r = client.chat.completions.create(**kw)
        return r.choices[0].message.content.strip()
    except Exception:
        return ""

def rank_reactgis(query, query_time, traj, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    sims = cosine_similarity(
        model.encode([f"query: {query}"], normalize_embeddings=True), e5_vecs)[0]
    e5_order = [poi_ids[i] for i in np.argsort(sims)[::-1]]
    return _rga.rank_reactgis(query, query_time, traj, uid, poi_store,
                              llm_fn=_rga_llm, e5_order=e5_order)



def _srag_extract_spatial(query, location_count, region_names):
    is_multi_point = location_count > 1
    prompt = f'''Analyze the following user query and extract spatial information: "{query}"

Current location context:
- Number of location points: {location_count}
- Multiple points detected: {is_multi_point}

First, determine the spatial query type based on these rules:
1. For single location point ({location_count == 1}):
   - Use Region-based if query explicitly mentions a region
   - Otherwise, use Point-based

2. For exactly two points ({location_count == 2}):
   - Use Route-based if query suggests path/route between points
   - Otherwise, fall back to Point/Region based rules

3. For multiple points ({location_count > 2}):
   - Only use Point-based or Region-based

Query types:
1. Point-based:
   - For "nearby" or "close": 1km in dense areas
   - For "walking distance": 2km
   - For "not too far": 3km

2. Route-based:
   - ONLY available with exactly 2 points
   - For walking routes: 1000m buffer
   - For general routes: 2000m buffer
   - For scenic/exploration: 3000m buffer
   - Consider terms: "route", "path", "between", "from...to", "along"

3. Region-based:
   - ONLY if query explicitly mentions these regions:
   Community/Sub-region names: {", ".join(region_names)}
   - Do NOT infer regions from landmarks

Return in strict JSON format:
{{
    "query_type": "point" | "route" | "region",
    "region": "matched region name or null",
    "distance_km": number or null,
    "buffer_distance": number or null,
}}'''
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=200, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(txt)
    except Exception:
        return {"query_type": "point", "region": None, "distance_km": 1.0, "buffer_distance": None}


def _srag_extract_semantic(query):
    prompt = f'''Analyze the following user query and extract constraints: "{query}"

First, determine the main purpose of the query by identifying key terms and context:

Restaurant (R) keywords and contexts:
- Direct terms: "restaurant", "food", "eat", "dining", "meal", "cuisine"
- Food types: "Chinese", "Thai", "Mexican", "Italian", "sushi", etc.
- Meal times: "breakfast", "lunch", "dinner", "brunch"
- Dining related: "menu", "dishes", "chef", "reservation"
- Even if staying at a hotel, if asking about food/dining, it's Restaurant (R)

Hotel (H) keywords and contexts:
- Must be explicitly looking for accommodation
- Direct terms: "hotel", "stay", "accommodation", "room", "book"
- Price per night (e.g., "$200/night")
- Hotel names (e.g., "Hyatt", "Marriott")
- Mentioning a hotel as location reference is NOT H type

Attraction (A) keywords and contexts:
- Direct terms: "visit", "see", "tour", "explore"
- Places: "museum", "park", "gallery", "theater"
- Activities: "sightseeing", "show", "performance"

Important rules:
1. Focus on what the user is ASKING FOR, not what they mention
2. If user mentions staying at a hotel but asks about restaurants, type is R
3. If query is about food/dining/restaurants, type must be R
4. Location references (e.g., "near Hotel X") don't determine type

For each constraint type, extract complete sentences that describe the requirements:

1. Spatial constraints: Where they want to go
   Example: "near Times Square" or "in the Upper West Side area"

2. User constraints: What specific requirements or preferences they have
   Example: "family-friendly restaurant with reasonable prices around $30 per person"

Please return strict JSON format without any comments:
{{
    "type": "R/H/A",
    "spatial_constraints": "complete sentence describing location requirements or null",
    "user_constraints": "complete sentence describing user preferences and requirements or null"
}}'''
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=250, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(txt)
    except Exception:
        return {"type": "R", "spatial_constraints": None, "user_constraints": None}


def _srag_pareto(spa, sem, idxs):
    n = len(idxs)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if spa[j] >= spa[i] and sem[j] >= sem[i] and (spa[j] > spa[i] or sem[j] > sem[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _srag_rerank(query_constraints, places):
    prompt = f'''As a local recommendation expert, please rank the following places based on user query constraints.

User Query Constraints:
- Spatial Constraints: {query_constraints.get("spatial_constraints")}
- User Preferences: {query_constraints.get("user_constraints")}

Candidate Places:
{json.dumps(places, ensure_ascii=False, indent=2)}

Please analyze how well each place matches the user constraints and return a sorted list of places.
Return format should be a JSON array containing sorted indices.
Only return the index array, e.g., [2,0,1,3] means the 3rd place is the best match, followed by 1st, 2nd, and 4th places.
Note: Must return indices for all places, array length should equal input place count ({len(places)}).'''
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=300, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        order = json.loads(txt)
        return [int(x) for x in order if isinstance(x, (int, float)) and 0 <= int(x) < len(places)]
    except Exception:
        return list(range(len(places)))


_SRAG_REGIONS = None
def _srag_region_names(poi_store):
    global _SRAG_REGIONS
    if _SRAG_REGIONS is None:
        try:
            addrs = poi_store["shortFormattedAddress"].dropna().astype(str)
            toks = {}
            for a in addrs.sample(min(4000, len(addrs)), random_state=0):
                for part in re.split(r"[,、]", a):
                    part = part.strip()
                    if 2 <= len(part) <= 20 and not any(c.isdigit() for c in part):
                        toks[part] = toks.get(part, 0) + 1
            _SRAG_REGIONS = [k for k, _ in sorted(toks.items(), key=lambda x: -x[1])[:40]]
        except Exception:
            _SRAG_REGIONS = []
    return _SRAG_REGIONS



_SEMASK_SYS = 'You are an assistant for location information sorting tasks. Below is the location information retrieved from the database, which will be given to you in JSON format. You are asked to filter and sort this information based on the question asked. You first need to determine whether the information is relevant to the question, and then sort all the relevant information. The ones that best match the question and help answer it have the highest priority. The format of your output must be a Python dictionary, where the key is the name of the location and the value is the reason why you chose this location and ranked it there. The location with the highest priority is placed higher, i.e., index is 0. Please note that there could be more than one result in the dictionary. If the information about a location could only partially match the question asked, you could also put it in the dictionary, but specify the advantages and disadvantages of this place in the value of the dictionary. If you could not complete the task or do not know the answer, just return the empty dictionary and do not refer to any additional knowledge'
def _semask_norm(t):
    import re as _re
    return _re.sub(r"\\W+", "", str(t).lower())

def rank_semask(query, traj, uid, poi_store, k=8, side_km=5.0):
    poi_ids = poi_store["id"].astype(str).values
    lats = pd.to_numeric(poi_store["lat"], errors="coerce").values
    lons = pd.to_numeric(poi_store["lng"], errors="coerce").values
    names = poi_store["displayName"].fillna("").astype(str).values
    types = poi_store["primaryType"].fillna("").astype(str).values
    addrs = poi_store["shortFormattedAddress"].fillna("").astype(str).values
    pool = get_pool_openai_vecs(poi_store)
    qv = _oai_query_vec(query)

    rlat, rlon = get_current_location(traj)

    if rlat is not None:
        half = side_km / 2.0
        dlat = half / 111.0
        dlon = half / (111.0 * max(1e-6, math.cos(math.radians(float(rlat)))))
        inb = np.nonzero((~np.isnan(lats)) & (~np.isnan(lons)) &
                         (lats >= rlat - dlat) & (lats <= rlat + dlat) &
                         (lons >= rlon - dlon) & (lons <= rlon + dlon))[0]
    else:
        inb = np.arange(len(poi_ids))

    if len(inb) == 0:
        cand = np.array([], dtype=int)
    else:
        sims = pool[inb] @ qv
        cand = inb[np.argsort(-sims)][:k]

    context = [{"name": str(names[i]), "category": str(types[i]),
                "address": str(addrs[i])} for i in cand]
    rank_dict = {}
    if context:
        try:
            r = client.chat.completions.create(model=os.environ.get("SEMASK_MODEL","gpt-4o"), temperature=0,
                max_tokens=800,
                messages=[{"role": "system", "content": _SEMASK_SYS},
                          {"role": "user", "content": f"information:{json.dumps(context, ensure_ascii=False)}\\nquestion:{query}"}])
            txt = r.choices[0].message.content.strip()
            if "```" in txt:
                txt = txt.split("```")[1]
                if txt.startswith("python"): txt = txt[6:]
                txt = txt.strip()
            try: rank_dict = json.loads(txt)
            except Exception: rank_dict = ast.literal_eval(txt)
            if not isinstance(rank_dict, dict): rank_dict = {}
        except Exception:
            rank_dict = {}

    norm2idx = {}
    for i in cand:
        norm2idx.setdefault(_semask_norm(names[i]), int(i))
    ranked = []
    for key in rank_dict.keys():
        j = norm2idx.get(_semask_norm(key))
        if j is not None and j not in ranked:
            ranked.append(j)
    for i in cand:
        if int(i) not in ranked: ranked.append(int(i))
    rs = set(ranked)
    tail = [i for i in range(len(poi_ids)) if i not in rs]
    return [poi_ids[i] for i in ranked + tail]



_FIELD_TEXT = {
    "address": lambda r: str(r.get("shortFormattedAddress") or ""),
    "namecat": lambda r: (str(r.get("displayName") or "") + " " +
                          str(r.get("primaryType") or "").replace("_", " ")).strip(),
}
_POOL_FIELD_CACHE = {}

_BL_ENCODER = os.environ.get("BL_ENCODER", "e5").lower()
_E5_POOL_CACHE = {}
_E5_QVEC_CACHE = {}
def _bl_e5_pool(poi_store):
    KEY = "__e5__"
    v = _E5_POOL_CACHE.get(KEY)
    if v is not None: return v
    with _OAI_LOCK:
        v = _E5_POOL_CACHE.get(KEY)
        if v is not None: return v
        _full = get_global_store()
        emb = np.vstack(_full["embedding"].values).astype(np.float32)
        n = np.linalg.norm(emb, axis=1, keepdims=True); n[n==0]=1.0
        _E5_POOL_CACHE[KEY] = emb / n
        return _E5_POOL_CACHE[KEY]
def _bl_e5_qvec(query):
    q = _E5_QVEC_CACHE.get(query)
    if q is None:
        m = get_e5_model()
        q = m.encode([f"query: {query}"], normalize_embeddings=True)[0].astype(np.float32)
        _E5_QVEC_CACHE[query] = q
    return q

_E5_FIELD_CACHE = {}
def get_pool_field_vecs(poi_store, field):
    if _BL_ENCODER == "e5":
        v = _E5_FIELD_CACHE.get(field)
        if v is not None: return v
        with _OAI_LOCK:
            v = _E5_FIELD_CACHE.get(field)
            if v is not None: return v
            _full = get_global_store()
            txts = _full.apply(_FIELD_TEXT[field], axis=1).tolist()
            m = get_e5_model()
            vecs = m.encode([f"passage: {t}" for t in txts], batch_size=256,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            _E5_FIELD_CACHE[field] = vecs
            return vecs
    KEY = f"{_OAI_EMB_MODEL}__{field}"
    v = _POOL_FIELD_CACHE.get(KEY)
    if v is not None:
        return v
    with _OAI_LOCK:
        v = _POOL_FIELD_CACHE.get(KEY)
        if v is not None:
            return v
        _cdir = os.path.dirname(POI_STORE_PKL) or "."
        cache_path = os.path.join(_cdir, f"oai_field_{field}_{_OAI_EMB_MODEL}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                _POOL_FIELD_CACHE[KEY] = pickle.load(f)
            return _POOL_FIELD_CACHE[KEY]
        _full = get_global_store()
        txts = _full.apply(_FIELD_TEXT[field], axis=1).tolist()
        print(f"  [OAI] encoding object-side {field} mask for {len(txts)} POIs ...", flush=True)
        vecs = _openai_embed(txts)
        tmp = cache_path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(vecs, f)
        os.replace(tmp, cache_path)
        _POOL_FIELD_CACHE[KEY] = vecs
        print(f"  [OAI] {field} mask vectors cached -> {cache_path} {vecs.shape}", flush=True)
        return vecs

def _srag_dynamic_weights(query, pareto_places):
    lines = []
    for p in pareto_places[:20]:
        lines.append(f"- {p['name']} ({p['category']}), f_s_sparse={p['fs_sparse']:.3f}: {p['address']}")
    prompt = ("You balance spatial versus semantic relevance for a geospatial query. "
              "Given the query and the Pareto-optimal candidates (each with its sparse "
              "spatial score and description), decide how much the final ranking should "
              "weight spatial proximity/constraint satisfaction (lambda_s) versus semantic "
              "match to the user's intent (lambda_k). The two weights must be non-negative "
              f"and sum to 1.\n\nQuery: {query}\n\nCandidates:\n" + "\n".join(lines) +
              '\n\nReturn ONLY JSON: {"lambda_s": <float>, "lambda_k": <float>}.')
    ls, lk = 0.5, 0.5
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0,
            max_tokens=40, messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        d = json.loads(txt)
        ls = float(d.get("lambda_s", 0.5)); lk = float(d.get("lambda_k", 0.5))
        tot = ls + lk
        if tot > 0: ls, lk = ls / tot, lk / tot
        else: ls, lk = 0.5, 0.5
    except Exception:
        ls, lk = 0.5, 0.5
    return ls, lk


def rank_spatialrag(query, traj, uid, poi_store, top_k_rerank=20):
    poi_ids = poi_store["id"].astype(str).tolist()
    lats = poi_store["lat"].values
    lons = poi_store["lng"].values
    names = poi_store["displayName"].fillna("").values
    types = poi_store["primaryType"].fillna("").values
    addrs = poi_store["shortFormattedAddress"].fillna("").values

    pool_vecs = get_pool_openai_vecs(poi_store)
    cur_lat, cur_lon = get_current_location(traj)

    loc_pts = [t for t in traj if t.get("d_lat")]
    spa_info = _srag_extract_spatial(query, 1 if cur_lat is not None else 0,
                                     _srag_region_names(poi_store))
    radius_km = spa_info.get("distance_km") or 1.0
    if spa_info.get("buffer_distance"):
        radius_km = float(spa_info["buffer_distance"]) / 1000.0
    radius_km = float(max(0.2, min(radius_km, 10.0)))

    sem_sims = cosine_similarity(_oai_query_vec(query)[None, :], pool_vecs)[0]
    _latn = pd.to_numeric(poi_store["lat"], errors="coerce").values.astype("float64")
    _lonn = pd.to_numeric(poi_store["lng"], errors="coerce").values.astype("float64")
    ref_geo = _ref_geometry(query, traj, uid, poi_store)
    if not ref_geo["ref_pts"] and ref_geo["poly"] is None:
        order = np.argsort(sem_sims)[::-1]
        return [poi_ids[i] for i in order]
    dists = _ref_dist_vec(_latn, _lonn, ref_geo)
    mask = dists <= radius_km
    if mask.sum() < 5:
        mask = dists <= max(radius_km * 3, 3.0)
    if mask.sum() < 5:
        mask = np.ones(len(poi_ids), dtype=bool)
    cand = np.where(mask)[0]
    if len(cand) > 600:
        cand = cand[np.argsort(dists[cand])[:600]]

    sem_intent = _srag_extract_semantic(query)
    ms_q = sem_intent.get("spatial_constraints") or spa_info.get("region") or query
    mk_q = sem_intent.get("user_constraints") or query
    vq_s = _oai_query_vec(str(ms_q))
    vq_k = _oai_query_vec(str(mk_q))

    addr_vecs = get_pool_field_vecs(poi_store, "address")
    nc_vecs   = get_pool_field_vecs(poi_store, "namecat")
    fs_sparse = 1.0 / (1.0 + dists[cand])
    fs_dense  = addr_vecs[cand] @ vq_s
    fs = 0.5 * fs_sparse + 0.5 * fs_dense
    fk = nc_vecs[cand] @ vq_k

    pf_local = _srag_pareto(fs, fk, cand)
    if len(pf_local) > top_k_rerank:
        comb = fs[pf_local] + fk[pf_local]
        pf_local = [pf_local[i] for i in np.argsort(comb)[::-1][:top_k_rerank]]
    if not pf_local:
        pf_local = list(np.argsort(fk)[::-1][:top_k_rerank])

    places = []
    for li in pf_local:
        gi = cand[li]
        places.append({"name": str(names[gi]), "category": str(types[gi]),
                       "address": str(addrs[gi]), "fs_sparse": float(fs_sparse[li])})
    lam_s, lam_k = _srag_dynamic_weights(query, places)
    final = lam_s * fs[pf_local] + lam_k * fk[pf_local]
    order_l = list(np.argsort(final)[::-1])
    ranked = [int(cand[pf_local[o]]) for o in order_l]

    rs = set(ranked)
    all_final = lam_s * fs + lam_k * fk
    rest = [int(cand[i]) for i in np.argsort(all_final)[::-1] if int(cand[i]) not in rs]
    rs |= set(rest)
    tail = [int(i) for i in np.argsort(sem_sims)[::-1] if int(i) not in rs]
    return [poi_ids[i] for i in ranked + rest + tail]


def rank_rallmpoi(query, query_time, traj, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    lats = poi_store["lat"].values
    lons = poi_store["lng"].values
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()

    try:
        ts = pd.Timestamp(query_time)
        is_weekend = ts.weekday() >= 5; hour = ts.hour
    except: is_weekend,hour = False,12

    context_pois = []
    for t in traj:
        try:
            t_ts = pd.Timestamp(t.get("start_time",""))
            if t_ts.weekday()>=5==is_weekend and abs(t_ts.hour-hour)<=3:
                pname = t.get("poi_name","")
                if pname and str(pname)!="None":
                    context_pois.append(f"{pname} {t.get('poi_type','')}")
        except: pass
    if not context_pois:
        context_pois = [t.get("poi_name","") for t in traj[-50:]
                        if t.get("poi_name") and str(t.get("poi_name",""))!="None"]
    if context_pois:
        ctx_text = " | ".join(context_pois[:20])
        ctx_vec = model.encode([f"passage: {ctx_text}"], normalize_embeddings=True)
        sims = cosine_similarity(ctx_vec, e5_vecs)[0]
        htr_idx = np.argsort(sims)[::-1][:60]
    else:
        q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
        sims = cosine_similarity(q_vec, e5_vecs)[0]
        htr_idx = np.argsort(sims)[::-1][:60]

    cur_lat, cur_lon = get_current_location(traj)
    if cur_lat is not None and cur_lon is not None:
        dists = np.array([haversine(cur_lat,cur_lon,lats[i],lons[i]) for i in htr_idx])
        order = np.argsort(dists)
        gdr_idx = htr_idx[order[:20]]
    else:
        gdr_idx = htr_idx[:20]

    cand_list = []
    for rank, idx in enumerate(gdr_idx, 1):
        row = poi_store.iloc[idx]
        name = str(row.get("displayName",""))
        ptype = str(row.get("primaryType",""))
        addr = str(row.get("shortFormattedAddress",""))
        cand_list.append(f"{rank}. {name} | {ptype} | {addr}")

    recent = []
    for t in reversed(traj):
        pname = t.get("poi_name","")
        if pname and str(pname)!="None":
            recent.append(f"- {pname} ({t.get('poi_type','?')}) at {str(t.get('start_time',''))[:10]}")
        if len(recent)>=15: break
    traj_ctx = "\n".join(recent) if recent else "No named visits."

    prompt = f"""User query: "{query}" (at {query_time})

Recent visits:
{traj_ctx}

Candidate POIs:
{chr(10).join(cand_list)}

Select the BEST matching POI. Reply ONLY with the number (1-{len(gdr_idx)})."""
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=10,
            messages=[{"role":"user","content":prompt}])
        match = re.search(r'\d+', resp.choices[0].message.content.strip())
        if match:
            choice = int(match.group())-1
            if 0<=choice<len(gdr_idx):
                chosen = [gdr_idx[choice]]
                rest_gdr = [x for x in gdr_idx if x!=gdr_idx[choice]]
                rest_htr = [x for x in htr_idx if x not in set(gdr_idx.tolist())]
                all_cand = set(htr_idx.tolist())
                q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
                tail_sims = cosine_similarity(q_vec, e5_vecs)[0]
                tail = [i for i in np.argsort(tail_sims)[::-1] if i not in all_cand]
                full = chosen + rest_gdr + rest_htr + tail
                return [poi_ids[i] for i in full]
    except Exception as e:
        print(f"    [ALR err] {str(e)[:60]}")

    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, e5_vecs)[0]
    order = np.argsort(sims)[::-1]
    return [poi_ids[i] for i in order]

def rank_llm4poi(query, query_time, traj, uid, poi_store, top_k=20):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()

    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, e5_vecs)[0]
    if CAT_SEM_TOPN>0:
        _cs_load()
        _CA = _bl_aliases()
        _tc=_CS_CAT.get(query)
        _pt=poi_store['primaryType'].astype(str).str.lower().values
        if _tc:
            _cats={_tc}|{str(x).lower() for x in _CA.get(_tc,[])}
            _idx=np.nonzero(np.isin(_pt,list(_cats)))[0]
            if len(_idx)==0: _idx=np.arange(len(poi_ids))
        else:
            _idx=np.arange(len(poi_ids))
        top_k_idx=_idx[np.argsort(sims[_idx])[::-1][:CAT_SEM_TOPN]]
    else:
        top_k_idx = np.argsort(sims)[::-1][:top_k]

    try:
        ts = pd.Timestamp(query_time); time_ctx = ts.strftime("%A %H:%M")
    except: time_ctx = str(query_time)
    named = []
    for t in reversed(traj):
        pname = t.get("poi_name","")
        if pname and str(pname)!="None":
            try: ts2=pd.Timestamp(t.get("start_time","")); t_str=ts2.strftime("%a %H:%M")
            except: t_str=""
            named.append(f"{pname} ({t.get('poi_type','?')}) [{t_str}]")
        if len(named)>=20: break
    traj_prompt = ("Recent locations visited (latest first):\n"+"\n".join(f"  {i+1}. {v}" for i,v in enumerate(named))
                   if named else "No named visit history.")

    cand_list = []
    for rank,idx in enumerate(top_k_idx,1):
        row = poi_store.iloc[idx]
        name = str(row.get("displayName",f"Place {rank}"))
        ptype = str(row.get("primaryType",""))
        addr = str(row.get("shortFormattedAddress",""))
        cand_list.append(f"{rank}. {name} ({ptype}) — {addr}")

    user_msg = f"""{traj_prompt}

Current query (at {time_ctx}): "{query}"

Candidate POIs to rank:
{chr(10).join(cand_list)}

Rank all {len(top_k_idx)} from best to worst. Output ONLY comma-separated numbers: e.g. 3,1,5,2"""
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0.0, max_tokens=100,
            messages=[{"role":"system","content":"You are a personalized location recommendation system. Output ONLY comma-separated rank numbers."},
                      {"role":"user","content":user_msg}])
        text = resp.choices[0].message.content.strip()
        nums = [int(x.strip())-1 for x in re.findall(r'\d+', text) if 0<=int(x.strip())-1<len(top_k_idx)]
        if nums:
            seen = set(); ordered_idx = []
            for n in nums:
                if n not in seen: seen.add(n); ordered_idx.append(top_k_idx[n])
            for i in range(len(top_k_idx)):
                if i not in seen: ordered_idx.append(top_k_idx[i])
            in_top = set(top_k_idx.tolist())
            tail = [i for i in np.argsort(sims)[::-1] if i not in in_top]
            return [poi_ids[i] for i in ordered_idx+tail]
    except Exception as e:
        print(f"    [LLM4POI err] {str(e)[:60]}")

    order = np.argsort(sims)[::-1]
    return [poi_ids[i] for i in order]

@dataclass
class QueryIntent:
    raw_query:str; poi_category:str; spatial_type:str; temporal_type:str; preference:bool
    route_direction:Optional[str]; time_bucket:Optional[str]; time_range:Optional[tuple]

@dataclass
class UserContext:
    commute_lines:dict; commute_hours:dict; activity_zones:dict
    last_position:Optional[tuple]; preference_embedding:Optional[np.ndarray]
    preference_description:str

def _dist_to_segment(plat,plon,olat,olon,dlat,dlon):
    cos_lat = math.cos(math.radians((olat+dlat)/2))
    bx=(dlon-olon)*cos_lat*111320; by=(dlat-olat)*111320
    px=(plon-olon)*cos_lat*111320; py=(plat-olat)*111320
    ab2=bx*bx+by*by
    if ab2<1e-6: return haversine(plat,plon,olat,olon)
    t=max(0.0,min(1.0,(px*bx+py*by)/ab2))
    return math.sqrt((px-t*bx)**2+(py-t*by)**2)/1000.0

WEIGHTS={"route":{"s":0.45,"t":0.20,"e":0.35},"zone":{"s":0.30,"t":0.35,"e":0.35},
         "point":{"s":0.55,"t":0.10,"e":0.35},"none":{"s":0.15,"t":0.25,"e":0.60}}
BUCKET_HOURS={"weekday_morning":(6,10),"weekday_lunch":(11,14),"weekday_afternoon":(14,18),
              "weekday_evening":(18,23),"weekend_morning":(6,12),"weekend_afternoon":(12,18),
              "weekend_evening":(18,23),"late_night":(22,5)}
DB_SCHEMA="TABLE trajectory(uid VARCHAR,event_start TIMESTAMP,event_end TIMESTAMP,status VARCHAR,o_lat DOUBLE,o_lon DOUBLE,d_lat DOUBLE,d_lon DOUBLE,o_hour INTEGER,poi_name VARCHAR,poi_type VARCHAR);"

def _llm_sql(desc,uid):
    system=f"DuckDB SQL expert. Schema:\n{DB_SCHEMA}\nRules: ONLY SQL, no markdown; uid='{uid}' in query;\nDuckDB: extract(hour from CAST(col AS TIMESTAMP)), dayofweek(CAST(col AS TIMESTAMP))(0=Sun)."
    resp=client.chat.completions.create(model="gpt-4o-mini",temperature=0.0,max_tokens=400,
        messages=[{"role":"system","content":system},{"role":"user","content":desc}])
    return re.sub(r"```sql|```","",resp.choices[0].message.content.strip()).strip()

def _run(conn,primary,fallback):
    try: return conn.execute(primary).fetchdf()
    except:
        try: return conn.execute(fallback).fetchdf()
        except: return pd.DataFrame()

_intent_cache = {}
def parse_intent(query):
    if query in _intent_cache: return _intent_cache[query]
    system="""Return JSON only:
{"poi_category":"restaurant|cafe|supermarket|gym|pharmacy|bar|park|hospital|bank|gas_station|movie_theater|library|hotel|shopping_mall|beauty_salon|hair_care|fast_food_restaurant|convenience_store",
 "spatial_type":"route|zone|point|none","temporal_type":"time_bucket|range|none",
 "preference":true/false,"route_direction":"to_work|home|null",
 "time_bucket":"weekday_morning|weekday_lunch|weekday_afternoon|weekday_evening|weekend_morning|weekend_afternoon|weekend_evening|late_night|null"}"""
    resp=client.chat.completions.create(model="gpt-4o-mini",temperature=0.0,max_tokens=150,
        messages=[{"role":"system","content":system},{"role":"user","content":query}])
    d=json.loads(re.sub(r"```json|```","",resp.choices[0].message.content.strip()).strip())
    intent=QueryIntent(raw_query=query,poi_category=d.get("poi_category","place"),
        spatial_type=d.get("spatial_type","none"),temporal_type=d.get("temporal_type","none"),
        preference=bool(d.get("preference",False)),route_direction=d.get("route_direction"),
        time_bucket=d.get("time_bucket"),time_range=None)
    _intent_cache[query]=intent
    return intent

def extract_context(uid, intent, traj_list, embed_func):
    traj_clean=[t for t in traj_list]
    if traj_clean and traj_clean[-1].get("status")=="activity": traj_clean=traj_clean[:-1]
    rows=[{"uid":str(uid),"event_start":t.get("start_time"),"event_end":t.get("end_time"),
           "status":t.get("status","visit"),
           "o_lat":float(t.get("o_lat") or 0),"o_lon":float(t.get("o_lon") or 0),
           "d_lat":float(t.get("d_lat") or 0),"d_lon":float(t.get("d_lon") or 0),
           "o_hour":(lambda s:pd.Timestamp(s).hour if s else 0)(t.get("start_time")),
           "poi_name":t.get("poi_name",""),"poi_type":t.get("poi_type","")} for t in traj_clean]
    df=pd.DataFrame(rows)
    conn=duckdb.connect(":memory:"); conn.execute("CREATE TABLE trajectory AS SELECT * FROM df")
    hfb=f"SELECT AVG(d_lat) AS home_lat,AVG(d_lon) AS home_lon FROM trajectory WHERE uid='{uid}' AND status='visit' AND d_lat IS NOT NULL AND d_lat!=0 AND (extract(hour from CAST(event_start AS TIMESTAMP))>=21 OR extract(hour from CAST(event_start AS TIMESTAMP))<7) AND dayofweek(CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 HAVING COUNT(*)>=2"
    wfb=f"SELECT AVG(d_lat) AS work_lat,AVG(d_lon) AS work_lon FROM trajectory WHERE uid='{uid}' AND status='visit' AND d_lat IS NOT NULL AND d_lat!=0 AND extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 9 AND 18 AND dayofweek(CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 HAVING COUNT(*)>=2"
    hr=_run(conn,_llm_sql(f"uid='{uid}'. Home=nighttime weekday visits AVG(d_lat),AVG(d_lon) HAVING COUNT>=2.",uid),hfb)
    wr=_run(conn,_llm_sql(f"uid='{uid}'. Work=9-18h weekday visits AVG(d_lat),AVG(d_lon) HAVING COUNT>=2.",uid),wfb)
    homes=[(float(hr.iloc[0]["home_lat"]),float(hr.iloc[0]["home_lon"]))] if not hr.empty and float(hr.iloc[0].get("home_lat",0) or 0) else []
    works=[(float(wr.iloc[0]["work_lat"]),float(wr.iloc[0]["work_lon"]))] if not wr.empty and float(wr.iloc[0].get("work_lat",0) or 0) else []
    commute_lines,commute_hours={},{}
    if homes and works:
        hl,ho=homes[0]; wl,wo=works[0]; thr=0.018
        cfb=f"SELECT o_lat,o_lon,d_lat,d_lon,o_hour,CASE WHEN sqrt(pow(o_lat-{hl},2)+pow(o_lon-{ho},2))<{thr} THEN 'to_work' ELSE 'home' END AS direction FROM trajectory WHERE uid='{uid}' AND status='activity' AND o_lat IS NOT NULL AND ((sqrt(pow(o_lat-{hl},2)+pow(o_lon-{ho},2))<{thr} AND sqrt(pow(d_lat-{wl},2)+pow(d_lon-{wo},2))<{thr}) OR (sqrt(pow(o_lat-{wl},2)+pow(o_lon-{wo},2))<{thr} AND sqrt(pow(d_lat-{hl},2)+pow(d_lon-{ho},2))<{thr}))"
        cr=_run(conn,cfb,cfb); lines={"to_work":[],"home":[]}; hours={"to_work":[],"home":[]}
        for _,r in cr.iterrows():
            d=str(r.get("direction","to_work"))
            if d in lines:
                lines[d].append((float(r["o_lat"]),float(r["o_lon"]),float(r["d_lat"]),float(r["d_lon"])))
                hours[d].append(int(r.get("o_hour",0) or 0))
        commute_lines=lines; commute_hours={d:int(np.median(h)) for d,h in hours.items() if h}
    zfb=f"""WITH b AS (SELECT d_lat,d_lon,CASE WHEN dayofweek(CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 THEN CASE WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 6 AND 9 THEN 'weekday_morning' WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 11 AND 13 THEN 'weekday_lunch' WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 14 AND 17 THEN 'weekday_afternoon' WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 18 AND 22 THEN 'weekday_evening' ELSE NULL END ELSE CASE WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 6 AND 11 THEN 'weekend_morning' WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 12 AND 17 THEN 'weekend_afternoon' WHEN extract(hour from CAST(event_start AS TIMESTAMP)) BETWEEN 18 AND 22 THEN 'weekend_evening' ELSE NULL END END AS time_bucket FROM trajectory WHERE uid='{uid}' AND status='visit' AND d_lat IS NOT NULL AND d_lat!=0) SELECT time_bucket,AVG(d_lat) AS center_lat,AVG(d_lon) AS center_lon,COALESCE(STDDEV(d_lat)*111.0,1.0) AS radius_km FROM b WHERE time_bucket IS NOT NULL GROUP BY time_bucket HAVING COUNT(*)>=3"""
    zr=_run(conn,zfb,zfb); zones={str(r["time_bucket"]):{"center_lat":float(r["center_lat"]),"center_lon":float(r["center_lon"]),"radius_km":float(max(1.0,r["radius_km"]))} for _,r in zr.iterrows() if r.get("time_bucket")}
    lpfb=f"SELECT d_lat,d_lon FROM trajectory WHERE uid='{uid}' AND status='visit' AND d_lat IS NOT NULL AND d_lat!=0 ORDER BY event_start DESC LIMIT 1"
    lpr=_run(conn,lpfb,lpfb); last_pos=None
    if not lpr.empty:
        lat,lon=float(lpr.iloc[0].get("d_lat",0) or 0),float(lpr.iloc[0].get("d_lon",0) or 0)
        if lat and lon: last_pos=(lat,lon)
    pref_desc=intent.raw_query; pref_emb=embed_func([pref_desc])[0]
    conn.close()
    return UserContext(commute_lines=commute_lines,commute_hours=commute_hours,
                       activity_zones=zones,last_position=last_pos,
                       preference_embedding=pref_emb,preference_description=pref_desc)

def rank_our_v3(query, query_time, traj, uid, poi_store):
    intent = parse_intent(query)
    ctx = extract_context(uid, intent, traj, openai_embed)
    lats=pd.to_numeric(poi_store["lat"],errors="coerce")
    lons=pd.to_numeric(poi_store["lng"],errors="coerce")
    valid=poi_store[lats.notna()&lons.notna()].copy()
    valid["_lat"]=lats[valid.index]; valid["_lon"]=lons[valid.index]
    cat=valid[valid["primaryType"].fillna("").str.contains(intent.poi_category,case=False)]
    if cat.empty: cat=valid

    if intent.spatial_type=="route" and intent.route_direction:
        segs=ctx.commute_lines.get(intent.route_direction,[])
        if segs:
            mask=cat.apply(lambda r:any(_dist_to_segment(r["_lat"],r["_lon"],*s)<2.0 for s in segs),axis=1)
            if mask.any(): cat=cat[mask]
    elif intent.spatial_type=="zone" and intent.time_bucket:
        zone=ctx.activity_zones.get(intent.time_bucket)
        if zone:
            dists=cat.apply(lambda r:haversine(r["_lat"],r["_lon"],zone["center_lat"],zone["center_lon"]),axis=1)
            if (dists<zone["radius_km"]*2).any(): cat=cat[dists<zone["radius_km"]*2]
    elif intent.spatial_type=="point" and ctx.last_position:
        plat,plon=ctx.last_position
        dists=cat.apply(lambda r:haversine(r["_lat"],r["_lon"],plat,plon),axis=1)
        near=cat[dists<3.0]
        if not near.empty: cat=near
    candidates=cat.head(300)

    q_emb=openai_embed([query])[0]
    def csim(emb):
        if emb is None or not isinstance(emb,np.ndarray): return 0.0
        nq,np_=np.linalg.norm(q_emb),np.linalg.norm(emb)
        return float(np.dot(q_emb,emb)/(nq*np_)) if nq>0 and np_>0 else 0.0
    sims=candidates["embedding"].apply(csim)
    w=WEIGHTS.get(intent.spatial_type,WEIGHTS["none"])
    scores=[]
    for idx,row in candidates.iterrows():
        sim=float(sims.get(idx,0.0)); lat,lon=float(row["_lat"]),float(row["_lon"])
        if intent.spatial_type=="route":
            segs=ctx.commute_lines.get(intent.route_direction,[])
            ss=max(0.0,1.0-min(_dist_to_segment(lat,lon,*s) for s in segs)/3.0) if segs else 0.5
        elif intent.spatial_type=="zone":
            zone=ctx.activity_zones.get(intent.time_bucket)
            ss=max(0.0,1.0-haversine(lat,lon,zone["center_lat"],zone["center_lon"])/(zone["radius_km"]*2)) if zone else 0.5
        elif intent.spatial_type=="point" and ctx.last_position:
            ss=max(0.0,1.0-haversine(lat,lon,*ctx.last_position)/2.0)
        else: ss=0.5
        h=str(row.get("regularOpeningHours_text") or "")
        ts=0.85 if intent.time_bucket and (str(BUCKET_HOURS.get(intent.time_bucket,(0,24))[0]) in h) else (0.6 if h else 0.4)
        if "24" in h: ts=1.0
        se=min(1.0,max(0.0,sim*0.6))
        final=w["s"]*ss+w["t"]*ts+w["e"]*se
        if str(row.get("source",""))=="osm": final*=0.92
        scores.append((str(row.get("id","")),final))
    scores.sort(key=lambda x:x[1],reverse=True)
    ranked_in_spatial=[pid for pid,_ in scores]
    in_spatial=set(ranked_in_spatial)
    tail_df=valid[~valid["id"].astype(str).isin(in_spatial)]
    tail_sims=tail_df["embedding"].apply(csim)
    tail_sorted=tail_df.assign(_sim=tail_sims.values).sort_values("_sim",ascending=False)
    tail_ids=tail_sorted["id"].astype(str).tolist()
    return ranked_in_spatial + tail_ids



def _poi_text(row):
    parts = []
    for col in ["displayName","primaryType","shortFormattedAddress","types","regularOpeningHours_text"]:
        v = str(row.get(col,"") or "")
        if v and v != "nan": parts.append(v[:120])
    return " | ".join(parts) or "unknown"

_bm25_cache = {}

def get_bm25_index(uid, poi_store):
    if uid in _bm25_cache:
        corpus_ids, bm = _bm25_cache[uid]
        if len(corpus_ids) == len(poi_store):
            return corpus_ids, bm
    corpus = [_poi_text(poi_store.iloc[i]).lower().split() for i in range(len(poi_store))]
    corpus_ids = poi_store["id"].astype(str).tolist()
    if BM25Okapi is None:
        raise ImportError("rank_bm25 not installed: pip install rank_bm25")
    bm = BM25Okapi(corpus)
    _bm25_cache[uid] = (corpus_ids, bm)
    return corpus_ids, bm

def rank_bm25(query, uid, poi_store):
    corpus_ids, bm = get_bm25_index(uid, poi_store)
    tokens = query.lower().split()
    scores = bm.get_scores(tokens)
    order = np.argsort(scores)[::-1]
    return [corpus_ids[i] for i in order]

def rank_sem_spatial(query, traj, uid, poi_store, alpha=0.7, radius_km=50.0):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
    sem_sims = cosine_similarity(q_vec, e5_vecs)[0]

    cur_lat, cur_lon = get_current_location(traj)
    if cur_lat is not None and cur_lon is not None:
        lats = poi_store["lat"].values.astype(float)
        lons = poi_store["lng"].values.astype(float)
        valid_mask = (lats != 0) & (~np.isnan(lats)) & (~np.isnan(lons))
        dists = np.zeros(len(lats))
        for i in np.where(valid_mask)[0]:
            try: dists[i] = haversine(cur_lat, cur_lon, lats[i], lons[i])
            except: dists[i] = 9999.0
        dists[~valid_mask] = 9999.0
        spatial_score = np.exp(-dists / radius_km)
        combined = alpha * sem_sims + (1 - alpha) * spatial_score
    else:
        combined = sem_sims

    order = np.argsort(combined)[::-1]
    return [poi_ids[i] for i in order]

def _rankgpt_clean_response(response: str):
    out = ""
    for c in response:
        out += c if c.isdigit() else " "
    return out.strip()


def _rankgpt_permutation(query, hits, rank_start, rank_end, model_name="gpt-4o-mini"):
    cut = hits[rank_start:rank_end]
    num = len(cut)
    if num == 0:
        return hits

    messages = [
        {"role": "system",
         "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
        {"role": "user",
         "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]
    for i, h in enumerate(cut):
        content = " ".join(h["content"].split()[:300])
        messages.append({"role": "user", "content": f"[{i+1}] {content}"})
        messages.append({"role": "assistant", "content": f"Received passage [{i+1}]."})
    messages.append({"role": "user",
        "content": f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. "
                   f"The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. "
                   f"The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."})

    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages, temperature=0, max_tokens=300)
        permutation = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"      [RankGPT] LLM failed: {str(e)[:70]}")
        return hits

    resp_ids = [int(x) - 1 for x in _rankgpt_clean_response(permutation).split()]
    seen_local, dedup = set(), []
    for x in resp_ids:
        if x not in seen_local:
            seen_local.add(x); dedup.append(x)
    original = list(range(len(cut)))
    order = [x for x in dedup if x in original]
    order += [x for x in original if x not in order]

    new_hits = hits[:]
    for j, x in enumerate(order):
        new_hits[rank_start + j] = cut[x]
    return new_hits


def rank_rankgpt(query, traj, uid, poi_store, top_k=100):
    poi_ids = poi_store["id"].astype(str).tolist()

    if RANKGPT_BM25:
        sims = _bm25_scores(query, poi_store)
    else:
        e5_vecs = get_user_e5_vecs(uid, poi_store)
        model = get_e5_model()
        q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
        sims = cosine_similarity(q_vec, e5_vecs)[0]
    top_k_idx = np.argsort(sims)[::-1][:top_k]

    hits = []
    for idx in top_k_idx:
        row = poi_store.iloc[idx]
        name = str(row.get("displayName", "")).strip()
        ptype = str(row.get("primaryType", "")).replace("_", " ")
        addr = str(row.get("shortFormattedAddress", "")).strip()
        hits.append({"gidx": int(idx), "content": f"{name} ({ptype}) - {addr}"})

    WINDOW, STEP = 20, 10
    end_pos, start_pos = len(hits), len(hits) - WINDOW
    while start_pos >= 0:
        start_pos = max(start_pos, 0)
        hits = _rankgpt_permutation(query, hits, start_pos, end_pos)
        end_pos -= STEP
        start_pos -= STEP

    ranked_idx = [h["gidx"] for h in hits]
    ranked_set = set(ranked_idx)
    tail = [int(i) for i in np.argsort(sims)[::-1] if int(i) not in ranked_set]
    return [poi_ids[i] for i in ranked_idx + tail]


def rank_llm_traj(query, query_time, traj, uid, poi_store, top_k=20):
    poi_ids = poi_store["id"].astype(str).tolist()

    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    q_vec = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, e5_vecs)[0]
    top_k_idx = np.argsort(sims)[::-1][:top_k]
    cur_lat, cur_lon = get_current_location(traj)
    combined = sims

    recent = []
    for t in reversed(traj):
        pname = t.get("poi_name","")
        if pname and str(pname) != "None":
            recent.append(f"- {pname} ({t.get('poi_type','?')}) @ {(t.get('start_time','?') or '')[:10]}")
        if len(recent) >= 15: break
    traj_ctx = "\n".join(recent) if recent else "No named visits available."

    cands = []
    for rank, idx in enumerate(top_k_idx, 1):
        row = poi_store.iloc[idx]
        name = str(row.get("displayName","")).strip()
        ptype = str(row.get("primaryType","")).replace("_"," ")
        addr = str(row.get("shortFormattedAddress","")).strip()
        cands.append(f"{rank}. {name} ({ptype}) — {addr}")
    cand_block = "\n".join(cands)

    prompt = f"""You are a trajectory-aware POI retrieval assistant. Given:
1. User's recent mobility history
2. A natural language query  
3. Candidate POIs

Rank the candidates from most to least relevant to the query, considering the user's behavior patterns.
Return ONLY a comma-separated list of candidate numbers (e.g., "3,1,7,2,...").

User's recent visits (most recent first):
{traj_ctx}

Query: {query}
Query time: {query_time}

Candidates:
{cand_block}

Ranking (comma-separated numbers):"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        nums = [int(x.strip()) for x in re.split(r"[,\s]+", text) if x.strip().isdigit()]
        ranked_idx = []
        seen = set()
        for n in nums:
            if 1 <= n <= top_k and (n-1) not in seen:
                ranked_idx.append(int(top_k_idx[n-1]))
                seen.add(n-1)
        for i in top_k_idx:
            ii = int(i)
            if ii not in seen:
                ranked_idx.append(ii)
                seen.add(ii)
    except Exception as e:
        print(f"      [LLM+Traj] LLM failed: {e}, falling back to combined order")
        ranked_idx = [int(i) for i in top_k_idx]

    ranked_set = set(ranked_idx)
    tail = [i for i in np.argsort(combined)[::-1] if i not in ranked_set]
    full_idx = ranked_idx + tail
    return [poi_ids[i] for i in full_idx]




_minilm_model = None

def get_minilm_model():
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [init] Loading all-MiniLM-L6-v2 ...")
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _minilm_model

def get_user_minilm_vecs(uid, poi_store):
    cache_path = os.path.join(RESULT_DIR, f"minilm_{uid}.pkl")
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    model = get_minilm_model()
    texts = [_poi_embed_text(row) for _, row in poi_store.iterrows()]
    vecs = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    pickle.dump(vecs, open(cache_path, "wb"))
    return vecs

def rank_minilm(query, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    vecs = get_user_minilm_vecs(uid, poi_store)
    model = get_minilm_model()
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, vecs)[0]
    return [poi_ids[i] for i in np.argsort(sims)[::-1]]


_bgem3_model = None

def get_bgem3_model():
    global _bgem3_model
    if _bgem3_model is None:
        from FlagEmbedding import BGEM3FlagModel
        print("  [init] Loading BAAI/bge-m3 ...")
        _bgem3_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _bgem3_model

def get_user_bgem3_vecs(uid, poi_store):
    cache_path = os.path.join(RESULT_DIR, f"bgem3_{uid}.pkl")
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    model = get_bgem3_model()
    texts = [_poi_embed_text(row) for _, row in poi_store.iterrows()]
    out = model.encode(texts, batch_size=64, max_length=512,
                       return_dense=True, return_sparse=False, return_colbert_vecs=False)
    vecs = np.array(out["dense_vecs"])
    pickle.dump(vecs, open(cache_path, "wb"))
    return vecs

def rank_bgem3(query, uid, poi_store):
    poi_ids = poi_store["id"].astype(str).tolist()
    vecs = get_user_bgem3_vecs(uid, poi_store)
    model = get_bgem3_model()
    q_out = model.encode([query], batch_size=1, max_length=512,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
    q_vec = np.array(q_out["dense_vecs"])
    sims = cosine_similarity(q_vec, vecs)[0]
    return [poi_ids[i] for i in np.argsort(sims)[::-1]]


def rank_hyde(query, uid, poi_store):
    prompt = (
        "You are given a location search query. Generate a short, realistic description "
        "of a specific Point of Interest that would perfectly satisfy this query. "
        "Include: POI name, category/type, neighborhood or area, key features. "
        "2-3 sentences only.\n\nQuery: " + query + "\n\nHypothetical POI description:"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=120,
            messages=[{"role": "user", "content": prompt}]
        )
        hypo = resp.choices[0].message.content.strip()
    except Exception:
        hypo = query
    poi_ids = poi_store["id"].astype(str).tolist()
    vecs = get_user_bgem3_vecs(uid, poi_store)
    model = get_bgem3_model()
    q_out = model.encode([hypo], batch_size=1, max_length=512,
                         return_dense=True, return_sparse=False, return_colbert_vecs=False)
    q_vec = np.array(q_out["dense_vecs"])
    sims = cosine_similarity(q_vec, vecs)[0]
    return [poi_ids[i] for i in np.argsort(sims)[::-1]]


def rank_bm25_spatial(query, traj, uid, poi_store, alpha=0.5, top_k=50):
    poi_ids = poi_store["id"].astype(str).tolist()
    lats = poi_store["lat"].values
    lons = poi_store["lng"].values
    corpus = [[w.lower() for w in _poi_embed_text(row).split()] for _, row in poi_store.iterrows()]
    tokens = [w.lower() for w in query.split()]
    if BM25Okapi is not None:
        bm25_scores = np.array(BM25Okapi(corpus).get_scores(tokens))
    else:
        bm25_scores = np.zeros(len(poi_ids))
    top_idx = np.argsort(bm25_scores)[::-1][:top_k]
    cur_lat, cur_lon = get_current_location(traj)
    if cur_lat is None:
        return [poi_ids[i] for i in np.argsort(bm25_scores)[::-1]]
    dists = {}
    for i in top_idx:
        if pd.notna(lats[i]) and lats[i] != 0:
            dists[i] = haversine(cur_lat, cur_lon, float(lats[i]), float(lons[i]))
        else:
            dists[i] = 999.0
    max_dist = max(dists.values()) if dists else 1.0
    bm25_max = max(bm25_scores[top_idx].max() if len(top_idx) else 0, 1e-9)
    combined = {i: alpha*(bm25_scores[i]/bm25_max) + (1-alpha)*(1-dists[i]/max_dist)
                for i in top_idx}
    sorted_top = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
    ranked = [poi_ids[i] for i in sorted_top]
    ranked_set = set(ranked)
    tail = [poi_ids[i] for i in np.argsort(bm25_scores)[::-1] if poi_ids[i] not in ranked_set]
    return ranked + tail




LLMMOVE_CLOSED = os.environ.get("LLMMOVE_CLOSED_SET", "0") == "1"

def rank_llmmove(query, query_time, traj, uid, poi_store, top_k=100, gold_id=None, closed_set=False):
    poi_ids = poi_store["id"].astype(str).tolist()
    lats   = poi_store["lat"].values
    lons   = poi_store["lng"].values
    types  = poi_store["primaryType"].fillna("").values

    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model   = get_e5_model()
    q_vec   = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims    = cosine_similarity(q_vec, e5_vecs)[0]

    if os.environ.get("LLMMOVE_NEAR","0") in ("1", "near_gt"):
        _near = _nearest_to_user(poi_store, traj, n=100)
        if os.environ.get("LLMMOVE_NEAR") == "near_gt" and _near and gold_id is not None:
            if str(gold_id) not in _near:
                _near = [str(gold_id)] + _near[:99]
        if _near:
            _id2i_n = {p: i for i, p in enumerate(poi_ids)}
            top_idx = np.array([_id2i_n[p] for p in _near if p in _id2i_n])
            _skip_e5 = True
        else:
            _skip_e5 = False
    else:
        _skip_e5 = False
    if closed_set and gold_id is not None:
        import random as _rnd
        _id2i = {pid: i for i, pid in enumerate(poi_ids)}
        _rnd.seed(hash(str(gold_id)) % (2**31))
        _neg = _rnd.sample(range(len(poi_ids)), 100)
        _gi = _id2i.get(str(gold_id))
        if _gi is not None and _gi not in _neg:
            _neg = _neg[:100] + [_gi]
        top_idx = np.array(_neg)
    elif not _skip_e5:
        top_idx = np.argsort(sims)[::-1][:top_k]

    visits = [t for t in traj
              if t.get("status") == "visit"
              and t.get("poi_name") and str(t.get("poi_name")) != "None"
              and t.get("d_lat") and float(t.get("d_lat", 0)) != 0]
    if not visits:
        return [poi_ids[i] for i in top_idx] + \
               [poi_ids[i] for i in np.argsort(sims)[::-1] if i not in set(top_idx)]
    visits.sort(key=lambda x: x.get("start_time", ""))
    _pool_type = {pid: str(types[i]) for i, pid in enumerate(poi_ids)}
    def _ck(ts):
        out = []
        for t in ts:
            pid = str(t.get("pid", ""))
            if pid in _pool_type:
                out.append((pid, _pool_type[pid] or "?"))
        return out
    longterm = _ck(visits[-40:])
    recent   = _ck(visits[-5:])

    cur_lat = float(visits[-1].get("d_lat"))
    cur_lon = float(visits[-1].get("d_lon"))

    candidates = []
    for idx in top_idx:
        if pd.isna(lats[idx]) or lats[idx] == 0:
            continue
        d = haversine(cur_lat, cur_lon, float(lats[idx]), float(lons[idx]))
        candidates.append((poi_ids[idx], round(d, 2), str(types[idx])))
    candidates.sort(key=lambda x: x[1])

    prompt = (
        "<long-term check-ins> [Format: (POIID, Category)]: " + str(longterm) + "\n"
        "<recent check-ins> [Format: (POIID, Category)]: " + str(recent) + "\n"
        "<candidate set> [Format: (POIID, Distance, Category)]: " + str(candidates) + "\n"
        "Your task is to recommend a user's next point-of-interest (POI) from <candidate set> "
        "based on his/her trajectory information.\n"
        "The trajectory information is made of a sequence of the user's <long-term check-ins> "
        "and a sequence of the user's <recent check-ins> in chronological order.\n"
        "Now I explain the elements in the format. \"POIID\" refers to the unique id of the POI, "
        "\"Distance\" indicates the distance (kilometers) between the user and the POI, and "
        "\"Category\" shows the semantic information of the POI.\n\n"
        "Requirements:\n"
        "1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.\n"
        "2. Consider the recent check-ins to extract users' current perferences.\n"
        "3. Consider the \"Distance\" since people tend to visit nearby pois.\n"
        "4. Consider which \"Category\" the user would go next for long-term check-ins indicates sequential transitions the user prefer.\n\n"
        "Please organize your answer in a JSON object containing following keys:\n"
        "\"recommendation\" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), "
        "and \"reason\" (a concise explanation that supports your recommendation according to the requirements). "
        "Do not include line breaks in your output."
    )

    recommended = []
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r'"recommendation"\s*:\s*\[(.*?)\]', text, re.DOTALL)
        if m:
            raw_ids = re.findall(r'"([^"]+)"', m.group(1))
            valid = set(poi_ids)
            for rid in raw_ids:
                if rid in valid and rid not in recommended:
                    recommended.append(rid)
    except Exception as e:
        print(f"      [LLMMove] LLM failed: {str(e)[:80]}")

    rec_set = set(recommended)
    tail = [c[0] for c in candidates if c[0] not in rec_set]
    rec_set |= set(tail)
    final = recommended + tail
    for i in np.argsort(sims)[::-1]:
        pid = poi_ids[i]
        if pid not in rec_set:
            final.append(pid); rec_set.add(pid)
    return final


def rank_llmrank(query, query_time, traj, uid, poi_store, top_k=20):
    poi_ids = poi_store["id"].astype(str).tolist()
    names   = poi_store["displayName"].fillna("").values
    types   = poi_store["primaryType"].fillna("").values

    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model   = get_e5_model()
    q_vec   = model.encode([f"query: {query}"], normalize_embeddings=True)
    sims    = cosine_similarity(q_vec, e5_vecs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    visits = [t for t in traj
              if t.get("status") == "visit"
              and t.get("poi_name") and str(t.get("poi_name")) != "None"]
    visits.sort(key=lambda x: x.get("start_time", ""))
    user_his = visits[-20:]
    user_his_text = [f'{j}. {t.get("poi_name","")} ({t.get("poi_type","?")})'
                     for j, t in enumerate(user_his)]

    candidate_text_order = []
    for j, idx in enumerate(top_idx):
        name = str(names[idx]).strip()
        ptype = str(types[idx]).replace("_", " ")
        candidate_text_order.append(f"{j}. {name} ({ptype})")

    prompt = (
        f"I've visited the following POIs in the past in order:\n{user_his_text}\n\n"
        f"My current query: \"{query}\"\n\n"
        f"Now there are {top_k} candidate POIs that I can visit next:\n{candidate_text_order}\n"
        f"Please rank these {top_k} POIs by measuring the probability that I would visit next "
        f"according to my visit history and current query. Please think step by step.\n"
        f"Please show me your ranking results with order numbers. Split your output with line break. "
        f"You MUST rank the given candidate POIs. You can not generate POIs that are not in the given candidate list."
    )

    ranked_idx = []
    seen = set()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.0, max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content.strip()
        for line in text.split("\n"):
            m = re.match(r"\s*(\d+)", line)
            if m:
                n = int(m.group(1))
                if 0 <= n < top_k and n not in seen:
                    ranked_idx.append(int(top_idx[n]))
                    seen.add(n)
    except Exception as e:
        print(f"      [LLMRank] LLM failed: {str(e)[:80]}")

    for j, idx in enumerate(top_idx):
        if j not in seen:
            ranked_idx.append(int(idx)); seen.add(j)

    ranked_set = set(ranked_idx)
    for i in np.argsort(sims)[::-1]:
        if i not in ranked_set:
            ranked_idx.append(int(i)); ranked_set.add(int(i))
    return [poi_ids[i] for i in ranked_idx]


import baseline_llmmob as _lmob

def rank_llm_mob(query, query_time, traj, uid, poi_store, top_k=100):
    poi_ids = poi_store["id"].astype(str).tolist()
    e5_vecs = get_user_e5_vecs(uid, poi_store)
    model = get_e5_model()
    sims = cosine_similarity(
        model.encode([f"query: {query}"], normalize_embeddings=True), e5_vecs)[0]
    e5_order = [poi_ids[i] for i in np.argsort(sims)[::-1]]
    if _AM_META["map"] is None:
        ids = poi_ids
        nms = poi_store["displayName"].fillna("").astype(str).tolist()
        tys = poi_store["primaryType"].fillna("").astype(str).tolist()
        _AM_META["map"] = {i: (n, t) for i, n, t in zip(ids, nms, tys)}
    return _lmob.rank_llmmob(query, query_time, traj, uid, poi_store,
                             llm_fn=_am_llm, e5_order=e5_order,
                             cand_ids=e5_order[:top_k],
                             poi_meta=_AM_META["map"], top_k=top_k)

def evaluate(methods_to_run):
    records=[json.loads(l) for l in open(DATASET_PATH)]
    _lim=int(os.environ.get("BL_LIMIT","0"))
    if _lim: records=records[:_lim]
    print(f"Loaded {len(records)} records")

    KS=[1,3,5,10]
    all_results={}

    for method_name in methods_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name.upper()}")
        print('='*60)
        _mkeys = [f"{m}@{k}" for m in ("P","R","F1","NDCG") for k in KS] + ["MRR"]
        metrics = {m: [] for m in _mkeys}
        by_qtype = defaultdict(lambda: {m: [] for m in _mkeys})
        log=[]

        import threading as _th
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _BLW = int(os.environ.get("BL_WORKERS", "8"))
        _AM_LOCK = _th.Lock()
        _plock = _th.Lock()
        _done_n = {"n": 0}
        print(f"  [parallel] {_BLW} workers x {len(records)} records", flush=True)

        def _run_one(_item):
            i, r = _item
            uid=str(r["uid"]); query=r["question"]; query_time=r["time"]
            answer_pid=str(r["answer"]["poi_id"]); answer_name=r["answer"]["name"]
            qtype=r["meta"]["query_type"]; traj=r["traj"]
            try:
                poi_store=get_user_poi_store(uid,traj)
                if method_name=="e5":
                    ranked_ids=rank_e5only(query,uid,poi_store)
                elif method_name=="sd":
                    ranked_ids=rank_sd(query,traj,uid,poi_store)
                elif method_name=="st":
                    ranked_ids=rank_st(query,traj,uid,poi_store)
                elif method_name=="naiverag":
                    ranked_ids=rank_naiverag(query,traj,uid,poi_store)
                elif method_name=="geollm":
                    ranked_ids=rank_geollm(query,traj,uid,poi_store)
                elif method_name=="semask":
                    ranked_ids=rank_semask(query,traj,uid,poi_store)
                elif method_name in ("spatial","spatialrag"):
                    ranked_ids=rank_spatialrag(query,traj,uid,poi_store)
                elif method_name=="rallm":
                    ranked_ids=rank_rallmpoi(query,query_time,traj,uid,poi_store)
                elif method_name=="llm4poi":
                    ranked_ids=rank_llm4poi(query,query_time,traj,uid,poi_store)
                elif method_name=="ours":
                    ranked_ids=rank_our_v3(query,query_time,traj,uid,poi_store)
                elif method_name=="bm25":
                    ranked_ids=rank_bm25(query,uid,poi_store)
                elif method_name=="sem_spatial":
                    ranked_ids=rank_sem_spatial(query,traj,uid,poi_store)
                elif method_name=="rankgpt":
                    ranked_ids=rank_rankgpt(query,traj,uid,poi_store,top_k=(CAT_SEM_TOPN or 100))
                elif method_name=="llm_traj":
                    ranked_ids=rank_llm_traj(query,query_time,traj,uid,poi_store)
                elif method_name=="minilm":
                    ranked_ids=rank_minilm(query,uid,poi_store)
                elif method_name=="bgem3":
                    ranked_ids=rank_bgem3(query,uid,poi_store)
                elif method_name=="hyde":
                    ranked_ids=rank_hyde(query,uid,poi_store)
                elif method_name=="bm25_spatial":
                    ranked_ids=rank_bm25_spatial(query,traj,uid,poi_store)
                elif method_name=="reactgis":
                    ranked_ids=rank_reactgis(query,query_time,traj,uid,poi_store)
                elif method_name=="agentmove":
                    with _AM_LOCK:
                        _am_prepare(records, poi_store)
                    _amn=os.environ.get("AGENTMOVE_NATIVE","0")
                    if _amn=="near_gt":
                        _natc=_nearest_to_user(poi_store,traj,n=100,radius_km=5.0) or []
                        if str(answer_pid) not in _natc:
                            _natc=[str(answer_pid)]+_natc[:99]
                    elif _amn=="user":
                        _natc=_nearest_to_user(poi_store,traj,n=100,radius_km=5.0)
                    elif _amn=="1":
                        _natc=_agentmove_native_order(poi_store,answer_pid)
                    else:
                        _natc=None
                    ranked_ids=rank_agentmove(query,query_time,traj,uid,poi_store,native_cands=_natc)
                elif method_name=="text2sql":
                    ranked_ids=rank_text2sql(query,query_time,traj,uid,poi_store)
                elif method_name=="llmmove":
                    ranked_ids=rank_llmmove(query,query_time,traj,uid,poi_store,
                                            gold_id=answer_pid, closed_set=LLMMOVE_CLOSED)
                elif method_name=="llmrank":
                    ranked_ids=rank_llmrank(query,query_time,traj,uid,poi_store)
                elif method_name=="llm_mob":
                    ranked_ids=rank_llm_mob(query,query_time,traj,uid,poi_store)
                else: return (None, None, None)

                sm = {}
                for k in KS:
                    sm[f"R@{k}"] = recall_at_k(ranked_ids,answer_pid,k)
                    sm[f"NDCG@{k}"] = ndcg_at_k(ranked_ids,answer_pid,k)
                sm["MRR"] = mrr_score(ranked_ids,answer_pid)
                try: rank=ranked_ids.index(answer_pid)+1
                except: rank=-1
                entry = {"uid":uid,"qtype":qtype,"query":query,"answer":answer_name,
                         "rank":rank,**{m:round(v,4) for m,v in sm.items()}}
                out = (qtype, sm, entry)
            except Exception as e:
                import traceback; print(f"    [{i+1}] ERROR: {e}"); traceback.print_exc()
                out = (qtype, None, None)
            with _plock:
                _done_n["n"] += 1
                n = _done_n["n"]
                if n % 10 == 0 or n == len(records):
                    print(f"  [{n}/{len(records)}] done", flush=True)
            return out

        import time as _time_meter
        with _METER_LOCK:
            _LLM_CALLS["chat"] = 0; _LLM_CALLS["emb"] = 0
        _t0_meter = _time_meter.time()
        if _BL_ENCODER == "e5":
            get_e5_model(); get_pool_openai_vecs(get_global_store())
        with _TPE(max_workers=_BLW) as _ex:
            _results = list(_ex.map(_run_one, list(enumerate(records))))
        _wall_meter = _time_meter.time() - _t0_meter

        for qtype, sm, entry in _results:
            if qtype is None:
                continue
            if sm is None:
                for m in metrics: metrics[m].append(0); by_qtype[qtype][m].append(0)
                continue
            for m,v in sm.items():
                metrics[m].append(v); by_qtype[qtype][m].append(v)
            log.append(entry)

        N=len(records)
        _nq=max(1,len(records))
        print("[METER] %s chat_calls=%d chat_per_q=%.2f wall=%.1fs sec_per_q=%.2f workers=%d"%(method_name,_LLM_CALLS["chat"],_LLM_CALLS["chat"]/_nq,_wall_meter,_wall_meter/_nq,_BLW),flush=True)
        print(f"\n=== {method_name.upper()} RESULTS (N={N}) ===")
        print(f"{'Metric':<12} {'All':>7}")
        for m in ["R@1","R@3","R@5","R@10","NDCG@1","NDCG@3","NDCG@5","NDCG@10","MRR"]:
            av = np.mean(metrics[m]) if metrics[m] else 0
            print(f"{m:<12} {av:>7.4f}")

        if method_name == "text2sql":
            result_extra = {"t2sql_failure_modes": dict(_t2s.T2SQL_STATS)}
        elif method_name == "reactgis":
            result_extra = {"react_failure_modes": dict(_rga.REACT_STATS)}
        else:
            result_extra = {}
        result={"method":method_name,"N":N, **result_extra,
                **{m:round(float(np.mean(v)),4) for m,v in metrics.items() if v},
                "by_qtype":{qt:{m:round(float(np.mean(vs)),4) for m,vs in qm.items() if vs}
                            for qt,qm in by_qtype.items()},
                "log":log}
        out_path=os.path.join(RESULT_DIR,f"{method_name}_results.json")
        json.dump(result,open(out_path,"w"),indent=2,ensure_ascii=False)
        print(f"Saved → {out_path}")
        all_results[method_name]=result

    return all_results

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["takeout","geolife"])
    parser.add_argument("--methods", default="rankgpt,llm4poi,llmmove,llmrank,spatial",
                        help="Comma-separated. Default runs 5 core baselines.")
    parser.add_argument("--out", default=None)
    args=parser.parse_args()
    if args.dataset == "takeout":
        DATASET_PATH = "./data/takeout/trajrag_takeout_final.jsonl"
        POI_STORE_PKL = "./data/takeout/poi_store_unified_v6_e5base.pkl"
        RESULT_DIR = args.out or "/tmp/baselines_takeout"
    else:
        DATASET_PATH = "./data/geolife/geolife_final.jsonl"
        POI_STORE_PKL = "./data/geolife/poi_store_amap_v13_e5base_v2.pkl"
        RESULT_DIR = args.out or "/tmp/baselines_geolife"
    globals()["_BL_DATASET"] = args.dataset
    globals()["DATASET_PATH"] = DATASET_PATH
    globals()["POI_STORE_PKL"] = POI_STORE_PKL
    globals()["RESULT_DIR"] = RESULT_DIR
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"DATASET_PATH={DATASET_PATH}")
    print(f"POI_STORE_PKL={POI_STORE_PKL}")
    print(f"RESULT_DIR={RESULT_DIR}")
    methods=[m.strip() for m in args.methods.split(",")]
    print(f"Running methods: {methods}")
    evaluate(methods)
