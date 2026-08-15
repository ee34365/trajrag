import os, sys, json, time
from collections import defaultdict

_early_st = None
try:
    from sentence_transformers import SentenceTransformer as _EarlyST
    _early_st = _EarlyST("intfloat/multilingual-e5-base")
    print("[AH2] pre-loaded e5-base", flush=True)
except Exception as _e:
    print(f"[AH2] pre-load failed: {_e}", flush=True)

import numpy as np, pandas as pd
import trajrag_env
trajrag_env.setup(os.environ.get("TRAJRAG_DATASET", "takeout"))

import pipeline as _pl
from pipeline import score_query, build_conn_from_traj
if _early_st is not None:
    _pl._LOCAL_EMBED = _early_st
    try:
        import evidence as _ev
        if hasattr(_ev, "_LOCAL_EMBED_MODEL"):
            _ev._LOCAL_EMBED_MODEL = _early_st
    except Exception:
        pass

from scoring import _category_match_bonus

import threading as _mth
from collections import Counter as _Ctr

LLM_METER = {"calls": _Ctr(), "prompt": 0, "completion": 0}
_meter_lock = _mth.Lock()


def _install_meter():
    def wrap(mod, name):
        cl = getattr(mod, "_client", None)
        if cl is None:
            return
        orig = cl.chat.completions.create
        if getattr(orig, "_metered", False):
            return

        def w(*a, **k):
            r = orig(*a, **k)
            with _meter_lock:
                LLM_METER["calls"][(name, k.get("model", "?"))] += 1
                u = getattr(r, "usage", None)
                if u is not None:
                    LLM_METER["prompt"] += getattr(u, "prompt_tokens", 0) or 0
                    LLM_METER["completion"] += getattr(u, "completion_tokens", 0) or 0
            return r

        w._metered = True
        cl.chat.completions.create = w

    import parser as _pr
    import evidence as _evd
    wrap(_pr, "parser")
    wrap(_evd, "evidence")


_install_meter()

from constants import CATEGORY_ALIASES as _CA

import threading
_TL = threading.local()

def _last():
    d = getattr(_TL, "d", None)
    if d is None:
        d = {}; _TL.d = d
    return d

_orig_sc = _pl.score_candidates
def _sc_wrapped(query, intent, R_tau, A_q, q_emb, poi_emb_cache, visit_map=None):
    ranked = _orig_sc(query, intent, R_tau, A_q, q_emb, poi_emb_cache, visit_map)
    tcat = (intent.get("target") or {}).get("category")
    for p in ranked:
        p["_cat_match"] = _category_match_bonus(p.get("category"), tcat, _CA)
    d = _last()
    _sp = intent.get("spatial") or {}
    _tm = intent.get("temporal") or {}
    d["mode"] = _sp.get("mode")
    d["anchor"] = _sp.get("anchor")
    d["route_dir"] = _sp.get("route_direction")
    d["zone_name"] = _sp.get("zone_name")
    d["time_bucket"] = _tm.get("time_bucket")
    d["day_hint"] = _tm.get("day_hint")
    d["target_cat"] = tcat
    d["n_Aq"] = len(A_q)
    _pt = (R_tau or {}).get("point") or {}
    d["pt_source"] = _pt.get("source")
    d["pt_name"] = _pt.get("anchor_name")
    d["pt_lat"] = _pt.get("lat")
    d["pt_lon"] = _pt.get("lon")
    d["pt_poi"] = _pt.get("poi_id")
    return ranked
_pl.score_candidates = _sc_wrapped

POOL_PATH = os.environ.get("POOL_PATH", "./data/takeout/poi_store_unified_v6_e5base.pkl")
DATA_PATH = os.environ.get("EVAL_DATA", "./data/takeout/trajrag_takeout_final.jsonl")
OUT = os.environ.get("DUMP_OUT", "/tmp/signals_takeout.jsonl")
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))

print(f"[1/3] Pool: {POOL_PATH}", flush=True)
df = pd.read_pickle(POOL_PATH)
A = []; cache = {}
for _, r in df.iterrows():
    pid = str(r["id"])
    A.append({"poi_id": pid, "name": str(r.get("displayName", "")), "category": str(r.get("primaryType", "")),
              "lat": float(r["lat"]) if pd.notna(r.get("lat")) else 0.0,
              "lon": float(r["lng"]) if pd.notna(r.get("lng")) else 0.0,
              "address": str(r.get("shortFormattedAddress", "")), "description": "",
              "opening_hours": str(r.get("regularOpeningHours_text", ""))})
    e = r.get("embedding")
    cache[pid] = e.astype(np.float32) if isinstance(e, np.ndarray) and len(e) == 768 else np.zeros(768, np.float32)
poi_table_df = pd.DataFrame({
    "poi_id":        df["id"].astype(str).tolist(),
    "name":          df["displayName"].astype(str).tolist(),
    "primary_type":  df["primaryType"].astype(str).tolist(),
    "lat":           pd.to_numeric(df["lat"], errors="coerce").tolist(),
    "lon":           pd.to_numeric(df["lng"], errors="coerce").tolist(),
    "address":       df["shortFormattedAddress"].astype(str).tolist(),
    "opening_hours": df["regularOpeningHours_text"].astype(str).tolist(),
})
del df
try:
    admin_df = pd.read_parquet(os.environ.get("ADMIN_PATH", "./data/japan_admin/admin_regions.parquet"))
except Exception as _e:
    admin_df = None; print(f"      admin load failed: {_e}", flush=True)
print(f"      {len(A):,} POIs | admin {0 if admin_df is None else len(admin_df):,}", flush=True)

recs = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
if LIMIT: recs = recs[:LIMIT]
print(f"[2/3] {len(recs)} queries -> {OUT}", flush=True)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: done.add(json.loads(l)["idx"])
        except Exception: pass
if done: print(f"      resuming: {len(done)} already done", flush=True)

def _r(x):
    return None if x is None else round(float(x), 5)

WORKERS = int(os.environ.get("WORKERS", "8"))
from concurrent.futures import ThreadPoolExecutor

f = open(OUT, "a")
pcache = {}
_wlock = threading.Lock()
_plock = threading.Lock()
_prog = {"n": 0}
t0 = time.time()
todo = [(i, r) for i, r in enumerate(recs) if i not in done]
print(f"      {WORKERS} workers, {len(todo)} queries remaining", flush=True)


def _one(item):
    i, r = item
    q = r["question"]; uid = str(r["uid"]); traj = r["traj"]
    ans = str(r["answer"]["poi_id"]); qt = r["meta"]["query_type"]
    try:
        _last().clear()
        conn = build_conn_from_traj(uid, traj, poi_df=poi_table_df, admin_df=admin_df)
        ranked = score_query(q, uid, traj, A, cache, conn=conn, profile_cache=pcache, verbose=False)
        cands = []
        for p in ranked:
            sg = p.get("_signals") or {}
            cands.append([str(p["poi_id"]),
                          _r(sg.get("spa")), _r(sg.get("tem")), _r(sg.get("sem")), _r(sg.get("pref")),
                          _r(p.get("_cat_match")), _r(p.get("_score"))])
        ids = [c[0] for c in cands]
        d = _last()
        rec = {"idx": i, "uid": uid, "qt": qt, "gold": ans,
               "gold_rank": (ids.index(ans) + 1) if ans in ids else -1,
               "mode": d.get("mode"), "target_cat": d.get("target_cat"),
               "anchor": d.get("anchor"), "route_dir": d.get("route_dir"),
               "pt_source": d.get("pt_source"), "pt_name": d.get("pt_name"),
               "pt_lat": d.get("pt_lat"), "pt_lon": d.get("pt_lon"),
               "pt_poi": d.get("pt_poi"),
               "zone_name": d.get("zone_name"), "time_bucket": d.get("time_bucket"),
               "day_hint": d.get("day_hint"),
               "n_Aq": d.get("n_Aq"), "n_cand": len(cands),
               "cols": ["poi_id", "spa", "tem", "sem", "pref", "cat", "score"],
               "cands": cands}
        try:
            conn.close()
        except Exception:
            pass
    except Exception as e:
        print(f"  [{i+1}] ERR {str(e)[:110]}", flush=True)
        rec = {"idx": i, "uid": uid, "qt": qt, "gold": ans, "gold_rank": -99,
               "err": str(e)[:150], "cands": []}
    with _wlock:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    with _plock:
        _prog["n"] += 1
        n = _prog["n"]
        if n % 10 == 0:
            el = time.time() - t0
            print(f"  [{n}/{len(todo)}] {el:.0f}s eta {el/n*(len(todo)-n):.0f}s", flush=True)
    return None


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    list(ex.map(_one, todo))
f.close()

rows = [json.loads(l) for l in open(OUT)]
ok = [x for x in rows if x.get("gold_rank", -99) != -99]
hit1 = sum(1 for x in ok if 0 < x["gold_rank"] <= 1) / max(1, len(ok))
hit10 = sum(1 for x in ok if 0 < x["gold_rank"] <= 10) / max(1, len(ok))
ncand = [x["n_cand"] for x in ok]
print(f"\n[3/3] dump complete: {len(ok)}/{len(rows)} valid", flush=True)
print(f"      current pipeline R@1={hit1:.3f} R@10={hit10:.3f}", flush=True)
print(f"      candidate count: median={int(np.median(ncand))} max={max(ncand)} min={min(ncand)}", flush=True)
cov = defaultdict(int)
for x in ok:
    for c in x["cands"][:50]:
        for j, k in enumerate(["spa", "tem", "sem", "pref", "cat"], start=1):
            if c[j] is not None: cov[k] += 1
tot = sum(len(x["cands"][:50]) for x in ok)
print("      signal availability (top50): " + "  ".join(f"{k}={cov[k]/max(1,tot):.1%}" for k in ["spa","tem","sem","pref","cat"]), flush=True)
_p, _c = LLM_METER["prompt"], LLM_METER["completion"]
print("\n[LLM meter] prompt=%d completion=%d" % (_p, _c), flush=True)
_nq = max(1, len(ok))
print("            per-query: prompt=%.0f completion=%.0f" % (_p / _nq, _c / _nq), flush=True)
for (_site, _mdl), _cnt in sorted(LLM_METER["calls"].items()):
    print("   %-9s model=%-36s calls=%d" % (_site, _mdl, _cnt), flush=True)
print("DUMP_DONE", flush=True)
