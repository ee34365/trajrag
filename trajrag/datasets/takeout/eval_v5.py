import os, sys, json, math, time
_early_st = None
try:
    from sentence_transformers import SentenceTransformer as _EarlyST
    _early_st = _EarlyST("intfloat/multilingual-e5-base")
    print("[AH2] pre-loaded e5-base before pipeline import", flush=True)
except Exception as _e:
    print(f"[AH2] pre-load failed: {_e}", flush=True)
import numpy as np, pandas as pd
from collections import defaultdict
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

POOL_PATH = "./data/takeout/poi_store_unified_v3_e5base.pkl"
DATA_PATH = os.environ.get("EVAL_DATA", "./data/takeout/trajrag_kept_v14_final_v5_balanced.jsonl")
RESULT_PATH = os.environ.get("EVAL_OUT", "/tmp/eval_v5_results.json")
LIMIT = int(os.environ.get("EVAL_LIMIT", "0"))

def load():
    print(f"[1/3] Pool: {POOL_PATH}", flush=True)
    df = pd.read_pickle(POOL_PATH)
    A=[]; cache={}
    for _,r in df.iterrows():
        pid=str(r["id"])
        A.append({"poi_id":pid,"name":str(r.get("displayName","")),"category":str(r.get("primaryType","")),
                  "lat":float(r["lat"]) if pd.notna(r.get("lat")) else 0.0,
                  "lon":float(r["lng"]) if pd.notna(r.get("lng")) else 0.0,
                  "address":str(r.get("shortFormattedAddress","")),"description":"",
                  "opening_hours":str(r.get("regularOpeningHours_text",""))})
        e=r.get("embedding")
        cache[pid]= e.astype(np.float32) if isinstance(e,np.ndarray) and len(e)==768 else np.zeros(768,np.float32)
    print(f"      {len(A):,} POIs", flush=True)
    recs=[json.loads(l) for l in open(DATA_PATH) if l.strip()]
    if LIMIT: recs=recs[:LIMIT]
    print(f"[2/3] {len(recs)} queries", flush=True)
    return A,cache,recs

def run():
    A,cache,recs=load()
    KS=[1,3,5,10]
    M={f"R@{k}":[] for k in KS}; M["NDCG@10"]=[]; M["MRR"]=[]
    byqt=defaultdict(lambda:{f"R@{k}":[] for k in KS}|{"MRR":[]})
    byb=defaultdict(lambda:{f"R@{k}":[] for k in KS}|{"MRR":[]})
    t0=time.time(); pcache={}
    for i,r in enumerate(recs):
        q=r["question"]; uid=str(r["uid"]); traj=r["traj"]
        ans=str(r["answer"]["poi_id"]); qt=r["meta"]["query_type"]; bk=r["meta"].get("bucket","?")
        try:
            conn=build_conn_from_traj(uid,traj)
            ranked=score_query(q,uid,traj,A,cache,conn=conn,profile_cache=pcache,verbose=False)
            rids=[str(p["poi_id"]) for p in ranked]
            rank=rids.index(ans)+1 if ans in rids else -1
        except Exception as e:
            print(f"  [{i+1}] ERR {str(e)[:100]}",flush=True); rank=-1
        for k in KS:
            v=1.0 if 0<rank<=k else 0.0
            M[f"R@{k}"].append(v); byqt[qt][f"R@{k}"].append(v); byb[bk][f"R@{k}"].append(v)
        nd=(1.0/math.log2(rank+1)) if 0<rank<=10 else 0.0
        mr=(1.0/rank) if rank>0 else 0.0
        M["NDCG@10"].append(nd); M["MRR"].append(mr); byqt[qt]["MRR"].append(mr); byb[bk]["MRR"].append(mr)
        if (i+1)%10==0:
            el=time.time()-t0
            print(f"  [{i+1}/{len(recs)}] R@1={np.mean(M['R@1']):.3f} R@5={np.mean(M['R@5']):.3f} "
                  f"R@10={np.mean(M['R@10']):.3f} MRR={np.mean(M['MRR']):.3f} ({el:.0f}s, eta {el/(i+1)*(len(recs)-i-1):.0f}s)",flush=True)
    res={"N":len(recs),"pool":len(A),
         **{m:round(float(np.mean(v)),4) for m,v in M.items() if v},
         "by_qtype":{k:{m:round(float(np.mean(vs)),4) for m,vs in d.items() if vs} for k,d in byqt.items()},
         "by_bucket":{k:{m:round(float(np.mean(vs)),4) for m,vs in d.items() if vs} for k,d in byb.items()}}
    json.dump(res,open(RESULT_PATH,"w"),indent=2,ensure_ascii=False)
    print("\n=== v5 (327) AL-pipeline / gpt-4o-mini / 270K pool ===")
    for k in ["N","pool","R@1","R@3","R@5","R@10","NDCG@10","MRR"]: print(f"  {k}: {res.get(k)}")
    print("\nby query_type:")
    for qt,d in sorted(res["by_qtype"].items()): print(f"  {qt:<8} R@1={d.get('R@1',0):.3f} R@10={d.get('R@10',0):.3f} MRR={d.get('MRR',0):.3f}")
    print("by bucket:")
    for bk,d in sorted(res["by_bucket"].items()): print(f"  {bk:<22} R@1={d.get('R@1',0):.3f} R@10={d.get('R@10',0):.3f} MRR={d.get('MRR',0):.3f}")

if __name__=="__main__": run()
