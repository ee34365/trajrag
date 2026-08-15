import json, math, os
KEEP_POINT={2,8,11,16,22,24,26,40,41,42,44,49,52,54,62,67,69,75,79,87,89,95,131,132,
            134,136,137,138,141,142,145,147,162,165,171,175,181,184}
recs=[json.loads(l) for l in open("./data/takeout/trajrag_takeout_final.jsonl")]
base={r["idx"]:r for r in map(json.loads,open("/tmp/takeout_fixcur.jsonl")) if r.get("gold_rank",-99)!=-99}
point_idx={i for i,r in base.items() if r.get("mode")=="point"}
SUB=sorted(set(base)-(point_idx-KEEP_POINT)); SUBQ={recs[i]["question"] for i in SUB}
def met(rk):
    n=len(rk) or 1
    return dict(n=len(rk),R1=sum(x<=1 and x>0 for x in rk)/n,R5=sum(0<x<=5 for x in rk)/n,
        R10=sum(0<x<=10 for x in rk)/n,ND=sum((1/math.log2(x+1)) if 0<x<=10 else 0 for x in rk)/n,
        MRR=sum((1/x) if x>0 else 0 for x in rk)/n)
def bl(p,qs=None):
    if not os.path.exists(p): return None
    lg=json.load(open(p))["log"]; return met([x["rank"] for x in lg if qs is None or x["query"] in qs])
def ours(p,sub=True):
    d={r["idx"]:r for r in map(json.loads,open(p)) if r.get("gold_rank",-99)!=-99}
    ids=[i for i in (SUB if sub else sorted(d)) if i in d]
    return met([d[i]["gold_rank"] for i in ids])
HDR="%-14s %5s %6s %6s %6s %8s %6s"%("","N","R@1","R@5","R@10","NDCG@10","MRR")
def row(l,m):
    if m is None: print("%-14s (missing)"%l);return
    print("%-14s %5d %6.3f %6.3f %6.3f %8.3f %6.3f"%(l,m["n"],m["R1"],m["R5"],m["R10"],m["ND"],m["MRR"]))
TK=[("SD","/tmp/bl_sd_tk/sd_results.json"),("GeoLLM","/tmp/bl_srag_tk/geollm_results.json"),
    ("Naive RAG","/tmp/bl_srag_tk/naiverag_results.json"),("Spatial-RAG","/tmp/bl_sr2_tk/spatialrag_results.json"),
    ("SemaSK","/tmp/bl_semask_tk/semask_results.json"),("ST","/tmp/bl_srag_tk/st_results.json")]
print("="*70);print("(1) Takeout 326-query subset");print("="*70);print(HDR)
o=ours("/tmp/takeout_fixcur.jsonl")
best=None
for nm,f in TK:
    m=bl(f,SUBQ);row(nm,m)
    if m and (best is None or m["MRR"]>best[1]["MRR"]): best=(nm,m)
row("TrajRAG(ours)",o)
if best: print("strongest baseline %s: TrajRAG MRR %.2fx, R@10 %.2fx"%(best[0],o["MRR"]/best[1]["MRR"],o["R10"]/best[1]["R10"]))
GE=[("Naive RAG","/tmp/nr_full233/naiverag_results.json"),("GeoLLM","/tmp/bl_srag_ge/geollm_results.json"),
    ("SemaSK","/tmp/bl_semask_ge/semask_results.json"),("SD","/tmp/bl_srag_ge/sd_results.json"),
    ("Spatial-RAG","/tmp/bl_sr2_ge/spatialrag_results.json"),("ST","/tmp/bl_srag_ge/st_results.json")]
print("\n"+"="*70);print("(2) GeoLife 233 (no pre-filter)");print("="*70);print(HDR)
og=ours("/tmp/fix_geolife.jsonl",sub=False)
bestg=None
for nm,f in GE:
    m=bl(f);row(nm,m)
    if m and (bestg is None or m["MRR"]>bestg[1]["MRR"]): bestg=(nm,m)
row("TrajRAG(ours)",og)
if bestg: print("strongest baseline %s: TrajRAG MRR %.2fx, R@10 %.2fx"%(bestg[0],og["MRR"]/bestg[1]["MRR"],og["R10"]/bestg[1]["R10"]))
