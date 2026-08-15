import json, math, os
import weighting as W
import numpy as np

KEEP_POINT={2,8,11,16,22,24,26,40,41,42,44,49,52,54,62,67,69,75,79,87,89,95,131,132,134,
            136,137,138,141,142,145,147,162,165,171,175,181,184}

def load(p):
    return {r["idx"]: r for r in map(json.loads, open(p)) if r.get("gold_rank",-99)!=-99} if os.path.exists(p) else {}

A = load("/tmp/takeout_fixcur.jsonl")
point_idx = {i for i,r in A.items() if r.get("mode")=="point"}
drop = point_idx - KEEP_POINT
sub = sorted(set(A) - drop)

def met(d, ids):
    rk=[d[i]["gold_rank"] for i in ids if i in d]; n=len(rk) or 1
    return dict(n=len(rk), R1=sum(1 for x in rk if 0<x<=1)/n, R10=sum(1 for x in rk if 0<x<=10)/n,
                ND=sum((1/math.log2(x+1)) if 0<x<=10 else 0 for x in rk)/n,
                MRR=sum((1/x) if x>0 else 0 for x in rk)/n)

ARMS=[("full (EWM)", "/tmp/takeout_fixcur.jsonl"),
      ("w/o planner", "/tmp/fix_P1.jsonl"),
      ("w/o geometry compiler", "/tmp/fix_B1.jsonl"),
      ("w/o trajectory evidence", "/tmp/fix_B2.jsonl")]

for scope, lbl in [(sorted(A), "all 342 queries"), (sub, "%d-query subset" % len(sub))]:
    print("=" * 68); print(lbl + "  [leakage fixed]"); print("=" * 68)
    print("%-18s %5s %6s %6s %8s %6s %8s" % ("variant","N","R@1","R@10","NDCG@10","MRR","dMRR"))
    base=None
    for nm, p in ARMS:
        d=load(p)
        if not d: print("%-18s (not run)" % nm); continue
        m=met(d, scope)
        if base is None: base=m["MRR"]
        dd = "" if m["MRR"]==base else "%+.0f%%" % (100*(m["MRR"]-base)/base)
        print("%-18s %5d %6.3f %6.3f %8.3f %6.3f %8s" % (nm,m["n"],m["R1"],m["R10"],m["ND"],m["MRR"],dd))
    recs=[r for r in W.load("/tmp/takeout_fixcur.jsonl") if r["idx"] in set(scope)]
    if recs:
        mu,_,_=W.evaluate(recs,"uniform",0.5); me,_,_=W.evaluate(recs,"ewm",0.5)
        print("%-18s %5d %6.3f %6.3f %8.3f %6.3f %8s" % ("w/o entropy wt (uniform)",len(recs),
              mu["R@1"],mu["R@10"],mu["NDCG@10"],mu["MRR"],
              "%+.0f%%" % (100*(mu["MRR"]-me["MRR"])/me["MRR"])))
    print()
