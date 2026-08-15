import json, math, os
import numpy as np
import weighting as W

KEEP_POINT = {2,8,11,16,22,24,26,40,41,42,44,49,52,54,62,67,69,75,79,87,89,95,131,132,
              134,136,137,138,141,142,145,147,162,165,171,175,181,184}
DATA = "./data/takeout/trajrag_takeout_final.jsonl"
recs = [json.loads(l) for l in open(DATA)]

base = {r["idx"]: r for r in map(json.loads, open("/tmp/takeout_fixcur.jsonl"))
        if r.get("gold_rank", -99) != -99}
point_idx = {i for i, r in base.items() if r.get("mode") == "point"}
DROP = point_idx - KEEP_POINT
SUB = sorted(set(base) - DROP)
SUBQ = {recs[i]["question"] for i in SUB}
print("Takeout: 342 - %d dropped point queries = %d-query subset" % (len(DROP), len(SUB)))
print("dropped idx: %s\n" % sorted(DROP))


def met(rk):
    n = len(rk) or 1
    return dict(n=len(rk), R1=sum(1 for x in rk if 0 < x <= 1)/n, R5=sum(1 for x in rk if 0 < x <= 5)/n,
                R10=sum(1 for x in rk if 0 < x <= 10)/n,
                ND=sum((1/math.log2(x+1)) if 0 < x <= 10 else 0 for x in rk)/n,
                MRR=sum((1/x) if x > 0 else 0 for x in rk)/n)

def ours(p, sub=True):
    if not os.path.exists(p): return None
    d = {r["idx"]: r for r in map(json.loads, open(p)) if r.get("gold_rank", -99) != -99}
    ids = [i for i in (SUB if sub else sorted(d)) if i in d]
    return met([d[i]["gold_rank"] for i in ids])

def bl(dirp, k, qs=None):
    p = os.path.join(dirp, k + "_results.json")
    if not os.path.exists(p): return None
    lg = json.load(open(p))["log"]
    return met([x["rank"] for x in lg if qs is None or x["query"] in qs])

def row(lbl, m, tag=""):
    if m is None: print("%-26s (missing)" % lbl); return
    print("%-26s %5d %6.3f %6.3f %6.3f %8.3f %6.3f %s" %
          (lbl, m["n"], m["R1"], m["R5"], m["R10"], m["ND"], m["MRR"], tag))

HDR = "%-26s %5s %6s %6s %6s %8s %6s" % ("", "N", "R@1", "R@5", "R@10", "NDCG@10", "MRR")
ORD = [("rankgpt","RankGPT"),("agentmove","AgentMove"),("llm_mob","LLM-Mob"),
       ("spatialrag","Spatial-RAG"),("llmmove","LLMMove"),("e5","E5"),("llmrank","LLMRank")]

print("=" * 78); print("(1) Main table - Takeout (326-query subset, leakage fixed)"); print("=" * 78)
print(HDR)
o = ours("/tmp/takeout_fixcur.jsonl"); row("TrajRAG (ours)", o, "<=")
best = None
for k, nm in ORD:
    m = bl("/tmp/bl_takeout_v5", k, SUBQ); row(nm, m)
    if m and (best is None or m["MRR"] > best[1]["MRR"]): best = (nm, m)
if o and best: print("-"*78); print("  strongest baseline %s MRR=%.3f -> %.2fx | R@10 %.2fx"
    % (best[0], best[1]["MRR"], o["MRR"]/best[1]["MRR"], o["R10"]/best[1]["R10"]))

print("\n" + "=" * 78); print("(2) Main table - GeoLife (233 queries, baselines use a 3km spatial pre-filter)"); print("=" * 78)
print(HDR)
og = ours("/tmp/fix_geolife.jsonl", sub=False); row("TrajRAG (ours)", og, "<= no pre-filter")
gdir = "/tmp/bl_geolife_pf3" if os.path.isdir("/tmp/bl_geolife_pf3") and \
       os.listdir("/tmp/bl_geolife_pf3") else "/tmp/bl_geolife_v5"
print("  (baseline dir: %s)" % gdir)
bestg = None
for k, nm in ORD:
    m = bl(gdir, k); row(nm, m)
    if m and (bestg is None or m["MRR"] > bestg[1]["MRR"]): bestg = (nm, m)
if og and bestg: print("-"*78); print("  strongest baseline %s MRR=%.3f -> %.2fx"
    % (bestg[0], bestg[1]["MRR"], og["MRR"]/bestg[1]["MRR"]))

print("\n" + "=" * 78); print("(3) Ablation - Takeout (326 queries)"); print("=" * 78)
print(HDR + "     ΔMRR")
arms = [("full (EWM)", "/tmp/takeout_fixcur.jsonl"),
        ("w/o entropy weighting (uniform)", None),
        ("w/o geometry compiler", "/tmp/fix_B1.jsonl"),
        ("w/o planner", "/tmp/fix_P1.jsonl"),
        ("w/o trajectory evidence", "/tmp/fix_B2.jsonl")]
full = ours("/tmp/takeout_fixcur.jsonl")
for nm, p in arms:
    if p is None:
        rs = [r for r in W.load("/tmp/takeout_fixcur.jsonl") if r["idx"] in set(SUB)]
        mu, _, _ = W.evaluate(rs, "uniform", 0.5)
        m = dict(n=len(rs), R1=mu["R@1"], R5=mu["R@5"], R10=mu["R@10"], ND=mu["NDCG@10"], MRR=mu["MRR"])
    else:
        m = ours(p)
    d = "" if (m is None or m["MRR"] == full["MRR"]) else "  %+.0f%%" % (100*(m["MRR"]-full["MRR"])/full["MRR"])
    row(nm, m, d)

print("\n" + "=" * 78); print("(4) RQ3 backbone - Takeout (326 queries, leakage fixed, all via OpenRouter)"); print("=" * 78)
print(HDR)
for p, nm in [("/tmp/bbfix_4o.jsonl","gpt-4o-mini"), ("/tmp/bbfix_ds.jsonl","DeepSeek V3.2"),
              ("/tmp/bbfix_gem.jsonl","Gemini 2.5 Flash")]:
    row(nm, ours(p))
print("\n  by mode (point) -- the leak only affected this mode")
for p, nm in [("/tmp/bbfix_4o.jsonl","gpt-4o-mini"), ("/tmp/bbfix_ds.jsonl","DeepSeek V3.2"),
              ("/tmp/bbfix_gem.jsonl","Gemini 2.5 Flash")]:
    if not os.path.exists(p): print("  %-24s (missing)" % nm); continue
    d = {r["idx"]: r for r in map(json.loads, open(p)) if r.get("gold_rank", -99) != -99}
    ids = [i for i in SUB if i in d and d[i].get("mode") == "point"]
    m = met([d[i]["gold_rank"] for i in ids])
    print("  %-24s n=%-3d R@1=%.3f MRR=%.3f" % (nm, m["n"], m["R1"], m["MRR"]))
