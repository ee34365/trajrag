#!/bin/bash
# 等 B1 与 Geolife 3km 都结束，自动汇总最终结果
cd .
while pgrep -f "[r]un_b1b2" > /dev/null || pgrep -f "[r]un_geo3km" > /dev/null; do sleep 60; done
echo "===== 全部完成，汇总 ====="
echo
python3 final_326.py 2>/dev/null
echo
echo "===== Geolife baseline (3km 预过滤) ====="
python3 - <<'PY'
import json, os
d3, d0 = "/tmp/bl_geolife_pf3", "/tmp/bl_geolife_v5"
ours = [json.loads(l) for l in open("/tmp/fix_geolife.jsonl")]
ok = [r for r in ours if r.get("gold_rank",-99)!=-99]
n = len(ok) or 1
rk = [r["gold_rank"] for r in ok]
import math
print("%-16s %6s %6s %6s %8s %6s" % ("方法","R@1","R@5","R@10","NDCG@10","MRR"))
print("%-16s %6.3f %6.3f %6.3f %8.3f %6.3f  <= 本文(无预过滤)" % ("TrajRAG",
      sum(1 for x in rk if 0<x<=1)/n, sum(1 for x in rk if 0<x<=5)/n, sum(1 for x in rk if 0<x<=10)/n,
      sum((1/math.log2(x+1)) if 0<x<=10 else 0 for x in rk)/n, sum((1/x) if x>0 else 0 for x in rk)/n))
for k,nm in [("rankgpt","RankGPT"),("agentmove","AgentMove"),("llm_mob","LLM-Mob"),
             ("spatialrag","Spatial-RAG"),("llmmove","LLMMove"),("e5","E5"),("llmrank","LLMRank")]:
    row=[]
    for d,tag in [(d0,"原版"),(d3,"3km")]:
        p=os.path.join(d,k+"_results.json")
        row.append(json.load(open(p)) if os.path.exists(p) else None)
    a,b=row
    if b: print("%-16s %6.3f %6.3f %6.3f %8.3f %6.3f   (原版 MRR %.3f)" % (nm,
        b.get("R@1",0),b.get("R@5",0),b.get("R@10",0),b.get("NDCG@10",0),b.get("MRR",0),
        a.get("MRR",0) if a else 0))
    elif a: print("%-16s (3km 未完成)  原版 MRR %.3f" % (nm, a.get("MRR",0)))
PY
echo FINAL_DONE
