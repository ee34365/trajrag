import json, math
import numpy as np
from collections import defaultdict

SIGS = ["spa", "tem", "sem", "pref"]


def load(path):
    out = []
    for l in open(path):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("gold_rank", -99) != -99 and r.get("cands"):
            out.append(r)
    return out


def _matrix(rec, alpha):
    cands = rec["cands"]
    ids = [c[0] for c in cands]
    raw = {"spa": [], "tem": [], "sem": [], "pref": [], "cat": []}
    for c in cands:
        raw["spa"].append(c[1]); raw["tem"].append(c[2]); raw["sem"].append(c[3])
        raw["pref"].append(c[4]); raw["cat"].append(c[5])

    def col(name):
        v = raw[name]
        if all(x is None for x in v):
            return None
        arr = np.array([np.nan if x is None else float(x) for x in v], dtype=np.float64)
        m = np.nanmean(arr) if not np.all(np.isnan(arr)) else 0.0
        return np.where(np.isnan(arr), m, arr)

    spa, tem, sem, pref, cat = (col(k) for k in ["spa", "tem", "sem", "pref", "cat"])
    if sem is None and cat is None:
        sem_new = None
    elif sem is None:
        sem_new = cat
    elif cat is None:
        sem_new = sem
    else:
        sem_new = alpha * sem + (1.0 - alpha) * cat

    cols = {"spa": spa, "tem": tem, "sem": sem_new, "pref": pref}
    avail = [k for k in SIGS if cols[k] is not None]
    if not avail:
        return None, [], ids
    M = np.column_stack([cols[k] for k in avail])
    return M, avail, ids


def _minmax(M):
    lo, hi = M.min(axis=0), M.max(axis=0)
    rng = hi - lo
    out = np.zeros_like(M)
    nz = rng > 1e-12
    out[:, nz] = (M[:, nz] - lo[nz]) / rng[nz]
    return out


def w_uniform(M):
    return np.ones(M.shape[1]) / M.shape[1]


def w_ewm(M):
    n = M.shape[0]
    if n < 2:
        return w_uniform(M)
    N = _minmax(M) + 1e-12
    P = N / N.sum(axis=0, keepdims=True)
    E = -(P * np.log(P)).sum(axis=0) / math.log(n)
    D = np.clip(1.0 - E, 0.0, None)
    if D.sum() <= 1e-12:
        return w_uniform(M)
    return D / D.sum()


def w_critic(M):
    n, m = M.shape
    if n < 3 or m < 2:
        return w_uniform(M)
    N = _minmax(M)
    sd = N.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(N, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    C = sd * (1.0 - R).sum(axis=1)
    C = np.clip(C, 0.0, None)
    if C.sum() <= 1e-12:
        return w_uniform(M)
    return C / C.sum()


SCHEMES = {"uniform": w_uniform, "ewm": w_ewm, "critic": w_critic}


def rank_gold(rec, scheme, alpha):
    M, avail, ids = _matrix(rec, alpha)
    if M is None:
        return -1, {}
    w = SCHEMES[scheme](M)
    S = _minmax(M) @ w
    order = np.argsort(-S, kind="stable")
    ranked = [ids[i] for i in order]
    g = rec["gold"]
    return (ranked.index(g) + 1 if g in ranked else -1), dict(zip(avail, w.round(4)))


def metrics(ranks):
    n = len(ranks) or 1
    out = {}
    for k in (1, 5, 10):
        out[f"R@{k}"] = sum(1 for r in ranks if 0 < r <= k) / n
    out["NDCG@10"] = sum((1 / math.log2(r + 1)) if 0 < r <= 10 else 0 for r in ranks) / n
    out["MRR"] = sum((1 / r) if r > 0 else 0 for r in ranks) / n
    return out


def evaluate(recs, scheme, alpha):
    ranks, wacc = [], defaultdict(list)
    for r in recs:
        rk, w = rank_gold(r, scheme, alpha)
        ranks.append(rk)
        for k, v in w.items():
            wacc[k].append(v)
    return metrics(ranks), {k: float(np.mean(v)) for k, v in wacc.items()}, ranks


def report(path, alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    recs = load(path)
    print(f"cache: {path}  {len(recs)} valid records\n")

    base = metrics([r["gold_rank"] for r in recs])
    print("%-28s %6s %6s %6s %8s %6s" % ("scheme", "R@1", "R@5", "R@10", "NDCG@10", "MRR"))
    print("%-28s %6.3f %6.3f %6.3f %8.3f %6.3f" % (
        "fixed (hand-tuned + 0.35*cat)", base["R@1"], base["R@5"], base["R@10"], base["NDCG@10"], base["MRR"]))

    best = None
    for scheme in ("uniform", "ewm", "critic"):
        for a in alphas:
            m, w, _ = evaluate(recs, scheme, a)
            tag = f"{scheme} a={a}"
            print("%-28s %6.3f %6.3f %6.3f %8.3f %6.3f" % (
                tag, m["R@1"], m["R@5"], m["R@10"], m["NDCG@10"], m["MRR"]))
            if best is None or m["MRR"] > best[0]["MRR"]:
                best = (m, w, scheme, a)

    m, w, scheme, a = best
    print(f"\nbest: {scheme} alpha={a}  MRR={m['MRR']:.3f}")
    print("  mean weights: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(w.items())))

    print("\n--- EWM mean weights by query type ---")
    byqt = defaultdict(list)
    for r in recs:
        byqt[r["qt"]].append(r)
    for qt in sorted(byqt):
        _, w2, rks = evaluate(byqt[qt], "ewm", a)
        mm = metrics(rks)
        print("  %-8s n=%-4d %s | MRR=%.3f" % (
            qt, len(byqt[qt]), "  ".join(f"{k}={v:.2f}" for k, v in sorted(w2.items())), mm["MRR"]))


if __name__ == "__main__":
    import sys
    report(sys.argv[1] if len(sys.argv) > 1 else "/tmp/signals_takeout.jsonl")
