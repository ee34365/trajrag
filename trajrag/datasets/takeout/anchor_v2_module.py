import math
from collections import Counter, defaultdict
from datetime import datetime

R_EARTH = 6371.0

def _hav(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(d))


def _parse_dt(s):
    if not s: return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00").replace(" ", "T"))
    except Exception:
        return None


def _extract_stays(traj):
    stays = []
    for v in traj or []:
        if v.get("status") != "visit":
            continue
        lat = v.get("d_lat") or v.get("lat")
        lon = v.get("d_lon") or v.get("lon")
        if lat is None or lon is None:
            continue
        s = _parse_dt(v.get("start_time"))
        e = _parse_dt(v.get("end_time"))
        if s is None:
            continue
        dwell = (e - s).total_seconds() / 60.0 if e else 30.0
        stays.append({
            "lat": float(lat), "lon": float(lon),
            "hour": s.hour, "wkday": s.weekday() < 5,
            "dwell": max(1.0, min(dwell, 24 * 60)),
            "name": v.get("poi_name") or "",
            "type": v.get("poi_type") or "",
            "pid": v.get("pid") or v.get("poi_id"),
        })
    return stays


def _aggregate_by_pid(stays):
    groups = defaultdict(list)
    solo = []
    for s in stays:
        if s.get("pid"):
            groups[s["pid"]].append(s)
        else:
            solo.append(s)
    merged = []
    for pid, ss in groups.items():
        w = sum(x["dwell"] for x in ss) or 1.0
        lat = sum(x["lat"] * x["dwell"] for x in ss) / w
        lon = sum(x["lon"] * x["dwell"] for x in ss) / w
        for x in ss:
            merged.append({**x, "lat": lat, "lon": lon})
    merged.extend(solo)
    return merged


def _dbscan(points, eps_km=0.15, min_pts=2):
    n = len(points)
    if n == 0: return []
    cell_deg = eps_km / 111.0
    grid = defaultdict(list)
    for i, p in enumerate(points):
        cx = int(p["lat"] / cell_deg)
        cy = int(p["lon"] / cell_deg)
        grid[(cx, cy)].append(i)

    def range_query(i):
        p = points[i]
        cx = int(p["lat"] / cell_deg)
        cy = int(p["lon"] / cell_deg)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    if _hav(p["lat"], p["lon"], points[j]["lat"], points[j]["lon"]) <= eps_km:
                        out.append(j)
        return out

    labels = [-1] * n
    cid = 0
    for i in range(n):
        if labels[i] != -1: continue
        nb = range_query(i)
        if len(nb) < min_pts:
            labels[i] = 0
            continue
        cid += 1
        labels[i] = cid
        seed = [x for x in nb if x != i]
        idx = 0
        while idx < len(seed):
            q = seed[idx]; idx += 1
            if labels[q] == 0:
                labels[q] = cid
            if labels[q] != -1:
                continue
            labels[q] = cid
            nb2 = range_query(q)
            if len(nb2) >= min_pts:
                for k in nb2:
                    if labels[k] in (-1, 0):
                        seed.append(k)
    return labels


def _cluster_stats(stays, labels):
    clusters = defaultdict(list)
    for s, l in zip(stays, labels):
        if l >= 1:
            clusters[l].append(s)
    out = []
    for cid, ss in clusters.items():
        total = sum(s["dwell"] for s in ss)
        night = sum(s["dwell"] for s in ss if s["hour"] < 6 or s["hour"] >= 22)
        work_h = sum(s["dwell"] for s in ss if s["wkday"] and 9 <= s["hour"] <= 17)
        wkday = sum(s["dwell"] for s in ss if s["wkday"])
        w = sum(s["dwell"] for s in ss)
        lat = sum(s["lat"] * s["dwell"] for s in ss) / w
        lon = sum(s["lon"] * s["dwell"] for s in ss) / w
        names = Counter(); types = Counter(); pids = Counter()
        for s in ss:
            if s["name"]: names[s["name"]] += s["dwell"]
            if s["type"]: types[s["type"]] += s["dwell"]
            if s["pid"]: pids[s["pid"]] += s["dwell"]
        out.append({
            "cid": cid,
            "n_stays": len(ss),
            "total_h": round(total / 60, 1),
            "night_h": round(night / 60, 1),
            "work_h": round(work_h / 60, 1),
            "wkday_pct": round(100 * wkday / max(1, total)),
            "night_pct": round(100 * night / max(1, total)),
            "work_pct": round(100 * work_h / max(1, total)),
            "lat": round(lat, 6), "lon": round(lon, 6),
            "top_name": names.most_common(1)[0][0] if names else "",
            "top_type": types.most_common(1)[0][0] if types else "",
            "top_pid": pids.most_common(1)[0][0] if pids else "",
        })
    return out


_EXCLUDE_TYPES = {
    "transit_station", "train_station", "subway_station", "bus_stop",
    "bus_station", "light_rail_station", "airport", "ferry_terminal",
    "gas_station", "parking", "convenience_store", "atm",
}


def _is_excluded(cl):
    return cl["top_type"] in _EXCLUDE_TYPES


def _pick_home(clusters):
    if not clusters: return None
    cands = [c for c in clusters
             if c["night_h"] >= 20 and c["wkday_pct"] >= 50 and not _is_excluded(c)]
    cands.sort(key=lambda c: (-c["night_h"], -c["total_h"]))
    if cands: return cands[0]
    cands = [c for c in clusters if c["night_h"] >= 10 and not _is_excluded(c)]
    cands.sort(key=lambda c: -c["night_h"])
    if cands: return cands[0]
    cands = [c for c in clusters if c["total_h"] >= 5 and c["night_h"] >= 3 and not _is_excluded(c)]
    cands.sort(key=lambda c: (-c["night_pct"], -c["night_h"]))
    return cands[0] if cands else None


def _pick_work(clusters, home):
    if not clusters: return None
    def dist_home(c):
        if not home: return 999
        return _hav(home["lat"], home["lon"], c["lat"], c["lon"])
    cands = [c for c in clusters
             if not _is_excluded(c)
             and c["work_h"] >= 15
             and c["wkday_pct"] >= 55
             and dist_home(c) >= 0.5]
    cands.sort(key=lambda c: (-c["work_h"], -c["wkday_pct"]))
    if cands: return cands[0]
    cands = [c for c in clusters
             if not _is_excluded(c)
             and c["work_h"] >= 8
             and c["wkday_pct"] >= 50
             and dist_home(c) >= 0.5]
    cands.sort(key=lambda c: -c["work_h"])
    return cands[0] if cands else None


def detect_from_traj(traj):
    stays_raw = _extract_stays(traj)
    if len(stays_raw) < 5:
        return {"home": None, "work": None, "clusters": [], "n_stays": len(stays_raw)}
    stays = _aggregate_by_pid(stays_raw)
    labels = _dbscan(stays, eps_km=0.15, min_pts=2)
    cls = _cluster_stats(stays, labels)
    cls.sort(key=lambda c: -c["total_h"])
    home = _pick_home(cls)
    work = _pick_work(cls, home)
    return {"home": home, "work": work, "clusters": cls[:8], "n_stays": len(stays_raw)}


def anchor_from_cluster(cl, kind="home"):
    if cl is None: return None
    return {
        "anchor_name": (cl.get("top_name") or kind)[:60],
        "poi_id": cl.get("top_pid") or "",
        "lat": float(cl["lat"]),
        "lon": float(cl["lon"]),
        "source": f"anchor_v2_{kind}",
    }
