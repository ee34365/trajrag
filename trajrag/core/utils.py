import math
from typing import Any, Iterable
import numpy as np
import pandas as pd


def safe_float(x: Any):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def valid_latlon(lat, lon) -> bool:
    lat = safe_float(lat)
    lon = safe_float(lon)
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def haversine(lat1, lon1, lat2, lon2):
    if not (valid_latlon(lat1, lon1) and valid_latlon(lat2, lon2)):
        return float("inf")
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def haversine_vec(lats1, lons1, lat2, lon2):
    lats1 = np.asarray(lats1, dtype=float)
    lons1 = np.asarray(lons1, dtype=float)
    if not valid_latlon(lat2, lon2):
        return np.full_like(lats1, np.inf, dtype=float)
    R = 6371.0
    lat1 = np.radians(lats1)
    lon1 = np.radians(lons1)
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def point_to_segment_dist(plat, plon, alat, alon, blat, blon):
    if not (valid_latlon(plat, plon) and valid_latlon(alat, alon) and valid_latlon(blat, blon)):
        return float("inf")
    plat, plon, alat, alon, blat, blon = map(float, [plat, plon, alat, alon, blat, blon])
    cos_lat = math.cos(math.radians((alat + blat) / 2))
    bx = (blon - alon) * cos_lat * 111320
    by = (blat - alat) * 111320
    px = (plon - alon) * cos_lat * 111320
    py = (plat - alat) * 111320
    ab2 = bx * bx + by * by
    if ab2 < 1e-6:
        return haversine(plat, plon, alat, alon)
    t = max(0.0, min(1.0, (px * bx + py * by) / ab2))
    dx = px - t * bx
    dy = py - t * by
    return math.sqrt(dx * dx + dy * dy) / 1000.0


def dist_to_polyline(plat, plon, polyline_points):
    if not polyline_points:
        return float("inf")
    if len(polyline_points) < 2:
        p0 = polyline_points[0]
        return haversine(plat, plon, p0.get("lat"), p0.get("lon"))
    d_min = float("inf")
    for i in range(len(polyline_points) - 1):
        a = polyline_points[i]
        b = polyline_points[i + 1]
        d = point_to_segment_dist(plat, plon, a.get("lat"), a.get("lon"), b.get("lat"), b.get("lon"))
        d_min = min(d_min, d)
    return d_min


def sample_segment(olat, olon, dlat, dlon, step_km=1.0):
    if not (valid_latlon(olat, olon) and valid_latlon(dlat, dlon)):
        return []
    dist = haversine(olat, olon, dlat, dlon)
    if not math.isfinite(dist) or dist < 1e-6:
        return []
    n = max(1, int(dist / step_km) + 1)
    pts = []
    for k in range(1, n):
        t = k / n
        pts.append((float(olat) + t * (float(dlat) - float(olat)),
                    float(olon) + t * (float(dlon) - float(olon))))
    return pts


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


def cosine_01(a, b):
    s = cos_sim(a, b)
    if s is None:
        return None
    return max(0.0, min(1.0, float(s)))


def cos_sim_vec(a, B, clip01=True):
    a = np.asarray(a, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if B.size == 0:
        return np.array([], dtype=np.float32)
    na = np.linalg.norm(a)
    nB = np.linalg.norm(B, axis=1)
    denom = (na + 1e-8) * (nB + 1e-8)
    sims = (B @ a) / denom
    sims[(nB < 1e-8) | (na < 1e-8)] = 0.0
    if clip01:
        sims = np.clip(sims, 0.0, 1.0)
    return sims.astype(np.float32)


def traj_to_dataframe(traj_list, uid):
    rows = []
    for t in traj_list or []:
        status = t.get("status", "")
        if status not in ("visit", "activity"):
            continue
        try:
            d_lat = safe_float(t.get("d_lat"))
            d_lon = safe_float(t.get("d_lon"))
            o_lat = safe_float(t.get("o_lat"))
            o_lon = safe_float(t.get("o_lon"))
            rows.append({
                "uid": str(uid),
                "event_start": str(t.get("start_time", "")),
                "event_end": str(t.get("end_time", "")),
                "poi_id": str(t.get("pid") or t.get("poi_id") or ""),
                "poi_name": str(t.get("poi_name") or ""),
                "poi_type": str(t.get("poi_type") or ""),
                "lat": d_lat if status == "visit" else None,
                "lon": d_lon if status == "visit" else None,
                "o_lat": o_lat,
                "o_lon": o_lon,
                "d_lat": d_lat,
                "d_lon": d_lon,
                "status": status,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def topk_indices(scores: np.ndarray, k: int):
    scores = np.asarray(scores)
    if len(scores) == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def brand_dedupe(candidates: list, k: int = 50) -> list:
    seen = set()
    out = []
    for p in candidates:
        name = str(p.get("name","")).strip().lower()
        import re as _re
        name = _re.sub(r"[\s\u3000;,.&\-_/]+", " ", name).strip()
        district = str(p.get("address",""))[:12].strip().lower()
        key = (name, district)
        if not name:
            out.append(p); continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= k:
            break
    return out
