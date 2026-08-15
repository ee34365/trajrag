from typing import List, Dict, Any
import numpy as np
from constants import BASE_WEIGHTS, R_ROUTE_KM, R_POINT_KM, ZONE_ALPHA
from utils import haversine, cosine_01, dist_to_polyline, valid_latlon

try:
    from shapely.geometry import Point as _ShpPoint
    from shapely import wkt as _shp_wkt
    _SHAPELY_OK = True
except Exception:
    _SHAPELY_OK = False

_POLYGON_CACHE = {}

def _get_polygon(wkt_str):
    if not _SHAPELY_OK or not wkt_str:
        return None
    if wkt_str in _POLYGON_CACHE:
        return _POLYGON_CACHE[wkt_str]
    try:
        poly = _shp_wkt.loads(wkt_str)
        _POLYGON_CACHE[wkt_str] = poly
        return poly
    except Exception:
        _POLYGON_CACHE[wkt_str] = None
        return None


def s_spatial(p, R_tau: dict, s_q: str):
    import math
    if not valid_latlon(p.get("lat"), p.get("lon")):
        return None

    if s_q == "point":
        point = R_tau.get("point")
        if not point: return None
        d = haversine(p["lat"], p["lon"], point.get("lat"), point.get("lon"))
        if not np.isfinite(d): return None
        return float(math.exp(-d / R_POINT_KM))

    if s_q == "zone":
        z = R_tau.get("zone")
        if not z: return None
        polygon_wkt = z.get("polygon_wkt")
        poly = _get_polygon(polygon_wkt) if polygon_wkt else None
        bbox = z.get("bbox")
        if poly is not None:
            pt = _ShpPoint(p["lon"], p["lat"])
            if poly.contains(pt) or poly.touches(pt):
                return 1.0
            try:
                d_deg = pt.distance(poly.boundary)
                d_km = d_deg * 111.0
            except Exception:
                d_km = None
            if d_km is not None and d_km < 0.5:
                return float(math.exp(-d_km / 0.5))
            if bbox and len(bbox) == 4 and d_km is None:
                lat_min, lat_max, lon_min, lon_max = bbox
                if lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max:
                    return 0.8
            if d_km is not None:
                return float(math.exp(-d_km / 1.0))
            return 0.0
        if bbox and len(bbox) == 4:
            lat_min, lat_max, lon_min, lon_max = bbox
            inside = (lat_min <= p["lat"] <= lat_max) and (lon_min <= p["lon"] <= lon_max)
            if inside:
                return 1.0
            dlat = max(0.0, lat_min - p["lat"], p["lat"] - lat_max)
            dlon = max(0.0, lon_min - p["lon"], p["lon"] - lon_max)
            cos_lat = math.cos(math.radians((lat_min + lat_max) / 2))
            d_edge = math.sqrt((dlat * 111.32) ** 2 + (dlon * 111.32 * cos_lat) ** 2)
            return float(math.exp(-d_edge / 1.0))
        c_lat = z.get("center_lat"); c_lon = z.get("center_lon")
        if c_lat is None or c_lon is None: return None
        d = haversine(p["lat"], p["lon"], c_lat, c_lon)
        if not np.isfinite(d): return None
        R_ZONE = 1.5
        return float(math.exp(-d / R_ZONE))

    if s_q == "route":
        route = R_tau.get("route")
        if not route: return None
        
        chains_to_work = route.get("chains_to_work") or []
        chains_to_home = route.get("chains_to_home") or []
        direction = route.get("route_name", "home_to_work")
        if direction == "home_to_work":
            target_chains = chains_to_work
            opposite_chains = chains_to_home
        else:
            target_chains = chains_to_home
            opposite_chains = chains_to_work
        
        opposite_used = False
        if not target_chains:
            if opposite_chains:
                target_chains = opposite_chains
                opposite_used = True
        
        if target_chains:
            
            d_perp_min = float("inf")
            for c in target_chains:
                pts = c.get("points") or []
                if len(pts) < 2:
                    if len(pts) == 1:
                        d_one = haversine(p["lat"], p["lon"], pts[0]["lat"], pts[0]["lon"])
                        if d_one < d_perp_min: d_perp_min = d_one
                    continue
                pts_list = [{"lat": pp["lat"], "lon": pp["lon"]} for pp in pts]
                d = dist_to_polyline(p["lat"], p["lon"], pts_list)
                if np.isfinite(d) and d < d_perp_min:
                    d_perp_min = d
            
            if not np.isfinite(d_perp_min):
                return None
            
            home = route.get("home", {}) or {}
            work = route.get("work", {}) or {}
            if home and work and home.get("lat") and work.get("lat"):
                if direction == "home_to_work":
                    O_lat, O_lon = home["lat"], home["lon"]
                    D_lat, D_lon = work["lat"], work["lon"]
                else:
                    O_lat, O_lon = work["lat"], work["lon"]
                    D_lat, D_lon = home["lat"], home["lon"]
                d_od = haversine(O_lat, O_lon, D_lat, D_lon)
                d_o_poi = haversine(O_lat, O_lon, p["lat"], p["lon"])
                d_poi_d = haversine(p["lat"], p["lon"], D_lat, D_lon)
                detour = max(0.0, (d_o_poi + d_poi_d) - d_od)
            else:
                detour = 0.0
            
            W = 1.0
            D_MAX = 2.0
            s_perp = math.exp(-d_perp_min / W)
            s_detour = math.exp(-detour / D_MAX)
            spa = float(s_perp * s_detour)
            if opposite_used:
                spa *= 0.4
            return spa
        
        if not route.get("points"): return None
        pts = [(rp["lat"], rp["lon"]) for rp in route["points"] if valid_latlon(rp.get("lat"), rp.get("lon"))]
        if len(pts) == 0: return None
        if len(pts) == 1:
            d = haversine(p["lat"], p["lon"], pts[0][0], pts[0][1])
            if not np.isfinite(d): return None
            R_POINT = 8.0
            return float(math.exp(-d / R_POINT))
        d_perp = dist_to_polyline(p["lat"], p["lon"], route["points"])
        if not np.isfinite(d_perp): return None
        O = pts[0]; D = pts[-1]
        d_od    = haversine(O[0], O[1], D[0], D[1])
        d_o_poi = haversine(O[0], O[1], p["lat"], p["lon"])
        d_poi_d = haversine(p["lat"], p["lon"], D[0], D[1])
        detour  = max(0.0, (d_o_poi + d_poi_d) - d_od)
        W = 1.0
        D_MAX = 2.0
        s_perp   = math.exp(-d_perp / W)
        s_detour = math.exp(-detour / D_MAX)
        return float(s_perp * s_detour)

    return None




import re as _re_tem
_HOUR_RANGE_RE = _re_tem.compile(r"(\d{1,2})(?::\d{2})?\s*[\u2013\u2014\u2010-\u2015~\-]\s*(\d{1,2})(?::\d{2})?")

_TEM_BUCKETS = {
    "breakfast":         (6, 10),
    "morning":           (6, 12),
    "weekday_morning":   (6, 10),
    "weekend_morning":   (6, 12),
    "lunch":             (11, 14),
    "weekday_lunch":     (11, 14),
    "afternoon":         (14, 18),
    "weekday_afternoon": (14, 18),
    "weekend_afternoon": (12, 18),
    "dinner":            (17, 21),
    "evening":           (18, 22),
    "weekday_evening":   (18, 23),
    "weekend_evening":   (18, 23),
    "late_night":        (22, 30),
    "night":             (20, 26),
}


def _hours_open_in_range(hours_str: str, lo: int, hi: int):
    if not hours_str:
        return None
    h_lc = str(hours_str).lower()
    if "24 hour" in h_lc or "24h" in h_lc or "24\u00a0hour" in h_lc:
        return True
    ranges = _HOUR_RANGE_RE.findall(h_lc)
    if not ranges:
        return None
    am_pm_aware = False
    has_open = False
    for o_s, c_s in ranges:
        try:
            o, c = int(o_s), int(c_s)
        except Exception:
            continue
        if o > 24 or c > 48:
            continue
        if c <= o:
            c += 24
        if o < hi and c > lo:
            has_open = True
            break
    return has_open


def s_temporal(p: Dict[str, Any], intent: dict | None) -> float | None:
    if not isinstance(intent, dict):
        return None
    tb = (intent.get("temporal", {}) or {}).get("time_bucket")
    if not tb:
        for c in intent.get("must_conditions", []) or []:
            if c.get("field") == "opening_hours" and c.get("value"):
                tb = str(c.get("value"))
                break
    if not tb:
        return None
    tb = str(tb).lower().strip()
    rng = _TEM_BUCKETS.get(tb)
    if rng is None:
        return None
    lo, hi = rng

    hours = p.get("opening_hours")
    if hours is None or str(hours).strip() == "" or str(hours).lower() == "nan":
        return None

    h_lc = str(hours).lower()
    if "24 hour" in h_lc or "24h" in h_lc:
        return 1.0

    is_open = _hours_open_in_range(h_lc, lo, hi)
    if is_open is None:
        return None
    return 1.0 if is_open else 0.0


def _embedding_for(p: Dict[str, Any], poi_emb_cache: dict):
    pid = str(p.get("poi_id") or p.get("id") or "")
    emb = poi_emb_cache.get(pid)
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32)
    if emb.size == 0 or np.linalg.norm(emb) < 1e-8:
        return None
    return emb


def s_semantic(p: Dict[str, Any], q_emb, poi_emb_cache: dict) -> float | None:
    p_emb = _embedding_for(p, poi_emb_cache)
    if p_emb is None:
        return None
    return cosine_01(q_emb, p_emb)



JAPAN_CHAIN_BRANDS = {
    '7-eleven': '7-Eleven', 'seven-eleven': '7-Eleven', 'セブン': '7-Eleven',
    'セブン-イレブン': '7-Eleven', 'セブンイレブン': '7-Eleven',
    'lawson': 'Lawson', 'ローソン': 'Lawson',
    'familymart': 'FamilyMart', 'family mart': 'FamilyMart',
    'ファミリーマート': 'FamilyMart',
    'ministop': 'Ministop', 'ミニストップ': 'Ministop',
    'mcdonald': 'McDonalds', 'マクドナルド': 'McDonalds',
    'mos burger': 'MosBurger', 'モスバーガー': 'MosBurger',
    'kfc': 'KFC', 'ケンタッキー': 'KFC',
    'sukiya': 'Sukiya', 'すき家': 'Sukiya',
    'yoshinoya': 'Yoshinoya', '吉野家': 'Yoshinoya',
    'matsuya': 'Matsuya', '松屋': 'Matsuya',
    'eneos': 'ENEOS', 'エネオス': 'ENEOS',
    'idemitsu': 'Idemitsu', '出光': 'Idemitsu',
    'shell': 'Shell',
    'cosmo': 'Cosmo', 'コスモ': 'Cosmo',
    'apollo': 'Apollo',
    'mobil': 'Mobil',
    'aeon': 'Aeon', 'イオン': 'Aeon',
    'tokyu store': 'TokyuStore', 'tokyu-store': 'TokyuStore', '東急ストア': 'TokyuStore',
    'sanwa': 'Sanwa', '三和': 'Sanwa',
    'welcia': 'Welcia', 'ウエルシア': 'Welcia',
    'tsuruha': 'Tsuruha', 'ツルハ': 'Tsuruha',
    'matsumoto': 'MatsumotoKiyoshi', 'マツモト': 'MatsumotoKiyoshi',
    'sundrug': 'SunDrug',
    'saizeriya': 'Saizeriya', 'サイゼリヤ': 'Saizeriya',
    'gusto': 'Gusto', 'ガスト': 'Gusto',
    'hama-sushi': 'HamaSushi', 'はま寿司': 'HamaSushi',
    'kura sushi': 'KuraSushi', 'くら寿司': 'KuraSushi',
    'sushiro': 'Sushiro', 'スシロー': 'Sushiro',
    'denny': 'Dennys',
    'starbucks': 'Starbucks', 'スタバ': 'Starbucks', 'スターバックス': 'Starbucks',
    'doutor': 'Doutor', 'ドトール': 'Doutor',
    'tully': 'Tullys',
    'apita': 'Apita', 'アピタ': 'Apita',
    'autobacs': 'Autobacs', 'オートバックス': 'Autobacs',
    'nissan': 'Nissan', '日産': 'Nissan',
    'toyota': 'Toyota', 'トヨタ': 'Toyota',
    'honda': 'Honda', 'ホンダ': 'Honda',
}


CHINA_CHAIN_BRANDS = {
    '瑞幸': 'Luckin', 'luckin': 'Luckin', '库迪': 'Cotti', 'cotti': 'Cotti',
    '星巴克': 'Starbucks', '麦当劳': 'McDonalds', '肯德基': 'KFC', '汉堡王': 'BurgerKing',
    '必胜客': 'PizzaHut', '德克士': 'Dicos', '华莱士': 'Wallace', '真功夫': 'Zkungfu',
    '永和': 'Yonghe', '沙县': 'Shaxian', '兰州拉面': 'Lanzhou', '兰州牛肉': 'Lanzhou',
    '蜜雪冰城': 'Mixue', '蜜雪': 'Mixue', '喜茶': 'Heytea', '奈雪': 'Naixue',
    '海底捞': 'Haidilao', '西贝': 'Xibei', '全家便利': 'FamilyMartCN', '罗森': 'Lawson',
    '便利蜂': 'Bianlifeng', '美宜佳': 'Meiyijia', '物美': 'Wumart', '华联': 'Hualian',
    '家乐福': 'Carrefour', '沃尔玛': 'Walmart', '屈臣氏': 'Watsons', '翰皇': 'Hanhuang',
    '中国银行': 'BOC', '工商银行': 'ICBC', '建设银行': 'CCB', '农业银行': 'ABC',
    '招商银行': 'CMB', '交通银行': 'BOCOM', '邮政储蓄': 'PSBC', '民生银行': 'CMBC',
    '中国邮政': 'ChinaPost', '中国石化': 'Sinopec', '中国石油': 'PetroChina',
    '丰巢': 'Fengchao', '顺丰': 'SF', '好邻居': 'Haolinju', '全时': 'Quanshi',
}

def extract_brand(name: str | None):
    if not name:
        return None
    n = str(name).lower()
    for k, v in JAPAN_CHAIN_BRANDS.items():
        if k in n:
            return v
    for k, v in CHINA_CHAIN_BRANDS.items():
        if k in n:
            return v
    return None


def s_preference_v2(poi: dict, visit_map: dict | None) -> float | None:
    if not visit_map:
        return None
    score = 0.0
    contributed = False

    brand = extract_brand(poi.get("name"))
    if brand:
        n_brand = (visit_map.get("by_brand") or {}).get(brand, 0)
        if n_brand >= 5:
            score += 0.35
            contributed = True
        elif n_brand >= 2:
            score += 0.20
            contributed = True
        elif n_brand >= 1:
            score += 0.10
            contributed = True

    cat = (poi.get("category") or "").lower()
    if cat:
        n_cat = (visit_map.get("by_category") or {}).get(cat, 0)
        total = max(1, visit_map.get("total_visits", 0))
        cat_pct = n_cat / total
        if n_cat >= 20 and cat_pct >= 0.05:
            score += 0.10
            contributed = True
        elif n_cat >= 5 and cat_pct >= 0.02:
            score += 0.05
            contributed = True

    if not contributed:
        return None
    return min(0.5, score)


def s_preference(p: Dict[str, Any], e_pref, poi_emb_cache: dict) -> float | None:
    if e_pref is None or np.linalg.norm(np.asarray(e_pref)) < 1e-8:
        return None
    p_emb = _embedding_for(p, poi_emb_cache)
    if p_emb is None:
        return None
    return cosine_01(e_pref, p_emb)


def renormalize_weights(s_q: str, available: set[str]) -> dict:
    base = BASE_WEIGHTS.get(s_q, BASE_WEIGHTS["none"])
    sub = {g: base[g] for g in available if g in base and base[g] > 0}
    total = sum(sub.values())
    if total < 1e-8:
        return {g: 1.0 / max(len(available), 1) for g in available}
    return {g: v / total for g, v in sub.items()}



def _category_match_bonus(p_cat: str, target_cat: str, alias_table: dict) -> float:
    if not p_cat or not target_cat:
        return 0.0
    p_lc = str(p_cat).lower().strip()
    t_lc = str(target_cat).lower().strip()
    if p_lc == t_lc:
        return 1.0
    aliases = {a.lower() for a in alias_table.get(t_lc, [])}
    if p_lc in aliases:
        return 0.5
    return 0.0


def _ewm_weights(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    if n < 2:
        return np.ones(M.shape[1]) / M.shape[1]
    N = M + 1e-12
    P = N / N.sum(axis=0, keepdims=True)
    E = -(P * np.log(P)).sum(axis=0) / np.log(n)
    D = np.clip(1.0 - E, 0.0, None)
    if D.sum() <= 1e-12:
        return np.ones(M.shape[1]) / M.shape[1]
    return D / D.sum()


def _minmax_cols(M: np.ndarray) -> np.ndarray:
    lo, hi = M.min(axis=0), M.max(axis=0)
    rng = hi - lo
    out = np.zeros_like(M)
    nz = rng > 1e-12
    out[:, nz] = (M[:, nz] - lo[nz]) / rng[nz]
    return out


def score_candidates(
    query: str,
    intent: dict,
    R_tau: dict,
    A_q: List[Dict[str, Any]],
    q_emb: np.ndarray,
    poi_emb_cache: dict,
    visit_map: dict | None = None,
) -> List[Dict[str, Any]]:
    import os as _os
    from constants import CATEGORY_ALIASES as _CA

    mode = _os.environ.get("TRAJRAG_FUSION", "ewm").lower()
    alpha = float(_os.environ.get("TRAJRAG_EWM_ALPHA", "0.5"))

    s_q = intent.get("spatial", {}).get("mode", "none")
    e_pref = R_tau.get("preference", {}).get("embedding") if R_tau.get("preference") else None
    target_cat = intent.get("target", {}).get("category")

    rows = []
    for p in A_q:
        sigs = {
            "spa":  s_spatial(p, R_tau, s_q),
            "tem":  s_temporal(p, intent),
            "sem":  s_semantic(p, q_emb, poi_emb_cache),
            "pref": (s_preference_v2(p, visit_map) if visit_map else s_preference(p, e_pref, poi_emb_cache)),
        }
        cat = _category_match_bonus(p.get("category"), target_cat, _CA)
        rows.append((p, sigs, cat))

    if not rows:
        return []

    if mode == "fixed":
        scored = []
        for p, sigs, cat in rows:
            available = {g for g, v in sigs.items() if v is not None}
            if not available:
                S, w = 0.0, {}
            else:
                w = renormalize_weights(s_q, available)
                S = sum(w.get(g, 0.0) * float(sigs[g]) for g in available if g in w)
            S += 0.35 * cat
            rec = dict(p)
            rec["_score"] = float(S)
            rec["_signals"] = {g: sigs[g] for g in available}
            rec["_weights"] = w
            rec["_cat_match"] = float(cat)
            scored.append(rec)
        scored.sort(key=lambda x: -x["_score"])
        return scored

    n = len(rows)

    def _col(name):
        vals = [sigs.get(name) for _, sigs, _ in rows]
        if all(v is None for v in vals):
            return None
        arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
        fill = np.nanmean(arr) if not np.all(np.isnan(arr)) else 0.0
        return np.where(np.isnan(arr), fill, arr)

    spa, tem, sem, pref = (_col(k) for k in ("spa", "tem", "sem", "pref"))
    cat = np.array([float(c) for _, _, c in rows], dtype=np.float64)

    if sem is None:
        sem_new = cat
    else:
        sem_new = alpha * sem + (1.0 - alpha) * cat

    cols = {"spa": spa, "tem": tem, "sem": sem_new, "pref": pref}
    avail = [k for k in ("spa", "tem", "sem", "pref") if cols[k] is not None]

    if not avail:
        scored = []
        for p, sigs, c in rows:
            rec = dict(p)
            rec["_score"] = 0.0
            rec["_signals"] = {}
            rec["_weights"] = {}
            rec["_cat_match"] = float(c)
            scored.append(rec)
        return scored

    M = _minmax_cols(np.column_stack([cols[k] for k in avail]))
    w_vec = _ewm_weights(M)
    S_all = M @ w_vec
    w_map = {k: float(v) for k, v in zip(avail, w_vec)}

    scored = []
    for idx, (p, sigs, c) in enumerate(rows):
        rec = dict(p)
        rec["_score"] = float(S_all[idx])
        rec["_signals"] = {g: sigs[g] for g in sigs if sigs[g] is not None}
        rec["_weights"] = w_map
        rec["_cat_match"] = float(c)
        rec["_sem_fused"] = float(sem_new[idx])
        scored.append(rec)

    scored.sort(key=lambda x: -x["_score"])
    return scored
