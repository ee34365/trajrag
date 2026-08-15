import json, math
from collections import Counter, defaultdict
from datetime import datetime

def parse_dt(s):
    if not s: return None
    try: return datetime.fromisoformat(str(s).replace("Z","+00:00").replace(" ","T"))
    except: return None

def haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    rlat1,rlon1,rlat2,rlon2 = map(math.radians, [lat1,lon1,lat2,lon2])
    a = math.sin((rlat2-rlat1)/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin((rlon2-rlon1)/2)**2
    return 2*R*math.asin(math.sqrt(a))

def build_profile(uid, traj, top_visits=10, top_types=10, top_brands=8):
    visits = [e for e in traj if e.get("status") == "visit"]
    acts   = [e for e in traj if e.get("status") == "activity"]

    poi_stats = defaultdict(lambda: {"count":0, "dwell_min":0.0, "hours":Counter(),
                                      "dows":Counter(), "name":None, "type":None,
                                      "lat":None, "lon":None, "first_dt":None, "last_dt":None})
    for v in visits:
        pid = v.get("pid") or v.get("poi_id")
        if not pid: continue
        s = parse_dt(v.get("start_time")); e = parse_dt(v.get("end_time"))
        dwell = (e - s).total_seconds() / 60.0 if (s and e) else 0
        ps = poi_stats[pid]
        ps["count"] += 1
        ps["dwell_min"] += max(0, dwell)
        if s:
            ps["hours"][s.hour] += 1
            ps["dows"][s.weekday()] += 1
            if ps["first_dt"] is None or s < ps["first_dt"]: ps["first_dt"] = s
            if ps["last_dt"]  is None or s > ps["last_dt"]:  ps["last_dt"]  = s
        if v.get("poi_name"): ps["name"] = v["poi_name"]
        if v.get("poi_type"): ps["type"] = v["poi_type"]
        if v.get("d_lat"):    ps["lat"]  = v["d_lat"]
        if v.get("d_lon"):    ps["lon"]  = v["d_lon"]

    hour_hist = Counter()
    dow_hist = Counter()
    for v in visits:
        s = parse_dt(v.get("start_time"))
        if s:
            hour_hist[s.hour] += 1
            dow_hist[s.weekday()] += 1

    sorted_pois = sorted(poi_stats.items(), key=lambda x: -x[1]["count"])
    def hour_bucket_pct(hs):
        total = sum(hs.values()) or 1
        return {
            "0-6":  round(100*sum(hs.get(h,0) for h in range(0,6))/total),
            "6-12": round(100*sum(hs.get(h,0) for h in range(6,12))/total),
            "12-18":round(100*sum(hs.get(h,0) for h in range(12,18))/total),
            "18-24":round(100*sum(hs.get(h,0) for h in range(18,24))/total),
        }
    def weekday_pct(ds):
        total = sum(ds.values()) or 1
        wd = sum(ds.get(d,0) for d in range(0,5))
        return round(100*wd/total)

    top_pois_list = []
    for pid, ps in sorted_pois[:top_visits]:
        top_pois_list.append({
            "poi_id": pid, "name": ps["name"] or "", "type": ps["type"] or "",
            "lat": round(ps["lat"],5) if ps["lat"] else None,
            "lon": round(ps["lon"],5) if ps["lon"] else None,
            "visits": ps["count"],
            "total_dwell_h": round(ps["dwell_min"]/60, 1),
            "median_dwell_min": round(ps["dwell_min"]/max(1,ps["count"])),
            "hour_pct": hour_bucket_pct(ps["hours"]),
            "weekday_pct": weekday_pct(ps["dows"]),
            "span_days": (ps["last_dt"] - ps["first_dt"]).days if (ps["first_dt"] and ps["last_dt"]) else 0,
        })

    type_counter = Counter()
    type_dwell = defaultdict(float)
    type_hours = defaultdict(Counter)
    type_examples = defaultdict(Counter)
    for pid, ps in poi_stats.items():
        if not ps["type"]: continue
        type_counter[ps["type"]] += ps["count"]
        type_dwell[ps["type"]] += ps["dwell_min"]
        type_hours[ps["type"]].update(ps["hours"])
        if ps["name"]:
            type_examples[ps["type"]][ps["name"]] += ps["count"]

    type_list = []
    for t, n in type_counter.most_common(top_types):
        peak_hour = type_hours[t].most_common(1)
        type_list.append({
            "type": t, "n": n,
            "median_dwell_min": round(type_dwell[t]/max(1,n)),
            "peak_hour": peak_hour[0][0] if peak_hour else None,
            "examples": [name for name,_ in type_examples[t].most_common(3)],
        })

    brand_visits = Counter()
    brand_stores = defaultdict(set)
    for pid, ps in poi_stats.items():
        if not ps["name"]: continue
        brand = ps["name"].split(";")[0].split(" ")[0]
        brand_visits[brand] += ps["count"]
        brand_stores[brand].add(pid)
    brand_list = []
    for b, n in brand_visits.most_common(top_brands*3):
        if n < 3: continue
        brand_list.append({"brand": b, "visits": n, "stores": len(brand_stores[b])})
        if len(brand_list) >= top_brands: break

    mode_counter = Counter(a.get("activity_type") for a in acts if a.get("activity_type"))
    mode_list = [{"mode":m,"n":n,"pct":round(100*n/max(1,sum(mode_counter.values())))}
                 for m,n in mode_counter.most_common(5)]

    dts = [parse_dt(v.get("start_time")) for v in visits if v.get("start_time")]
    dts = [d for d in dts if d]
    if dts:
        date_first = min(dts).date().isoformat()
        date_last = max(dts).date().isoformat()
        n_days = (max(dts) - min(dts)).days + 1
    else:
        date_first = date_last = None; n_days = 0

    lats = [p["lat"] for p in poi_stats.values() if p["lat"]]
    lons = [p["lon"] for p in poi_stats.values() if p["lon"]]
    if lats and lons:
        extent = {
            "lat_range":[round(min(lats),4), round(max(lats),4)],
            "lon_range":[round(min(lons),4), round(max(lons),4)],
            "diag_km": round(haversine(min(lats),min(lons),max(lats),max(lons)),1),
        }
    else:
        extent = {}

    return {
        "uid": uid,
        "meta": {
            "n_visits": len(visits), "n_activities": len(acts),
            "n_unique_pois": len(poi_stats),
            "date_first": date_first, "date_last": date_last, "n_days": n_days,
        },
        "hour_histogram": {str(h):hour_hist[h] for h in range(24)},
        "dow_histogram": {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d]:dow_hist[d] for d in range(7)},
        "top_pois": top_pois_list,
        "type_distribution": type_list,
        "brand_loyalty": brand_list,
        "activity_modes": mode_list,
        "extent": extent,
    }


if __name__ == "__main__":
    import sys
    recs = [json.loads(l) for l in open("./data/takeout/trajrag_kept_enriched.jsonl")]
    r = recs[0]
    profile = build_profile(r["uid"], r["traj"])
    s = json.dumps(profile, ensure_ascii=False, indent=2)
    print(s)
    print(f"\n=== Profile size: {len(s)} chars, ~{len(s)//4} tokens ===")


def build_visit_map(traj):
    from collections import Counter
    try:
        from scoring import extract_brand
    except Exception:
        def extract_brand(n): return None

    by_brand = Counter()
    by_category = Counter()
    by_poi_id = Counter()
    total = 0
    for t in traj or []:
        if t.get("status") != "visit":
            continue
        total += 1
        pid = t.get("pid") or t.get("poi_id")
        if pid:
            by_poi_id[pid] += 1
        cat = (t.get("poi_type") or "").lower()
        if cat:
            by_category[cat] += 1
        name = t.get("poi_name") or ""
        b = extract_brand(name)
        if b:
            by_brand[b] += 1
    return {
        "by_brand":    dict(by_brand),
        "by_category": dict(by_category),
        "by_poi_id":   dict(by_poi_id),
        "total_visits": total,
    }
