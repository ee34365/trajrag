import sys
sys.path.insert(0, "/tmp")
from profile import build_profile
import json

def compact(profile):
    p = profile
    lines = []
    lines.append(f"# User Profile (uid={p['uid']})")
    m = p["meta"]
    lines.append(f"## Meta: visits={m['n_visits']} activities={m['n_activities']} unique_pois={m['n_unique_pois']} period={m['date_first']}→{m['date_last']} ({m['n_days']}d)")
    lines.append(f"## Spatial extent: lat={p['extent'].get('lat_range')} lon={p['extent'].get('lon_range')} diag={p['extent'].get('diag_km')}km")

    lines.append("\n## Hour histogram (visits by start_hour 0-23):")
    hh = p["hour_histogram"]
    lines.append("  " + " ".join(f"h{h}:{hh[str(h)]}" for h in range(24)))

    lines.append("\n## Day-of-week histogram:")
    lines.append("  " + " ".join(f"{d}:{p['dow_histogram'][d]}" for d in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]))

    lines.append(f"\n## Top {len(p['top_pois'])} POIs by visit count (lat,lon,visits,dwell_h,med_dwell_min,hour_pct,wkday%,span_d):")
    for i, x in enumerate(p["top_pois"],1):
        h = x["hour_pct"]
        lines.append(f"  P{i}. id={x['poi_id'][-10:]} \"{x['name']}\" type={x['type']!r} ({x['lat']},{x['lon']}) "
                     f"visits={x['visits']} dwell_h={x['total_dwell_h']} med_min={x['median_dwell_min']} "
                     f"h[0-6,6-12,12-18,18-24]=[{h['0-6']},{h['6-12']},{h['12-18']},{h['18-24']}] wkday={x['weekday_pct']}% span={x['span_days']}d")

    lines.append(f"\n## Type distribution (visits, dwell, peak hour, examples):")
    for x in p["type_distribution"]:
        lines.append(f"  - {x['type']}: n={x['n']} med_dwell={x['median_dwell_min']}min peak_h={x['peak_hour']} top=[{', '.join(x['examples'])}]")

    lines.append(f"\n## Brand loyalty (visits / n_stores):")
    for b in p["brand_loyalty"]:
        lines.append(f"  - {b['brand']}: {b['visits']}v / {b['stores']}stores")

    lines.append(f"\n## Activity modes:")
    lines.append("  " + " ".join(f"{m['mode']}:{m['pct']}%" for m in p["activity_modes"]))

    return "\n".join(lines)


if __name__ == "__main__":
    recs = [json.loads(l) for l in open("./data/takeout/trajrag_kept_enriched.jsonl")]
    r = recs[0]
    profile = build_profile(r["uid"], r["traj"])
    text = compact(profile)
    print(text)
    print(f"\n=== Compact: {len(text)} chars, ~{len(text)//4} tokens ===")
