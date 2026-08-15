from __future__ import annotations
import re
import datetime as dt
import numpy as np
import pandas as pd
import duckdb
from openai import OpenAI
from constants import OPENAI_KEY, OPENAI_BASE_URL, LLM_MODEL, EMBED_MODEL, USE_LOCAL_EMBED, LOCAL_EMBED_MODEL, EMBED_DIM, PREF_TOP_N, PREF_MIN_SUPPORT, CATEGORY_ALIASES, EMBED_DIM
from utils import haversine, haversine_vec, sample_segment, valid_latlon
from parser import _chat_kwargs  

_ACTIVE_DATASET = "japan"

def set_active_dataset(name: str):
    global _ACTIVE_DATASET
    _ACTIVE_DATASET = name.lower() if name else "japan"

def get_active_dataset() -> str:
    return _ACTIVE_DATASET

_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()


DB_SCHEMA = """
TABLE trajectory (
  uid VARCHAR,
  event_start TIMESTAMP,
  event_end TIMESTAMP,
  poi_id VARCHAR,
  poi_name VARCHAR,
  poi_type VARCHAR,
  lat DOUBLE,        -- visit destination latitude
  lon DOUBLE,        -- visit destination longitude
  o_lat DOUBLE,      -- activity origin latitude
  o_lon DOUBLE,
  d_lat DOUBLE,      -- activity destination latitude
  d_lon DOUBLE,
  status VARCHAR     -- 'visit' or 'activity'
)
"""

DUCKDB_SYNTAX_HINTS = """
DuckDB syntax notes:
- Use EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) for hour.
- Use EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) for day-of-week.
- dow: 0=Sunday, 1=Monday, ..., 6=Saturday.
- Use single quotes for string literals.
- Use NULL not None.
- For "most frequent location" queries: GROUP BY poi_id and ORDER BY COUNT(*) DESC LIMIT 1.
- Do NOT use AVG(lat) or AVG(lon) for anchor detection — return the actual top POI row.
- HAVING works for aggregate filters.
"""



def _parse_extent_center(profile_text: str):
    import re as _re
    try:
        m_p1 = _re.search('P1\\. id=\\S+ "[^"]*" type=[^()]*\\(([0-9.\\-]+),([0-9.\\-]+)\\)', profile_text)
        if m_p1:
            return float(m_p1.group(1)), float(m_p1.group(2))
    except Exception:
        pass
    try:
        m_lat = _re.search(r"lat=\[([0-9.\-]+), ?([0-9.\-]+)\]", profile_text)
        m_lon = _re.search(r"lon=\[([0-9.\-]+), ?([0-9.\-]+)\]", profile_text)
        if m_lat and m_lon:
            return (float(m_lat.group(1)) + float(m_lat.group(2))) / 2.0, \
                   (float(m_lon.group(1)) + float(m_lon.group(2))) / 2.0
    except Exception:
        pass
    return 35.0, 138.0

def retrieve_evidence(intent: dict, uid: str, conn: duckdb.DuckDBPyConnection,
                       profile_text: str = "") -> dict:
    R_tau = {"point": None, "route": None, "zone": None, "preference": None}

    for r in intent.get("evidence_requirements", []):
        etype = r.get("type")
        args = r.get("arguments", {}) or {}
        if etype == "point":
            R_tau["point"] = retrieve_point(uid, args, conn, profile_text)
        elif etype == "route":
            R_tau["route"] = retrieve_route(uid, args, conn, profile_text)
        elif etype == "zone":
            R_tau["zone"] = retrieve_zone(uid, args, conn, profile_text)
        elif etype == "preference":
            R_tau["preference"] = retrieve_preference(uid, args, conn, profile_text)

    return R_tau


ADMIN_SCHEMA = """TABLE admin_regions(
  osm_id BIGINT, name VARCHAR, name_en VARCHAR, name_en_norm VARCHAR, name_ja VARCHAR,
  admin_level INT,  -- 4=都道府県, 7=市町村, 8=政令市内区 (Ward)
  iso3166_2 VARCHAR,
  lat_min, lat_max, lon_min, lon_max DOUBLE,
  cen_lat, cen_lon DOUBLE,
  polygon_wkt VARCHAR
)
-- name_en_norm: lowercased, accent-stripped name_en for matching"""


def _llm_choose_source(goal: str, uid: str, profile_text: str = "") -> dict:
    sys_msg = f"""You are a retrieval planner. You have:
  (A) USER_PROFILE — pre-computed mobility statistics (PREFERRED — use whenever possible)
  (B) SQL on tables trajectory + admin_regions (FALLBACK — only when profile cannot answer)

★★★ CRITICAL: For HOME and WORK detection goals, you MUST use USER_PROFILE. NEVER use SQL for these.
The profile's Top-10 POIs with hour-bucket histogram h[0-6,6-12,12-18,18-24] and wkday% are SUFFICIENT.
Goals containing "home", "night hours", "work", "work hours", "workplace", "residence"
are NOT recency goals — they are aggregation goals that the profile already answered offline.

▼ EXAMPLE 1 — HOME goal (use profile):
  Goal: "Find user X's home, during night hours..."
  Profile Top-10 has: P1 "X-residence" type=premise visits=400 h[0-6,6-12,12-18,18-24]=[30,5,15,50] wkday=70%
  → DECISION: source=profile, lat/lon=P1's coords, anchor_name="X-residence", poi_id=P1's id
  → reasoning="Rule 1: P1 is premise with highest h[18-24]+h[0-6]=80%"

▼ EXAMPLE 2 — WORK goal with EXCLUDE home (use profile):
  Goal: "Find user X's work, EXCLUDE poi_id=P1_id..."
  Profile Top-10: P2 "Big Corp" type=corporate_office visits=120 h[0-6,6-12,12-18,18-24]=[0,55,40,5] wkday=95%
  → DECISION: source=profile, lat/lon=P2's coords, anchor_name="Big Corp", poi_id=P2's id
  → reasoning="Rule 2: P2 has highest h[6-12]+h[12-18]=95%, wkday=95%, type=corporate_office"

▼ EXAMPLE 3 — "most recent visit" (use SQL):
  Goal: "Get the most recent visit of user X" → source=sql (profile has no timestamp).

★ STRONG PREFERENCE: Use USER_PROFILE whenever a heuristic on the Top-10 POIs
  can answer the goal. The profile already contains hour buckets, weekday%, dwell time,
  and POI type — these are SUFFICIENT for home/work/preference goals. DO NOT fall back
  to SQL just because the goal says "hour >= 21"; the 4-bucket histogram h[0-6,6-12,12-18,18-24]
  IS the same information at coarser granularity.

DECISION RULES (apply in order):

▼ Rule 1: HOME detection goal (keywords: "home", "night hours", "residence")
  Scan Top-10 POIs. Pick the POI with HIGHEST h[18-24]+h[0-6] AND type matching one of:
  {{premise, condominium_complex, apartment_building, house, residential, lodging, housing_complex, apartment_complex}}
  If such POI exists with visits ≥ 10% of total: source="profile", lat/lon from that POI.
  ★ If no "residential" type POI exists, fall back to the POI with the highest
  h[18-24]+h[0-6] regardless of type (still use PROFILE).
  Only use SQL if Top-10 is empty or no POI has h[18-24]+h[0-6] > 30%.

▼ Rule 2: WORK detection goal (keywords: "work", "work hours", "workplace", "office")
  Scan Top-10 POIs, EXCLUDE the home POI from Rule 1.
  Pick the POI with HIGHEST h[6-12]+h[12-18] AND wkday% ≥ 60% AND type matching one of:
  {{corporate_office, primary_school, school, university, government_office, hospital,
    premise, store, restaurant, business, place_of_worship, community_center, gym, factory,
    industrial_park, post_office}}
  If such POI exists: source="profile", lat/lon from that POI.
  ★ CRITICAL: home_poi_id and work_poi_id MUST BE DIFFERENT. If the highest h[6-12]+h[12-18]
  POI is the same as home, pick the SECOND one. Never return home==work.
  Only use SQL if all of Top-10 are clearly residential or no work-time POI exists.

▼ Rule 3: "current" / "most recent visit" goal
  Profile cannot answer recency (only frequency). Use SQL.

▼ Rule 4: Preference / category-history goals ("top visited of type X")
  Look at profile's Type distribution section. If type X is listed with examples, use profile examples
  (return ANY one of the example POI names — lat/lon may be unknown, that is OK, use 0,0 then).
  If type X NOT listed in profile's Type distribution, the user never visited that type:
  source="sql" but the SQL is expected to return 0 rows (still try).

USER_PROFILE:
{profile_text}

TRAJECTORY SCHEMA:
{DB_SCHEMA}

ADMIN_REGIONS SCHEMA:
{ADMIN_SCHEMA}

{DUCKDB_SYNTAX_HINTS}

OUTPUT (strict JSON):
- source="profile": {{"source":"profile", "lat":<num>, "lon":<num>, "anchor_name":<str>, "poi_id":<str>, "reasoning":"<which rule + which POI>"}}
- source="sql": {{"source":"sql", "sql":"<duckdb sql>", "reasoning":"<why SQL needed>"}}

Rules:
- Use uid = '{uid}' in any trajectory SQL WHERE clause.
- Output ONLY a single JSON object, no markdown, no explanation prose."""
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=500,
            timeout=25,
            response_format={"type": "json_object"},
            **_chat_kwargs(LLM_MODEL),
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": goal},
            ],
        )
        import json as _json
        text = resp.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
        return _json.loads(text)
    except Exception as e:
        print(f"  [Text2SQL] LLM failed: {str(e)[:80]}")
        return {"source": "sql", "sql": "", "reasoning": "llm_failed"}


def _llm_generate_sql(goal: str, uid: str, profile_text: str = "") -> str:
    plan = _llm_choose_source(goal, uid, profile_text)
    if plan.get("source") == "sql":
        return plan.get("sql", "") or ""
    return ""


def _run_sql_with_fallback(primary_sql: str, fallback_sql: str, conn) -> pd.DataFrame:
    for sql, tag in [(primary_sql, "LLM"), (fallback_sql, "fallback")]:
        if not sql:
            continue
        try:
            df = conn.execute(sql).fetchdf()
            if len(df) > 0:
                return df
        except Exception as e:
            if tag == "fallback":
                print(f"  [SQL] fallback failed: {str(e)[:80]}")
    return pd.DataFrame()



def _parse_top_pois_from_profile_text(profile_text: str):
    import re as _re
    out = []
    pattern = _re.compile(
        'P\\d+\\.\\s+id=(\\S+)\\s+"([^"]*)"\\s+type=\\\'?([^)\\\\\']*?)\\\'?\\s+\\(([0-9.\\-]+),([0-9.\\-]+)\\)\\s+visits=(\\d+).*?h\\[0-6,6-12,12-18,18-24\\]=\\[(\\d+),(\\d+),(\\d+),(\\d+)\\]\\s+wkday=(\\d+)%'
    )
    for m in pattern.finditer(profile_text or ""):
        try:
            out.append({
                "id_suffix": m.group(1),
                "name": m.group(2),
                "type": m.group(3).strip(),
                "lat": float(m.group(4)),
                "lon": float(m.group(5)),
                "visits": int(m.group(6)),
                "h": [int(m.group(7)), int(m.group(8)), int(m.group(9)), int(m.group(10))],
                "wkday": int(m.group(11)),
            })
        except Exception:
            continue
    return out


_HOME_TYPES = {"premise", "condominium_complex", "apartment_building", "house",
               "residential", "lodging", "housing_complex", "apartment_complex"}
_WORK_TYPES = {"corporate_office", "primary_school", "school", "university",
               "government_office", "hospital", "premise", "store", "office",
               "business", "place_of_worship", "community_center", "gym",
               "industrial_park", "post_office", "library", "shopping_mall",
               "restaurant", "factory", "city_hall", "local_government_office"}


def _profile_pick_home(profile_text: str) -> dict | None:
    pois = _parse_top_pois_from_profile_text(profile_text)
    if not pois:
        return None
    residential = [p for p in pois if p["type"].lower() in _HOME_TYPES]
    pool = residential if residential else pois
    if not pool:
        return None
    pool_sorted = sorted(pool, key=lambda p: -(p["h"][0] + p["h"][3]))
    cand = pool_sorted[0]
    if cand["h"][0] + cand["h"][3] < 30:
        return None
    if not valid_latlon(cand["lat"], cand["lon"]):
        return None
    return {
        "anchor_name": cand["name"] or "home",
        "poi_id": cand["id_suffix"],
        "lat": float(cand["lat"]),
        "lon": float(cand["lon"]),
        "source": "profile_home",
        "_profile_score": cand["h"][0] + cand["h"][3],
    }


def _profile_pick_work(profile_text: str, exclude_poi_id: str = "") -> dict | None:
    pois = _parse_top_pois_from_profile_text(profile_text)
    if not pois:
        return None
    excl_suffix = exclude_poi_id or ""
    cands = []
    for p in pois:
        if excl_suffix:
            if p["id_suffix"] == excl_suffix or excl_suffix.endswith(p["id_suffix"]) or p["id_suffix"].endswith(excl_suffix):
                continue
        if p["wkday"] < 50:
            continue
        if p["type"].lower() not in _WORK_TYPES:
            continue
        work_score = p["h"][1] + p["h"][2]
        if work_score < 30:
            continue
        cands.append((work_score, p))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    _, cand = cands[0]
    if not valid_latlon(cand["lat"], cand["lon"]):
        return None
    return {
        "anchor_name": cand["name"] or "work",
        "poi_id": cand["id_suffix"],
        "lat": float(cand["lat"]),
        "lon": float(cand["lon"]),
        "source": "profile_work",
        "_profile_score": cand["h"][1] + cand["h"][2],
    }


def retrieve_point(uid: str, args: dict, conn, profile_text: str = "") -> dict | None:
    anchor = args.get("anchor", "current")
    exclude_poi_id = args.get("exclude_poi_id")
    skip = int(args.get("skip", 0))

    if skip == 0 and profile_text:
        if anchor == "home":
            home_pick = _profile_pick_home(profile_text)
            if home_pick is not None:
                print(f"[retrieve_point] PROFILE shortcut HOME for uid={uid}: {home_pick["anchor_name"]} score={home_pick.get("_profile_score")}")
                home_pick.pop("_profile_score", None)
                return home_pick
        elif anchor == "work":
            work_pick = _profile_pick_work(profile_text, exclude_poi_id or "")
            if work_pick is not None:
                print(f"[retrieve_point] PROFILE shortcut WORK for uid={uid}: {work_pick["anchor_name"]} score={work_pick.get("_profile_score")}")
                work_pick.pop("_profile_score", None)
                return work_pick

    excl_clause = f" AND poi_id != '{exclude_poi_id}'" if exclude_poi_id else ""

    if anchor == "home":
        goal = (
            f"Find the SINGLE most frequently visited POI of user '{uid}' as their home, "
            "during night hours (event_start hour >= 21 or < 7) on weekdays (dow BETWEEN 1 AND 5). "
            "GROUP BY poi_id and ORDER BY COUNT(*) DESC. Return poi_id, poi_name, d_lat AS lat, d_lon AS lon. LIMIT 1."
        )
        fallback = f"""
            SELECT poi_id, poi_name, d_lat AS lat, d_lon AS lon, COUNT(*) AS freq
            FROM trajectory
            WHERE uid = '{uid}' AND status = 'visit' AND d_lat IS NOT NULL AND d_lon IS NOT NULL
              AND poi_id IS NOT NULL AND poi_id <> ''
              AND (EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) >= 21
                   OR EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) < 7)
              AND EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5
              {excl_clause}
            GROUP BY poi_id, poi_name, d_lat, d_lon
            HAVING COUNT(*) >= 2
            ORDER BY freq DESC
            LIMIT 1 OFFSET {skip}
        """
    elif anchor == "work":
        excl_hint = f" EXCLUDE the home POI (poi_id='{exclude_poi_id}')." if exclude_poi_id else ""
        goal = (
            f"Find the SINGLE most frequently visited POI of user '{uid}' as their WORK location, "
            "during work hours (event_start hour BETWEEN 9 AND 18) on weekdays (dow BETWEEN 1 AND 5)."
            f"{excl_hint}"
            " home_poi_id and work_poi_id MUST BE DIFFERENT."
            " GROUP BY poi_id and ORDER BY COUNT(*) DESC. Return poi_id, poi_name, d_lat AS lat, d_lon AS lon. LIMIT 1."
        )
        fallback = f"""
            SELECT poi_id, poi_name, d_lat AS lat, d_lon AS lon, COUNT(*) AS freq
            FROM trajectory
            WHERE uid = '{uid}' AND status = 'visit' AND d_lat IS NOT NULL AND d_lon IS NOT NULL
              AND poi_id IS NOT NULL AND poi_id <> ''
              AND EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 9 AND 18
              AND EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5
              {excl_clause}
            GROUP BY poi_id, poi_name, d_lat, d_lon
            HAVING COUNT(*) >= 2
            ORDER BY freq DESC
            LIMIT 1 OFFSET {skip}
        """
    else:
        goal = (
            f"Get the DESTINATION coordinates of user '{uid}'s single MOST RECENT "
            "trajectory record, REGARDLESS OF STATUS. Both status='activity' rows "
            "(travel-in-progress; d_lat/d_lon = travel destination) and "
            "status='visit' rows (arrived; d_lat/d_lon = current location) are "
            "valid; take whichever has the latest event_start timestamp. "
            "Return poi_id, poi_name, d_lat AS lat, d_lon AS lon. "
            "SQL: ORDER BY event_start DESC LIMIT 1, WITHOUT any status filter."
        )
        fallback = f"""
            SELECT poi_id, poi_name, d_lat AS lat, d_lon AS lon
            FROM trajectory
            WHERE uid = '{uid}' AND d_lat IS NOT NULL AND d_lon IS NOT NULL
            ORDER BY TRY_CAST(event_start AS TIMESTAMP) DESC
            LIMIT 1
        """

    plan = _llm_choose_source(goal, uid, profile_text)
    if plan.get("source") == "profile" and plan.get("lat") is not None and plan.get("lon") is not None:
        if valid_latlon(plan["lat"], plan["lon"]):
            return {
                "anchor_name": str(plan.get("anchor_name", anchor) or anchor),
                "poi_id": str(plan.get("poi_id", "") or ""),
                "lat": float(plan["lat"]),
                "lon": float(plan["lon"]),
                "source": f"profile_{anchor}",
            }
    primary = plan.get("sql", "") if plan.get("source") == "sql" else ""
    df = _run_sql_with_fallback(primary, fallback, conn)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    _lat = row.get("lat") if "lat" in df.columns else row.get("d_lat")
    _lon = row.get("lon") if "lon" in df.columns else row.get("d_lon")
    if not valid_latlon(_lat, _lon):
        return None
    return {
        "anchor_name": str(row.get("poi_name", anchor) or anchor),
        "poi_id": str(row.get("poi_id", "") or ""),
        "lat": float(_lat),
        "lon": float(_lon),
        "source": f"text2sql_{anchor}",
    }


_RAW_TRAJ_CACHE = {}

_GEOM_CAP = {}
def _cap_geom(**kw):
    try: _GEOM_CAP.update(kw)
    except Exception: pass


def _extract_real_commute_chains(traj_list, home_id, work_id, home_lat, home_lon, work_lat, work_lon):
    EXCL_KM = 0.1
    
    def is_at(lat, lon, tgt_lat, tgt_lon):
        try:
            return haversine(lat, lon, tgt_lat, tgt_lon) < EXCL_KM
        except Exception:
            return False
    
    records = []
    for i, r in enumerate(traj_list):
        if r.get("status") != "visit":
            continue
        pid = r.get("pid") or r.get("poi_id") or ""
        dl, dd = r.get("d_lat"), r.get("d_lon")
        ts = r.get("start_time", "")
        kind = None
        if home_id and (pid == home_id or (home_id and pid.endswith(home_id))):
            kind = "home"
        elif work_id and (pid == work_id or (work_id and pid.endswith(work_id))):
            kind = "work"
        elif dl is not None and dd is not None:
            if is_at(dl, dd, home_lat, home_lon):
                kind = "home"
            elif is_at(dl, dd, work_lat, work_lon):
                kind = "work"
        if kind:
            records.append({"idx": i, "kind": kind, "ts": ts, "lat": dl, "lon": dd, "poi_id": pid})
    
    if len(records) < 2:
        return []
    
    chains = []
    for k in range(len(records) - 1):
        start = records[k]
        end = records[k + 1]
        if start["kind"] == end["kind"]:
            continue
        direction = f"{start['kind']}_to_{end['kind']}"
        
        chain_points = [{"lat": start["lat"], "lon": start["lon"], "kind": "endpoint", 
                        "ts": str(start["ts"])}]
        n_gwp = 0
        for j in range(start["idx"] + 1, end["idx"]):
            rr = traj_list[j]
            if rr.get("status") == "visit":
                lat, lon = rr.get("d_lat"), rr.get("d_lon")
                if lat is not None and lon is not None:
                    chain_points.append({"lat": float(lat), "lon": float(lon), 
                                        "kind": "visit", "ts": str(rr.get("start_time", "")),
                                        "poi_name": rr.get("poi_name", "")})
            elif rr.get("status") == "activity":
                wps = rr.get("waypoints") or []
                if wps:
                    for wp in wps:
                        chain_points.append({"lat": float(wp["lat"]), "lon": float(wp["lon"]),
                                            "kind": "gwp", "ts": str(rr.get("start_time", ""))})
                        n_gwp += 1
                else:
                    ol, oo = rr.get("o_lat"), rr.get("o_lon")
                    dl, dd = rr.get("d_lat"), rr.get("d_lon")
                    if ol is not None: chain_points.append({"lat": float(ol), "lon": float(oo), "kind": "act_o"})
                    if dl is not None: chain_points.append({"lat": float(dl), "lon": float(dd), "kind": "act_d"})
        
        chain_points.append({"lat": end["lat"], "lon": end["lon"], "kind": "endpoint",
                            "ts": str(end["ts"])})
        
        chains.append({
            "direction": direction,
            "n_points": len(chain_points),
            "n_google_waypoints": n_gwp,
            "points": chain_points,
        })
    
    return chains



def retrieve_route(uid: str, args: dict, conn, profile_text: str = "") -> dict | None:
    direction = args.get("direction", "home_to_work")
    if direction in {"to_work", "home_to_work_route"}:
        direction = "home_to_work"
    if direction in {"home", "work_to_home_route"}:
        direction = "work_to_home"

    _rows = conn.execute(
        "SELECT poi_id, ANY_VALUE(poi_name) AS poi_name, AVG(COALESCE(d_lat,o_lat)) AS lat, AVG(COALESCE(d_lon,o_lon)) AS lon, COUNT(*) AS freq "
        "FROM trajectory WHERE uid = '" + str(uid) + "' AND status = 'visit' "
        "AND poi_id IS NOT NULL AND poi_id <> '' "
        "GROUP BY poi_id HAVING COUNT(*) >= 2 "
        "ORDER BY freq DESC LIMIT 25"
    ).fetchdf()
    if len(_rows) < 2:
        return None
    _A = _rows.iloc[0]
    home = {"poi_id": _A["poi_id"], "poi_name": _A["poi_name"], "lat": float(_A["lat"]), "lon": float(_A["lon"])}
    work = None
    for _i in range(1, len(_rows)):
        _r = _rows.iloc[_i]
        if haversine(home["lat"], home["lon"], float(_r["lat"]), float(_r["lon"])) > 1.0:
            work = {"poi_id": _r["poi_id"], "poi_name": _r["poi_name"], "lat": float(_r["lat"]), "lon": float(_r["lon"])}
            break
    if work is None:
        return None

    raw_traj = _RAW_TRAJ_CACHE.get(str(uid))
    chains_to_work, chains_to_home = [], []
    if raw_traj:
        all_chains = _extract_real_commute_chains(
            raw_traj,
            home.get("poi_id", ""), work.get("poi_id", ""),
            home["lat"], home["lon"], work["lat"], work["lon"],
        )
        chains_to_work = [c for c in all_chains if c["direction"] == "home_to_work"]
        chains_to_home = [c for c in all_chains if c["direction"] == "work_to_home"]
        print(f"[retrieve_route v2] uid={uid} chains_to_work={len(chains_to_work)} chains_to_home={len(chains_to_home)}")
    
    if direction == "home_to_work":
        dir_chains = chains_to_work
    else:
        dir_chains = chains_to_home
    
    if dir_chains:
        flat_points = []
        for c in dir_chains:
            for p in c["points"]:
                flat_points.append((p["lat"], p["lon"], p.get("ts")))
        return {
            "route_name": direction,
            "points": [{"lat": float(lat), "lon": float(lon), "timestamp": str(ts) if ts is not None else None}
                       for lat, lon, ts in _dedupe_points(flat_points)],
            "chains_to_work": chains_to_work,
            "chains_to_home": chains_to_home,
            "source": "real_commute_chains_v2",
            "home": home, "work": work,
        }
    elif chains_to_work or chains_to_home:
        opp_chains = chains_to_work if chains_to_work else chains_to_home
        flat_points = []
        for c in opp_chains:
            for p in c["points"]:
                flat_points.append((p["lat"], p["lon"], p.get("ts")))
        return {
            "route_name": direction,
            "points": [{"lat": float(lat), "lon": float(lon), "timestamp": str(ts) if ts is not None else None}
                       for lat, lon, ts in _dedupe_points(flat_points)],
            "chains_to_work": chains_to_work,
            "chains_to_home": chains_to_home,
            "source": "real_commute_chains_opposite_only",
            "home": home, "work": work,
        }
    
    traj_df = _load_traj_records(uid, conn)
    points = _extract_commute_waypoints(traj_df, home, work, direction)

    if len(points) < 2:
        if direction == "home_to_work":
            points = [(home["lat"], home["lon"], None), (work["lat"], work["lon"], None)]
        else:
            points = [(work["lat"], work["lon"], None), (home["lat"], home["lon"], None)]
        source = "route_straight_line_fallback"
    else:
        source = "route_commute_chain_scatter"

    route_points = [{"lat": float(lat), "lon": float(lon), "timestamp": str(ts) if ts is not None else None}
                    for lat, lon, ts in _dedupe_points(points)]
    return {"route_name": direction, "points": route_points, "source": source,
            "home": home, "work": work}


def _load_traj_records(uid: str, conn) -> pd.DataFrame:
    try:
        return conn.execute(f"""
            SELECT uid, event_start, event_end, status, poi_name, poi_type,
                   o_lat, o_lon, d_lat, d_lon
            FROM trajectory
            WHERE uid = '{uid}'
              AND (d_lat IS NOT NULL OR o_lat IS NOT NULL)
            ORDER BY TRY_CAST(event_start AS TIMESTAMP)
        """).fetchdf()
    except Exception:
        return pd.DataFrame()


def _parse_ts(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return pd.Timestamp(str(x).replace("Z", "+00:00")).to_pydatetime()
    except Exception:
        return None


def _coord_start(row):
    if valid_latlon(row.get("o_lat"), row.get("o_lon")):
        return float(row["o_lat"]), float(row["o_lon"])
    if valid_latlon(row.get("d_lat"), row.get("d_lon")):
        return float(row["d_lat"]), float(row["d_lon"])
    return None, None


def _coord_end(row):
    if valid_latlon(row.get("d_lat"), row.get("d_lon")):
        return float(row["d_lat"]), float(row["d_lon"])
    if valid_latlon(row.get("o_lat"), row.get("o_lon")):
        return float(row["o_lat"]), float(row["o_lon"])
    return None, None


def _extract_commute_waypoints(traj_df: pd.DataFrame, home: dict, work: dict, direction: str) -> list[tuple[float, float, object]]:
    if traj_df is None or len(traj_df) == 0:
        return []

    hl, ho = home["lat"], home["lon"]
    wl, wo = work["lat"], work["lon"]
    EXCL_KM = 0.3
    MIN_SEG_KM = 1.0
    CORRIDOR_KM = 8.0

    def at_home(lat, lon): return haversine(lat, lon, hl, ho) < EXCL_KM
    def at_work(lat, lon): return haversine(lat, lon, wl, wo) < EXCL_KM

    def near_user_region(lat, lon):
        return (haversine(lat, lon, hl, ho) < CORRIDOR_KM or
                haversine(lat, lon, wl, wo) < CORRIDOR_KM)

    if direction == "home_to_work":
        hour_lo, hour_hi = 5, 12
    else:
        hour_lo, hour_hi = 15, 23

    waypoints = []
    for _, row in traj_df.iterrows():
        ts = _parse_ts(row.get("event_start"))
        if ts is None:
            continue
        if ts.weekday() >= 5:
            continue
        if not (hour_lo <= ts.hour < hour_hi):
            continue

        status = row.get("status")
        if status == "activity":
            olat, olon = row.get("o_lat"), row.get("o_lon")
            dlat, dlon = row.get("d_lat"), row.get("d_lon")
            if not (valid_latlon(olat, olon) and valid_latlon(dlat, dlon)):
                continue
            seg_len = haversine(olat, olon, dlat, dlon)
            if seg_len < MIN_SEG_KM:
                continue
            if not (near_user_region(olat, olon) or near_user_region(dlat, dlon)):
                continue
            for lat, lon in sample_segment(olat, olon, dlat, dlon, step_km=0.3):
                if at_home(lat, lon) or at_work(lat, lon):
                    continue
                if not near_user_region(lat, lon):
                    continue
                waypoints.append((lat, lon, ts))
        elif status == "visit":
            lat, lon = _coord_end(row)
            if not valid_latlon(lat, lon):
                continue
            if at_home(lat, lon) or at_work(lat, lon):
                continue
            if not near_user_region(lat, lon):
                continue
            waypoints.append((lat, lon, ts))

    if direction == "home_to_work":
        endpoints = [(hl, ho, None), (wl, wo, None)]
    else:
        endpoints = [(wl, wo, None), (hl, ho, None)]
    return [endpoints[0]] + waypoints + [endpoints[1]]


def _dedupe_points(points: list[tuple[float, float, object]]) -> list[tuple[float, float, object]]:
    seen = set()
    out = []
    for lat, lon, ts in points:
        if not valid_latlon(lat, lon):
            continue
        key = (round(float(lat), 4), round(float(lon), 4))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(lat), float(lon), ts))
    return out


def retrieve_zone(uid: str, args: dict, conn, profile_text: str = "") -> dict | None:
    zone_name = (args.get("zone_name") or "").strip()
    if zone_name:
        user_cen_lat, user_cen_lon = _parse_extent_center(profile_text)

        _zone_prompt_tmpl = ("Look up Chinese Beijing administrative region '__ZONE__' in admin_regions table.\nUSER ACTIVITY CENTER (for disambiguation): lat=__ULAT__, lon=__ULON__\n\nCRITICAL: Some Chinese region names exist in multiple places. For example:\n  - 'Zhongguancun Subdistrict' is in Haidian District, Beijing.\n  - 'Sanlitun' is a neighbourhood in Chaoyang District, Beijing.\n  - 'Xicheng' names various sub-areas; disambiguate by distance.\n\nTherefore, you MUST disambiguate by distance to USER ACTIVITY CENTER.\n\nSCHEMA REMINDER (admin_regions columns):\n  name, name_en, name_en_norm (lowercased), admin_level,\n  lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n\nREQUIRED SQL FORMAT (use this template exactly, only substitute the zone name and user coords):\n  SELECT name, name_en, admin_level, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n  FROM admin_regions\n  WHERE LOWER(name_en_norm) = LOWER('<zone>')\n     OR LOWER(name_en_norm) LIKE LOWER('<zone>%')\n     OR LOWER(name_en) = LOWER('<zone>')\n  ORDER BY ((cen_lat - <user_lat>) * (cen_lat - <user_lat>) +\n            (cen_lon - <user_lon>) * (cen_lon - <user_lon>)) ASC\n  LIMIT 1\n\nEXAMPLE for zone='Sanlitun' with user_center=(39.94, 116.45):\n  SELECT name, name_en, admin_level, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n  FROM admin_regions\n  WHERE LOWER(name_en_norm) = 'sanlitun' OR LOWER(name_en_norm) LIKE 'sanlitun%' OR LOWER(name_en) = 'sanlitun'\n  ORDER BY ((cen_lat - 39.94) * (cen_lat - 39.94) + (cen_lon - 116.45) * (cen_lon - 116.45)) ASC\n  LIMIT 1\n\nNOW: produce the SQL for zone='__ZONE__' with user_center=(__ULAT__,__ULON__).\nStrip suffixes like ' Subdistrict', ' Community', ' Town', ' Township', ' Village', ' District', ' area' from the zone name before matching.") if _ACTIVE_DATASET == 'china' else ("Look up Japanese administrative region '__ZONE__' in admin_regions table.\nUSER ACTIVITY CENTER (for disambiguation): lat=__ULAT__, lon=__ULON__\n\nCRITICAL: Many Japanese region names exist in multiple cities. For example:\n  - 'Naka Ward' exists in Hiroshima, Nagoya, Yokohama, Sakai, Okayama, Hamamatsu.\n  - 'Nishi Ward' exists in Sapporo, Yokohama, Nagoya, Osaka, Kobe, Hiroshima, Fukuoka.\n  - 'Chiyoda' exists in Tokyo (Chiyoda Ward) AND Gunma (Chiyoda Town).\n  - 'Setagaya' (Setagaya Ward) exists only in Tokyo.\n\nTherefore, you MUST disambiguate by distance to USER ACTIVITY CENTER.\n\nSCHEMA REMINDER (admin_regions columns):\n  name, name_en, name_en_norm (lowercased), admin_level,\n  lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n\nREQUIRED SQL FORMAT (use this template exactly, only substitute the zone name and user coords):\n  SELECT name, name_en, admin_level, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n  FROM admin_regions\n  WHERE LOWER(name_en_norm) = LOWER('<zone>')\n     OR LOWER(name_en_norm) LIKE LOWER('<zone>%')\n     OR LOWER(name_en) = LOWER('<zone>')\n  ORDER BY ((cen_lat - <user_lat>) * (cen_lat - <user_lat>) +\n            (cen_lon - <user_lon>) * (cen_lon - <user_lon>)) ASC\n  LIMIT 1\n\nEXAMPLE for zone='Setagaya' with user_center=(35.64, 139.65):\n  SELECT name, name_en, admin_level, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon\n  FROM admin_regions\n  WHERE LOWER(name_en_norm) = 'setagaya' OR LOWER(name_en_norm) LIKE 'setagaya%' OR LOWER(name_en) = 'setagaya'\n  ORDER BY ((cen_lat - 35.64) * (cen_lat - 35.64) + (cen_lon - 139.65) * (cen_lon - 139.65)) ASC\n  LIMIT 1\n\nNOW: produce the SQL for zone='__ZONE__' with user_center=(__ULAT__,__ULON__).\nStrip suffixes like ' Ward', ' City', ' Prefecture', ' area' from the zone name before matching.")
        goal_admin = (_zone_prompt_tmpl
                      .replace("__ZONE__", zone_name)
                      .replace("__ULAT__", f"{user_cen_lat:.4f}")
                      .replace("__ULON__", f"{user_cen_lon:.4f}"))

        norm = zone_name.lower().replace("-", " ").replace("_", " ").strip()
        for suf in [" ward", " city", " prefecture", " area", " region", " town"]:
            if norm.endswith(suf):
                norm = norm[:-len(suf)].strip()
        norm_safe = norm.replace("'", "''")
        deterministic_sql = f"""
            SELECT name, name_en, admin_level, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon, polygon_wkt,
                   ((cen_lat - {user_cen_lat}) * (cen_lat - {user_cen_lat}) +
                    (cen_lon - {user_cen_lon}) * (cen_lon - {user_cen_lon})) AS sq_dist,
                   CASE
                       WHEN LOWER(name_en) = '{norm_safe}' THEN 0
                       WHEN LOWER(name_en_norm) = '{norm_safe}' THEN 0
                       WHEN LOWER(name_en) = '{norm_safe} ward' THEN 0
                       WHEN LOWER(name_en) = '{norm_safe} city' THEN 0
                       ELSE 1
                   END AS match_priority
            FROM admin_regions
            WHERE LOWER(name_en) = '{norm_safe}'
               OR LOWER(name_en_norm) = '{norm_safe}'
               OR LOWER(name_en) = '{norm_safe} ward'
               OR LOWER(name_en) = '{norm_safe} city'
               OR LOWER(name_en_norm) LIKE '{norm_safe} %'
               OR LOWER(name_en_norm) LIKE '{norm_safe}%'
               OR LOWER(name_en) LIKE '{norm_safe} %'
            ORDER BY match_priority ASC, sq_dist ASC
            LIMIT 1
        """
        df_a = None
        used_source = None
        try:
            df_a = conn.execute(deterministic_sql).fetchdf()
            used_source = "admin_regions_deterministic_t2"
            print(f"[retrieve_zone T2] deterministic for '{zone_name}': {len(df_a) if df_a is not None else 0} rows")
        except Exception as e:
            print(f"  [zone deterministic SQL failed] {str(e)[:80]}")
            df_a = None

        if df_a is not None and len(df_a) > 0:
            row = df_a.iloc[0]
            lat_min = float(row.get("lat_min")); lat_max = float(row.get("lat_max"))
            lon_min = float(row.get("lon_min")); lon_max = float(row.get("lon_max"))
            poly_wkt = row.get("polygon_wkt") if "polygon_wkt" in df_a.columns else None
            if poly_wkt is not None and pd.isna(poly_wkt):
                poly_wkt = None
            print(f"[retrieve_zone] {used_source}: '{zone_name}' -> '{row.get('name_en','?')}' "
                  f"center=({row.get('cen_lat')},{row.get('cen_lon')}) polygon={'YES' if poly_wkt else 'NO'}")
            return {
                "zone_name": str(row.get("name_en", zone_name)),
                "center_lat": float(row.get("cen_lat", (lat_min + lat_max)/2)),
                "center_lon": float(row.get("cen_lon", (lon_min + lon_max)/2)),
                "bbox": [lat_min, lat_max, lon_min, lon_max],
                "polygon_wkt": str(poly_wkt) if poly_wkt is not None else None,
                "radius_km": 1.5,
                "support_size": int(len(df_a)),
                "source": used_source,
            }
        print(f"[retrieve_zone] '{zone_name}' not found in admin_regions, fall through")

    bucket = args.get("time_bucket", "lunch")
    hour_ranges = {
        "breakfast": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 6 AND 9",
        "lunch": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 11 AND 13",
        "afternoon": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 14 AND 17",
        "evening": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 18 AND 22",
        "dinner": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 17 AND 21",
        "morning": "EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 6 AND 11",
        "late_night": "(EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) >= 22 OR EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) < 5)",
        "weekday_lunch": "EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 AND EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 11 AND 13",
        "weekday_morning": "EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 AND EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 6 AND 9",
        "weekday_evening": "EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 1 AND 5 AND EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN 18 AND 22",
    }
    bucket_filter = hour_ranges.get(bucket, hour_ranges["lunch"])

    goal = f"Find all visit locations of user '{uid}' during {bucket} time. Return columns: lat, lon."
    fallback = f"""
        SELECT d_lat AS lat, d_lon AS lon
        FROM trajectory
        WHERE uid = '{uid}' AND status = 'visit' AND d_lat IS NOT NULL AND d_lon IS NOT NULL
          AND {bucket_filter}
    """
    primary = _llm_generate_sql(goal, uid, profile_text)
    df = _run_sql_with_fallback(primary, fallback, conn)
    if len(df) < 3:
        return None

    c_lat = float(df["lat"].mean())
    c_lon = float(df["lon"].mean())
    dists = haversine_vec(df["lat"].values, df["lon"].values, c_lat, c_lon)
    r_km = float(np.percentile(dists, 75))
    r_km = max(r_km, 0.5)

    return {
        "zone_name": f"{bucket}_zone",
        "center_lat": c_lat,
        "center_lon": c_lon,
        "radius_km": r_km,
        "support_size": int(len(df)),
        "source": f"text2sql_zone_{bucket}",
    }


PREF_PROMPT = """The user often visits these places: {names}.

Describe their place preference in 1–2 sentences.

CRITICAL RULES:
- DO NOT mention any brand names or specific place names.
- Focus on: style, price tier, atmosphere, usage pattern, typical time-of-day.
- Be concise and generic."""


def retrieve_preference(uid: str, args: dict, conn, profile_text: str = "") -> dict | None:
    category = args.get("category") or args.get("target_category")
    top_n = int(args.get("top_n", PREF_TOP_N))
    if not category:
        return None

    categories = CATEGORY_ALIASES.get(str(category).lower(), [str(category).lower()])
    cat_list = ", ".join([f"'{c}'" for c in categories])

    goal = (
        f"Retrieve the top-{top_n} most frequently visited POIs whose poi_type is related to '{category}' "
        f"for user '{uid}'. Return columns: poi_name, poi_type, visit_count."
    )
    fallback = f"""
        SELECT poi_name, poi_type, COUNT(*) AS visit_count
        FROM trajectory
        WHERE uid = '{uid}' AND status = 'visit'
          AND lower(poi_type) IN ({cat_list})
          AND poi_name IS NOT NULL AND poi_name <> ''
        GROUP BY poi_name, poi_type
        ORDER BY visit_count DESC
        LIMIT {top_n}
    """
    primary = _llm_generate_sql(goal, uid, profile_text)
    df = _run_sql_with_fallback(primary, fallback, conn)

    if len(df) < PREF_MIN_SUPPORT:
        like_terms = " OR ".join([f"lower(poi_type) LIKE '%{c}%'" for c in categories])
        broad = f"""
            SELECT poi_name, poi_type, COUNT(*) AS visit_count
            FROM trajectory
            WHERE uid = '{uid}' AND status = 'visit'
              AND ({like_terms})
              AND poi_name IS NOT NULL AND poi_name <> ''
            GROUP BY poi_name, poi_type
            ORDER BY visit_count DESC
            LIMIT {top_n}
        """
        try:
            df = conn.execute(broad).fetchdf()
        except Exception:
            pass

    if len(df) < PREF_MIN_SUPPORT:
        return None

    names = ", ".join(df["poi_name"].astype(str).tolist())
    desc = _generate_anonymized_description(names)
    emb = _embed_text(desc)

    return {
        "category": category,
        "support_pois": df.to_dict("records"),
        "anonymized_description": desc,
        "embedding": emb,
    }


def _generate_anonymized_description(names: str) -> str:
    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            max_tokens=120,
            timeout=20,
            **_chat_kwargs(LLM_MODEL),
            messages=[
                {"role": "system", "content": "You produce brand-anonymized user preference summaries."},
                {"role": "user", "content": PREF_PROMPT.format(names=names)},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [pref] LLM failed: {str(e)[:80]}")
        return ""


_LOCAL_EMBED_MODEL = None

def _get_local_embed_model():
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_EMBED_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _LOCAL_EMBED_MODEL

def _embed_text(text: str):
    if not text:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    if USE_LOCAL_EMBED:
        m = _get_local_embed_model()
        text_in = text if text.startswith(("query: ", "passage: ")) else "passage: " + text
        vec = m.encode(text_in, normalize_embeddings=True)
        return vec.astype(np.float32)
    try:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"  [embed] failed: {str(e)[:80]}")
        return np.zeros(EMBED_DIM, dtype=np.float32)



POI_SCHEMA_HINT = """
TABLE pois (
  poi_id VARCHAR,
  name VARCHAR,
  primary_type VARCHAR,  -- one of the 192 Google Places types
  lat DOUBLE,
  lon DOUBLE,
  address VARCHAR,
  opening_hours VARCHAR
)
"""




_CN_SPATIAL_PLAN_SYSTEM = """You are a spatial query planner for a Chinese (Beijing) POI retrieval system.
Given the user's intent and the retrieved spatial evidence, you do NOT write SQL.
Instead you return a STRUCTURED PLAN as JSON describing:

  * geometry_filter: which spatial predicate to use
      "polygon_contains"   - use ST_Contains(zone_polygon, point) (zone mode)
      "polyline_distance"  - use ST_Distance(point, route_multilinestring) (route mode)
      "anchor_distance"    - haversine distance to anchor (point mode)
      "none"               - no geometric filter, pure category retrieval

  * use_bbox_prefilter: bool. If true, system adds bbox WHERE clauses around the
    polygon/polyline/anchor. ALWAYS true unless geometry_filter == "none".

  * category_whitelist: array of lowercase POI primary_type strings to filter on.
    The system gives you a `default_whitelist` from rule-based alias expansion.
    You MAY refine (add/remove) based on query semantics. If empty list, no
    category filter is applied. KEEP at least 3 categories.
    CRITICAL: category values MUST be from the AMap Chinese category set
    (e.g. chinese_restaurant, hair_salon, bathhouse, cold_drink_shop, courier_service,
     photo_studio, info_center, elderly_care_facility, life_service, residential_area,
     office_building, agency, tea_house, ...). DO NOT use Google Places / Takeout
     categories such as hospital, doctor, dental_clinic, wellness_center, coffee_shop,
     public_bath, sauna, cafeteria, resort_hotel, housing_complex, apartment_building.

  * ordering: "polygon_score" | "polyline_distance" | "anchor_haversine"
      Should match geometry_filter.

  * top_k: integer (default to top_k_hint from context; never < 50).

  * reasoning: ONE short sentence explaining the plan.

OUTPUT FORMAT: Strictly a JSON object, no markdown, no commentary. Example:

{"geometry_filter":"polygon_contains","use_bbox_prefilter":true,"category_whitelist":["specialized_hospital","clinic","elderly_care_facility"],"ordering":"polygon_score","top_k":1500,"reasoning":"User asks for hospital in Haidian Subdistrict; polygon containment plus medical categories."}
"""


def _pick_spatial_plan_system() -> str:
    try:
        ds = get_active_dataset()
    except Exception:
        ds = "japan"
    if ds == "china":
        return globals()["_CN_SPATIAL_PLAN_SYSTEM"]
    return globals()["_SPATIAL_PLAN_SYSTEM"]


_SPATIAL_PLAN_SYSTEM = """You are a spatial query planner for a Japanese POI retrieval system.
Given the user's intent and the retrieved spatial evidence, you do NOT write SQL.
Instead you return a STRUCTURED PLAN as JSON describing:

  • geometry_filter: which spatial predicate to use
      "polygon_contains"   – use ST_Contains(zone_polygon, point) (zone mode)
      "polyline_distance"  – use ST_Distance(point, route_multilinestring) (route mode)
      "anchor_distance"    – haversine distance to anchor (point mode)
      "none"               – no geometric filter, pure category retrieval

  • use_bbox_prefilter: bool. If true, system adds bbox WHERE clauses around the
    polygon/polyline/anchor. ALWAYS true unless geometry_filter == "none".

  • category_whitelist: array of lowercase POI primary_type strings to filter on.
    The system gives you a `default_whitelist` from rule-based alias expansion.
    You MAY refine (add/remove) based on query semantics. If empty list, no
    category filter is applied. KEEP at least 3 categories.

  • ordering: "polygon_score" | "polyline_distance" | "anchor_haversine"
      Should match geometry_filter: polygon→polygon_score, polyline→polyline_distance,
      anchor→anchor_haversine.

  • top_k: integer (you should default to top_k_hint from context unless query
    is very specific where you can return a smaller number, but never < 50).

  • reasoning: ONE short sentence explaining the plan.

OUTPUT FORMAT: Strictly a JSON object, no markdown, no commentary. Example:

{"geometry_filter":"polygon_contains","use_bbox_prefilter":true,"category_whitelist":["hospital","doctor","clinic","medical_lab","health"],"ordering":"polygon_score","top_k":1500,"reasoning":"User asks for hospital in Naka Ward; polygon containment plus medical category whitelist."}
"""

def _llm_write_spatial_sql(intent: dict, R_tau: dict, cat_aliases: list, top_k: int) -> str:
    import json as _json
    mode = intent.get("spatial", {}).get("mode") or (
        "zone" if R_tau.get("zone") else
        "route" if R_tau.get("route") else
        "point" if R_tau.get("point") else "?"
    )
    if mode == "?":
        return ""

    ctx_parts = [f'intent_mode = "{mode}"']
    query_text = intent.get("query") or ""
    if query_text:
        ctx_parts.append(f'query = "{query_text[:200]}"')
    if cat_aliases:
        ctx_parts.append(f"default_whitelist = {[str(a).lower() for a in cat_aliases]}")
    else:
        ctx_parts.append("default_whitelist = []")
    ctx_parts.append(f"top_k_hint = {top_k}")

    polygon_wkt_full = None
    mls_wkt_full = None
    anchor_lat = anchor_lon = None
    bbox = None

    if mode == "zone" and R_tau.get("zone") and R_tau["zone"].get("polygon_wkt"):
        z = R_tau["zone"]
        polygon_wkt_full = z["polygon_wkt"]
        bbox = z.get("bbox")
        ctx_parts.append(f'zone_name = "{z.get("zone_name","?")}"')
        if bbox:
            ctx_parts.append(f"zone_bbox_lat_lon = {bbox}  # [lat_min,lat_max,lon_min,lon_max]")
        ctx_parts.append("geometry_available = polygon")
    elif mode == "route" and R_tau.get("route"):
        r = R_tau["route"]
        direction = r.get("route_name", "home_to_work")
        chains = r.get("chains_to_work") if direction == "home_to_work" else r.get("chains_to_home")
        if not chains:
            chains = r.get("chains_to_home") if direction == "home_to_work" else r.get("chains_to_work")
        if chains:
            linestrings = []
            all_lats, all_lons = [], []
            for c in chains:
                pts = c.get("points", [])
                coords = [(float(p["lat"]), float(p["lon"])) for p in pts
                          if p.get("lat") is not None and p.get("lon") is not None]
                if len(coords) >= 2:
                    ls = "(" + ", ".join(f"{lo} {la}" for la, lo in coords) + ")"
                    linestrings.append(ls)
                    all_lats.extend([la for la, _ in coords])
                    all_lons.extend([lo for _, lo in coords])
            if linestrings:
                mls_wkt_full = "MULTILINESTRING(" + ", ".join(linestrings) + ")"
                pad = 0.012
                bbox = [min(all_lats) - pad, max(all_lats) + pad,
                        min(all_lons) - pad, max(all_lons) + pad]
                ctx_parts.append(f'direction = "{direction}"')
                ctx_parts.append(f"chain_count = {len(linestrings)}")
                ctx_parts.append(f"route_bbox_lat_lon = {bbox}")
                ctx_parts.append("geometry_available = polyline")
    elif mode == "point" and R_tau.get("point"):
        p = R_tau["point"]
        anchor_lat = p.get("lat"); anchor_lon = p.get("lon")
        if anchor_lat is None or anchor_lon is None:
            return ""
        ctx_parts.append(f"anchor = [{anchor_lat:.5f}, {anchor_lon:.5f}]  # [lat, lon]")
        ctx_parts.append("geometry_available = anchor_point")
    else:
        return ""

    user_msg = "INPUT_CONTEXT:\n  " + "\n  ".join(ctx_parts) + "\n\nReturn the JSON plan now."

    try:
        rsp = _client.chat.completions.create(
            model=LLM_MODEL,
            **_chat_kwargs(LLM_MODEL),
            messages=[
                {"role": "system", "content": _pick_spatial_plan_system()},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = rsp.choices[0].message.content or ""
        plan = _json.loads(raw)
    except Exception as e:
        print(f"  [POI cand X] LLM plan gen failed: {str(e)[:120]}")
        return ""

    geom = plan.get("geometry_filter", "none")
    if mode == "zone" and geom != "polygon_contains":
        if geom not in ("polygon_contains", "none", "anchor_distance"):
            geom = "polygon_contains"
    if mode == "route" and geom != "polyline_distance":
        if geom not in ("polyline_distance", "none"):
            geom = "polyline_distance"
    if mode == "point" and geom != "anchor_distance":
        if geom not in ("anchor_distance", "none"):
            geom = "anchor_distance"

    use_bbox = bool(plan.get("use_bbox_prefilter", True))
    whitelist = plan.get("category_whitelist") or cat_aliases or []
    whitelist = [str(c).lower().strip() for c in whitelist if str(c).strip()]
    if cat_aliases and len(cat_aliases) >= 3 and len(whitelist) < 3:
        whitelist = list({*[c.lower() for c in cat_aliases], *whitelist})
    ordering = plan.get("ordering", "polygon_score")
    k = int(plan.get("top_k", top_k))
    if k < 50: k = 50
    if k > top_k: k = top_k

    print(f"  [POI cand X] LLM plan: geom={geom} bbox={use_bbox} cats={len(whitelist)} order={ordering} k={k}")

    where_parts = ["lat IS NOT NULL", "lon IS NOT NULL"]
    if use_bbox and bbox is not None:
        lat_min, lat_max, lon_min, lon_max = bbox
        where_parts.append(f"lat BETWEEN {lat_min} AND {lat_max}")
        where_parts.append(f"lon BETWEEN {lon_min} AND {lon_max}")
    elif use_bbox and anchor_lat is not None:
        pad_lat = 0.05
        pad_lon = 0.06
        where_parts.append(f"lat BETWEEN {anchor_lat - pad_lat} AND {anchor_lat + pad_lat}")
        where_parts.append(f"lon BETWEEN {anchor_lon - pad_lon} AND {anchor_lon + pad_lon}")
    if whitelist:
        cat_list_sql = ", ".join(f"'{c}'" for c in whitelist)
        where_parts.append(f"lower(primary_type) IN ({cat_list_sql})")

    dist_expr = "0.0"
    geom_pred = None
    if geom == "polygon_contains" and polygon_wkt_full is not None:
        poly_safe = polygon_wkt_full.replace("'", "''")
        geom_pred = f"ST_Contains(ST_GeomFromText('{poly_safe}'), ST_Point(lon, lat))"
        if bbox is not None:
            cen_lat = (bbox[0] + bbox[1]) / 2
            cen_lon = (bbox[2] + bbox[3]) / 2
            dist_expr = (f"6371.0 * acos(LEAST(1.0, GREATEST(-1.0, "
                         f"sin(radians({cen_lat})) * sin(radians(lat)) + "
                         f"cos(radians({cen_lat})) * cos(radians(lat)) * cos(radians(lon - {cen_lon}))"
                         f")))")
    elif geom == "polyline_distance" and mls_wkt_full is not None:
        mls_safe = mls_wkt_full.replace("'", "''")
        dist_expr = f"ST_Distance(ST_Point(lon, lat), ST_GeomFromText('{mls_safe}')) * 100.0"
    elif geom == "anchor_distance" and anchor_lat is not None:
        dist_expr = (f"6371.0 * acos(LEAST(1.0, GREATEST(-1.0, "
                     f"sin(radians({anchor_lat})) * sin(radians(lat)) + "
                     f"cos(radians({anchor_lat})) * cos(radians(lat)) * cos(radians(lon - {anchor_lon}))"
                     f")))")

    if geom_pred:
        where_parts.append(geom_pred)

    where_sql = "\n  AND ".join(where_parts)

    order_clause = ""
    if dist_expr != "0.0":
        order_clause = f"ORDER BY ({dist_expr}) ASC"

    sql = f"""
        SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours,
               ({dist_expr}) AS dist_km
        FROM pois
        WHERE {where_sql}
        {order_clause}
        LIMIT {k}
    """
    _cap_geom(
        mode=mode, geom=geom, use_bbox=use_bbox,
        bbox=list(bbox) if bbox is not None else None,
        polygon_wkt=polygon_wkt_full, mls_wkt=mls_wkt_full,
        anchor_lat=anchor_lat, anchor_lon=anchor_lon,
        whitelist=list(whitelist) if whitelist else [],
        ordering=ordering, top_k=k,
        sql=sql.strip()[:2000], source='patch_w_llm',
    )
    return sql.strip()




def _validate_spatial_sql_result(df, expected_cols, mode):
    if df is None or len(df) == 0:
        return False, "empty result"
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        return False, f"missing columns {missing}"
    try:
        valid_frac = df[["lat", "lon"]].notna().all(axis=1).mean()
        if valid_frac < 0.5:
            return False, f"only {valid_frac:.1%} rows have valid lat/lon"
    except Exception:
        pass
    return True, "ok"


def retrieve_poi_candidates_via_sql(intent: dict, R_tau: dict, conn, top_k: int = 1500) -> list[dict]:
    try:
        _spatial_ok_w = False
        try:
            conn.execute("LOAD spatial")
            _spatial_ok_w = True
        except Exception:
            _spatial_ok_w = False

        _target_cat_w = intent.get("target", {}).get("category")
        _target_cats_list_w = intent.get("target", {}).get("categories") or []
        if not _target_cats_list_w and _target_cat_w:
            _target_cats_list_w = [_target_cat_w]
        _cat_aliases_w = []
        if _target_cats_list_w:
            for _c_one in _target_cats_list_w:
                _c_lc = str(_c_one).lower().strip()
                if not _c_lc: continue
                _cur_w = list(CATEGORY_ALIASES.get(_c_lc, []))
                if not _cur_w:
                    try:
                        from candidate import _resolve_category_to_valid_types
                        _cur_w = _resolve_category_to_valid_types(_c_one, top_k=5)
                    except Exception:
                        _cur_w = []
                if _c_lc not in [a.lower() for a in _cur_w]:
                    _cur_w.append(_c_lc)
                _cat_aliases_w.extend(_cur_w)
            _seen_w = set()
            _cat_aliases_w = [c for c in _cat_aliases_w if not (c.lower() in _seen_w or _seen_w.add(c.lower()))]

        _mode_w = intent.get("spatial", {}).get("mode")
        _can_use_llm = (
            (_mode_w == "zone" and R_tau.get("zone") and R_tau["zone"].get("polygon_wkt")) or
            (_mode_w == "route" and R_tau.get("route") and (R_tau["route"].get("chains_to_work") or R_tau["route"].get("chains_to_home"))) or
            (_mode_w == "point" and R_tau.get("point") and R_tau["point"].get("lat") is not None)
        )

        if _spatial_ok_w and _can_use_llm:
            _sql_w = _llm_write_spatial_sql(intent, R_tau, _cat_aliases_w, top_k)
            if _sql_w and _sql_w.lower().startswith("select"):
                try:
                    _df_w = conn.execute(_sql_w).fetchdf()
                    _ok, _why = _validate_spatial_sql_result(
                        _df_w,
                        expected_cols=["poi_id","name","category","lat","lon","dist_km"],
                        mode=_mode_w,
                    )
                    print(f"  [POI cand W] LLM SQL (mode={_mode_w}) -> {len(_df_w) if _df_w is not None else 0} rows, valid={_ok} ({_why})")
                    if _ok:
                        out_w = []
                        for _, row in _df_w.iterrows():
                            out_w.append({
                                "poi_id": str(row["poi_id"]),
                                "name": str(row["name"]),
                                "category": str(row["category"]),
                                "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                                "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                                "address": str(row.get("address","")) if pd.notna(row.get("address")) else "",
                                "opening_hours": str(row.get("opening_hours","")) if pd.notna(row.get("opening_hours")) else "",
                                "dist_km": float(row["dist_km"]) if pd.notna(row["dist_km"]) else 0.0,
                            })
                        _cap_geom(cand_ids=[(c["poi_id"], c.get("lat"), c.get("lon")) for c in out_w[:100]], cand_n=len(out_w), path="patch_w_ok")
                        return out_w
                except Exception as _ew:
                    print(f"  [POI cand W] LLM SQL exec failed: {str(_ew)[:120]} → fall back to deterministic Patch V")
    except Exception as _outer_w:
        print(f"  [POI cand W] outer error → fall back to deterministic Patch V: {str(_outer_w)[:120]}")

    try:
        _spatial_ok = False
        try:
            conn.execute("LOAD spatial")
            _spatial_ok = True
        except Exception:
            _spatial_ok = False

        target_cat_v = intent.get("target", {}).get("category")
        target_cats_list_v = intent.get("target", {}).get("categories") or []
        if not target_cats_list_v and target_cat_v:
            target_cats_list_v = [target_cat_v]
        cat_aliases_v = []
        if target_cats_list_v:
            for c_one in target_cats_list_v:
                c_lc = str(c_one).lower().strip()
                if not c_lc: continue
                cur_v = list(CATEGORY_ALIASES.get(c_lc, []))
                if not cur_v:
                    try:
                        from candidate import _resolve_category_to_valid_types
                        cur_v = _resolve_category_to_valid_types(c_one, top_k=5)
                    except Exception:
                        cur_v = []
                if c_lc not in [a.lower() for a in cur_v]:
                    cur_v.append(c_lc)
                cat_aliases_v.extend(cur_v)
            _seen_v = set()
            cat_aliases_v = [c for c in cat_aliases_v if not (c.lower() in _seen_v or _seen_v.add(c.lower()))]
        if cat_aliases_v:
            cat_list_v = ", ".join([f"'{a.lower()}'" for a in cat_aliases_v])
            cat_clause_v = f"lower(primary_type) IN ({cat_list_v})"
        else:
            cat_clause_v = "1=1"

        if _spatial_ok and R_tau.get("zone") and R_tau["zone"].get("polygon_wkt"):
            z = R_tau["zone"]
            poly_wkt_v = z["polygon_wkt"]
            lat_min_v, lat_max_v, lon_min_v, lon_max_v = z.get("bbox", [None]*4)
            if all(x is not None for x in (lat_min_v, lat_max_v, lon_min_v, lon_max_v)):
                poly_safe = poly_wkt_v.replace("'", "''")
                sql_v = f"""
                    SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours,
                           0.0 AS dist_km,
                           1 AS in_polygon
                    FROM pois
                    WHERE lat IS NOT NULL AND lon IS NOT NULL
                      AND lat BETWEEN {lat_min_v} AND {lat_max_v}
                      AND lon BETWEEN {lon_min_v} AND {lon_max_v}
                      AND ({cat_clause_v})
                      AND ST_Contains(
                            ST_GeomFromText('{poly_safe}'),
                            ST_Point(lon, lat)
                          )
                    LIMIT {top_k}
                """
                try:
                    df_v = conn.execute(sql_v).fetchdf()
                    print(f"  [POI cand V] ZONE polygon ST_Contains -> {len(df_v)} POIs (cat={cat_aliases_v[:5] if cat_aliases_v else 'any'})")
                    if len(df_v) < min(top_k, 200):
                        sql_v2 = f"""
                            SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours,
                                   ((lat-{(lat_min_v+lat_max_v)/2})*(lat-{(lat_min_v+lat_max_v)/2}) +
                                    (lon-{(lon_min_v+lon_max_v)/2})*(lon-{(lon_min_v+lon_max_v)/2})) AS dist_km,
                                   0 AS in_polygon
                            FROM pois
                            WHERE lat IS NOT NULL AND lon IS NOT NULL
                              AND lat BETWEEN {lat_min_v-0.02} AND {lat_max_v+0.02}
                              AND lon BETWEEN {lon_min_v-0.02} AND {lon_max_v+0.02}
                              AND ({cat_clause_v})
                              AND NOT EXISTS (
                                SELECT 1 FROM (SELECT poi_id AS pid FROM pois LIMIT 0) WHERE pid = pois.poi_id
                              )
                            ORDER BY dist_km ASC
                            LIMIT {top_k - len(df_v)}
                        """
                        try:
                            df_v2 = conn.execute(sql_v2).fetchdf()
                            inner_ids = set(df_v["poi_id"].astype(str).tolist()) if len(df_v) > 0 else set()
                            df_v2 = df_v2[~df_v2["poi_id"].astype(str).isin(inner_ids)]
                            df_v = pd.concat([df_v, df_v2], ignore_index=True)
                            print(f"  [POI cand V] ZONE bbox soft-expand +{len(df_v2)} -> {len(df_v)} POIs")
                        except Exception as _e2:
                            print(f"  [POI cand V] zone bbox expand failed: {str(_e2)[:80]}")
                    if len(df_v) > 0:
                        out_v = []
                        for _, row in df_v.iterrows():
                            out_v.append({
                                "poi_id": str(row["poi_id"]),
                                "name": str(row["name"]),
                                "category": str(row["category"]),
                                "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                                "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                                "address": str(row["address"]) if pd.notna(row["address"]) else "",
                                "opening_hours": str(row["opening_hours"]) if pd.notna(row["opening_hours"]) else "",
                                "dist_km": float(row["dist_km"]) if pd.notna(row["dist_km"]) else 0.0,
                                "in_polygon": int(row["in_polygon"]) if "in_polygon" in df_v.columns else 0,
                            })
                        _cap_geom(
                            mode='zone', geom='polygon_contains',
                            polygon_wkt=polygon_wkt_v if 'polygon_wkt_v' in dir() else None,
                            bbox=[lat_min_v, lat_max_v, lon_min_v, lon_max_v] if 'lat_min_v' in dir() else None,
                            whitelist=list(cat_aliases_v)[:20] if cat_aliases_v else [],
                            sql=sql_v[:2000], source='patch_v_zone',
                            cand_ids=[(c['poi_id'], c.get('lat'), c.get('lon')) for c in out_v[:100]],
                            cand_n=len(out_v),
                        )
                        return out_v
                except Exception as _ev:
                    print(f"  [POI cand V] zone ST_Contains SQL failed: {str(_ev)[:120]}")

        if _spatial_ok and R_tau.get("route"):
            rinfo_v = R_tau["route"]
            direction_v = rinfo_v.get("route_name", "home_to_work")
            chains_v = rinfo_v.get("chains_to_work") if direction_v == "home_to_work" else rinfo_v.get("chains_to_home")
            if not chains_v:
                chains_v = rinfo_v.get("chains_to_home") if direction_v == "home_to_work" else rinfo_v.get("chains_to_work")
            if chains_v:
                linestrings = []
                all_lats, all_lons = [], []
                for c in chains_v:
                    pts = c.get("points", [])
                    coords = [(float(p["lat"]), float(p["lon"])) for p in pts
                              if p.get("lat") is not None and p.get("lon") is not None]
                    if len(coords) >= 2:
                        ls = "(" + ", ".join(f"{lo} {la}" for la, lo in coords) + ")"
                        linestrings.append(ls)
                        all_lats.extend([la for la, _ in coords])
                        all_lons.extend([lo for _, lo in coords])
                if linestrings:
                    mls_wkt = "MULTILINESTRING(" + ", ".join(linestrings) + ")"
                    mls_safe = mls_wkt.replace("'", "''")
                    pad = 0.012
                    lat_lo = min(all_lats) - pad; lat_hi = max(all_lats) + pad
                    lon_lo = min(all_lons) - pad; lon_hi = max(all_lons) + pad
                    sql_v = f"""
                        SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours,
                               ST_Distance(ST_Point(lon, lat), ST_GeomFromText('{mls_safe}')) AS dist_deg
                        FROM pois
                        WHERE lat IS NOT NULL AND lon IS NOT NULL
                          AND lat BETWEEN {lat_lo} AND {lat_hi}
                          AND lon BETWEEN {lon_lo} AND {lon_hi}
                          AND ({cat_clause_v})
                        ORDER BY dist_deg ASC
                        LIMIT {top_k}
                    """
                    try:
                        df_v = conn.execute(sql_v).fetchdf()
                        print(f"  [POI cand V] ROUTE MULTILINESTRING ST_Distance ({len(chains_v)} chains, dir={direction_v}) -> {len(df_v)} POIs")
                        if len(df_v) > 0:
                            out_v = []
                            for _, row in df_v.iterrows():
                                dist_deg_v = float(row["dist_deg"]) if pd.notna(row["dist_deg"]) else 0.0
                                dist_km_v = dist_deg_v * 100.0
                                out_v.append({
                                    "poi_id": str(row["poi_id"]),
                                    "name": str(row["name"]),
                                    "category": str(row["category"]),
                                    "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
                                    "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
                                    "address": str(row["address"]) if pd.notna(row["address"]) else "",
                                    "opening_hours": str(row["opening_hours"]) if pd.notna(row["opening_hours"]) else "",
                                    "dist_km": dist_km_v,
                                })
                            _cap_geom(
                                mode='route', geom='polyline_distance',
                                mls_wkt=mls_wkt if 'mls_wkt' in dir() else None,
                                bbox=[lat_lo, lat_hi, lon_lo, lon_hi] if 'lat_lo' in dir() else None,
                                whitelist=list(cat_aliases_v)[:20] if cat_aliases_v else [],
                                sql=sql_v[:2000], source='patch_v_route',
                                cand_ids=[(c['poi_id'], c.get('lat'), c.get('lon')) for c in out_v[:100]],
                                cand_n=len(out_v),
                                chains_wkt=linestrings if 'linestrings' in dir() else None,
                                direction=direction_v if 'direction_v' in dir() else None,
                            )
                            return out_v
                    except Exception as _ev:
                        print(f"  [POI cand V] route ST_Distance SQL failed: {str(_ev)[:120]}")
    except Exception as _outer:
        print(f"  [POI cand V] outer error, fall back to legacy: {str(_outer)[:120]}")

    """v6: pull POI candidates directly via SQL using anchor + category.

    Returns a list of POI dicts compatible with scoring.score_candidates.
    SQL filters by category whitelist (LLM-resolved + alias expansion),
    then orders by distance to anchor.
    """
    target_cat = intent.get("target", {}).get("category")
    target_cats_list = intent.get("target", {}).get("categories") or []
    if not target_cats_list and target_cat:
        target_cats_list = [target_cat]

    target_cat_aliases = []
    if not target_cats_list:
        target_cat_aliases = []
    else:
        for cat_one in target_cats_list:
            cat_lc = str(cat_one).lower().strip()
            if not cat_lc:
                continue
            cur = list(CATEGORY_ALIASES.get(cat_lc, []))
            if not cur:
                try:
                    from candidate import _resolve_category_to_valid_types
                    cur = _resolve_category_to_valid_types(cat_one, top_k=5)
                    if cur:
                        print(f"  [POI cand] cat='{cat_one}' resolved by embedding -> {cur}")
                except Exception as e:
                    print(f"  [POI cand] embedding resolve failed: {str(e)[:80]}")
                    cur = []
            if cat_lc not in [a.lower() for a in cur]:
                cur.append(cat_lc)
            target_cat_aliases.extend(cur)
        _seen = set()
        target_cat_aliases = [c for c in target_cat_aliases
                              if not (c.lower() in _seen or _seen.add(c.lower()))]
        if target_cat_aliases:
            print(f"  [POI cand] multi-cat expansion: {target_cats_list} -> {target_cat_aliases[:10]}{'...' if len(target_cat_aliases) > 10 else ''}")

    anchor_lat = anchor_lon = None
    route_anchors = []
    if R_tau.get("point"):
        anchor_lat = R_tau["point"].get("lat")
        anchor_lon = R_tau["point"].get("lon")
    elif R_tau.get("route") and R_tau["route"].get("points"):
        rinfo = R_tau["route"]
        pts = rinfo["points"]
        if rinfo.get("home"):
            route_anchors.append((rinfo["home"]["lat"], rinfo["home"]["lon"]))
        if rinfo.get("work"):
            wlat, wlon = rinfo["work"]["lat"], rinfo["work"]["lon"]
            if not route_anchors or (abs(wlat - route_anchors[0][0]) + abs(wlon - route_anchors[0][1]) > 0.001):
                route_anchors.append((wlat, wlon))
        if len(pts) >= 2:
            mid = pts[len(pts)//2]
            mlat, mlon = mid.get("lat"), mid.get("lon")
            if mlat is not None and mlon is not None:
                close = any(abs(mlat-a[0])+abs(mlon-a[1]) < 0.001 for a in route_anchors)
                if not close:
                    route_anchors.append((mlat, mlon))
        if route_anchors:
            anchor_lat, anchor_lon = route_anchors[0]
        elif pts:
            mid = pts[len(pts)//2]
            anchor_lat = mid.get("lat"); anchor_lon = mid.get("lon")
    elif R_tau.get("zone"):
        anchor_lat = R_tau["zone"].get("center_lat")
        anchor_lon = R_tau["zone"].get("center_lon")

    if target_cat_aliases:
        cat_list = ", ".join([f"'{a.lower()}'" for a in target_cat_aliases])
        cat_clause = f"lower(primary_type) IN ({cat_list})"
    else:
        cat_clause = "1=1"

    if anchor_lat is not None and anchor_lon is not None:
        def _hav_expr(lat_v, lon_v):
            return (f"(6371.0 * acos(LEAST(1.0, GREATEST(-1.0, "
                    f"sin(radians({lat_v})) * sin(radians(lat)) + "
                    f"cos(radians({lat_v})) * cos(radians(lat)) * cos(radians(lon - {lon_v}))"
                    f"))))")
        if len(route_anchors) > 1:
            dist_terms = [_hav_expr(la, lo) for la, lo in route_anchors]
            dist_expr = "LEAST(" + ", ".join(dist_terms) + ")"
        else:
            dist_expr = _hav_expr(anchor_lat, anchor_lon)
        sql = f"""
            SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours,
                   {dist_expr} AS dist_km
            FROM pois
            WHERE lat IS NOT NULL AND lon IS NOT NULL
              AND ({cat_clause})
            ORDER BY dist_km ASC
            LIMIT {top_k}
        """
    else:
        sql = f"""
            SELECT poi_id, name, primary_type AS category, lat, lon, address, opening_hours
            FROM pois
            WHERE lat IS NOT NULL AND lon IS NOT NULL
              AND ({cat_clause})
            LIMIT {top_k}
        """

    try:
        df = conn.execute(sql).fetchdf()
    except Exception as e:
        print(f"  [unified] SQL failed: {str(e)[:100]}")
        return []

    out = []
    for _, row in df.iterrows():
        out.append({
            "poi_id": str(row["poi_id"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
            "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
            "address": str(row.get("address", "")),
            "opening_hours": str(row.get("opening_hours", "")),
            "description": "",
        })
    _cap_geom(
        mode=(intent.get("spatial",{}) or {}).get("mode"),
        geom='anchor_distance' if anchor_lat is not None else 'none',
        anchor_lat=anchor_lat, anchor_lon=anchor_lon,
        route_anchors=list(route_anchors) if route_anchors else None,
        whitelist=list(target_cat_aliases)[:20] if target_cat_aliases else [],
        sql=sql[:2000] if sql else None,
        source='legacy_anchor',
        cand_ids=[(c['poi_id'], c.get('lat'), c.get('lon')) for c in out[:100]],
        cand_n=len(out),
    )
    return out


def unified_retrieve(intent: dict, uid: str, conn, top_k_total: int = 1500, profile_text: str = "") -> tuple[dict, list[dict]]:
    R_tau = retrieve_evidence(intent, uid, conn, profile_text)
    candidates = retrieve_poi_candidates_via_sql(intent, R_tau, conn, top_k=top_k_total)
    return R_tau, candidates
