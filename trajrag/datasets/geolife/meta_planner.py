import os, json, sys
from openai import OpenAI
try:
    from evidence import get_active_dataset
except Exception:
    def get_active_dataset(): return 'japan'
sys.path.insert(0, os.path.dirname(__file__))
try:
    from profile_compact import compact as compact_profile
except ImportError:
    from build_profile_v2 import compact as compact_profile
try:
    from profile import build_profile
except ImportError:
    from build_profile import build_profile

from constants import OPENAI_KEY, OPENAI_BASE_URL, LLM_MODEL
from parser import _chat_kwargs  

_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY)


_DATASET_TEMPLATES = {
    "japan": {
        "DATASET_DISPLAY": "Japan",
        "ADMIN_LEVEL_DESC": "admin_level: 4=都道府県, 7=市町村, 8=政令市内区 (Ward)",
        "NAME_EXAMPLE": '"Naka Ward", "Ishioka"',
        "REGION_DESC": "city/ward/prefecture",
        "REGION_EXAMPLES": '"in Naka Ward", "in Ishioka", "around Tokyo", "Setagaya"',
        "ZONE_EX_1": '"hospital in Naka Ward"',
    },
    "china": {
        "DATASET_DISPLAY": "Beijing (China)",
        "ADMIN_LEVEL_DESC": "admin_level: 6=区(District), 8=街道(Subdistrict)/社区(Community)/镇(Town), 9=neighbourhood",
        "NAME_EXAMPLE": '"Zhongguancun Subdistrict", "Sanlitun", "Wudaokou", "Haidian District"',
        "REGION_DESC": "district/subdistrict/community/neighbourhood",
        "REGION_EXAMPLES": '"in Zhongguancun Subdistrict", "around Sanlitun", "in Wudaokou", "Haidian area"',
        "ZONE_EX_1": '"clinic in Zhongguancun Subdistrict"',
    },
}

def _render_planner_prompt(template_str):
    ds = get_active_dataset()
    vars = _DATASET_TEMPLATES.get(ds, _DATASET_TEMPLATES["japan"])
    out = template_str
    for k, v in vars.items():
        out = out.replace("{" + k + "}", v)
    return out

_CATS_BY_DATASET = {}
for _name in ("japan", "china"):
    try:
        _fn = "valid_types.json" if _name == "japan" else "valid_types_geolife.json"
        _p = os.path.join(os.path.dirname(__file__), _fn)
        _CATS_BY_DATASET[_name] = json.load(open(_p))["types"]
    except Exception as _e:
        _CATS_BY_DATASET[_name] = []
        print(f"[meta_planner] load {_name} valid_types failed: {_e}")

_CATS = _CATS_BY_DATASET.get("japan", []) or []
_CAT_LIST = ", ".join(_CATS) if _CATS else "restaurant, cafe, ..."

def _current_cat_list():
    ds = get_active_dataset()
    cats = _CATS_BY_DATASET.get(ds) or _CATS_BY_DATASET.get("japan", []) or []
    return ", ".join(cats) if cats else _CAT_LIST


META_PROMPT_TMPL = """You are a POI retrieval assistant. Given an intent (from Step 1),
a user trajectory profile (offline statistics), and the trajectory table schema (for SQL),
produce a JSON plan that selects spatial anchors needed to retrieve POIs.

────────────────────────────────────────────
USER QUERY: "{query}"

────────────────────────────────────────────
USER_PROFILE (offline statistics):
{profile_text}

────────────────────────────────────────────
AVAILABLE SQL TABLES (use SQL when profile is insufficient):

TABLE trajectory(uid, event_start, event_end, poi_id, poi_name, poi_type,
                 o_lat, o_lon, d_lat, d_lon, status)
  - raw user events. Use EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) for hour.
  - Use EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) for day-of-week.
  - dow: 0=Sunday, 1=Monday, ..., 6=Saturday

TABLE admin_regions(osm_id, name, name_en, name_en_norm, name_ja,
                    admin_level, iso3166_2,
                    lat_min, lat_max, lon_min, lon_max,
                    cen_lat, cen_lon, polygon_wkt)
  - {DATASET_DISPLAY} administrative boundaries.
  - {ADMIN_LEVEL_DESC}
  - name_en: English name (e.g. {NAME_EXAMPLE})
  - name_en_norm: lowercased, accent-stripped name_en for matching
  - lat_min/lat_max/lon_min/lon_max: bbox

────────────────────────────────────────────
POI CATEGORY WHITELIST (target.category MUST be an exact lowercase match from this list):
{cat_list}

────────────────────────────────────────────
SPATIAL MODES — pick exactly ONE based on the SPATIAL SCOPE implied by the query:

★ "point" — Query implies a SINGLE POINT (nearby search around one location).
   The user has a specific anchor in mind, looking for POIs close to it.
   Triggers:
   - "nearby", "near me", "close by", "around here"
   - Implicit current-location queries: "tonight", "today", "this afternoon",
     "this morning", "this evening", "right now", "later today"
   - Just a category with no spatial cue
   Examples:
     "good ramen nearby" → point
     "convenience store close by" → point
     "ramen tonight" → point  (anchor = current location)
     "sushi this afternoon" → point
     "place to grab a quick bite" → point

★ "route" — Query implies a PATH (search along a trajectory line).
   The user is moving from A to B, looking for POIs along the way.
   Triggers:
   - "on my way to/from work/home"
   - "during my commute" / "on commute"
   - "while heading to X" / "on the route to Y"
   Examples:
     "coffee on the way to work" → route
     "convenience store on commute" → route

★ "zone" — Query implies a REGION (search within an area, not a single point).
   The scope is broader than "around one location"; the user has a region 
   (named or calendar-implicit) where they want to search.
   Triggers:
   - Named {DATASET_DISPLAY} region: {REGION_DESC} with explicit place name
     ({REGION_EXAMPLES})
   - Calendar-implicit region: query mentions a specific day-of-week or weekend
     ("this Saturday morning", "weekend afternoon", "Tuesday evening").
     The region is the user's TYPICAL activity area during that calendar slot,
     which is broader than a single current-location anchor.
   Examples:
     {ZONE_EX_1} → zone (named region)
     "park this Saturday morning" → zone (weekend AM region)
     "sushi Friday evening" → zone (specific weekday evening region)

   DO NOT pick zone when:
   - "Nearby" / "close by" / no spatial scope cue → point
   - "Tonight" / "today" / "this afternoon" without dow → point
   - The "name" looks like a category (karaoke, park, gym, restaurant) → these
     are category words, not place names; use point (or zone Case B if calendar word)

────────────────────────────────────────────
SQL FOR ZONE — emit SQL ONLY when spatial_mode == "zone":

  Case A (named place): SQL on admin_regions
    SELECT name, name_en, lat_min, lat_max, lon_min, lon_max, cen_lat, cen_lon
    FROM admin_regions
    WHERE lower(name_en_norm) IN ('<lowercase_place>', '<lowercase_place> ward')
      AND admin_level IN (7, 8)
    ORDER BY ((cen_lat - <user_cen_lat>)*(cen_lat - <user_cen_lat>) +
              (cen_lon - <user_cen_lon>)*(cen_lon - <user_cen_lon>)) ASC
    LIMIT 1
    -- Replace <user_cen_lat>/<user_cen_lon> with the center of user's extent from profile.

  Case B (calendar-implicit): SQL on trajectory
    SELECT
      AVG(d_lat) AS cen_lat, AVG(d_lon) AS cen_lon,
      MIN(d_lat) AS lat_min, MAX(d_lat) AS lat_max,
      MIN(d_lon) AS lon_min, MAX(d_lon) AS lon_max,
      COUNT(*) AS n_visits
    FROM trajectory
    WHERE uid = '<uid>' AND status = 'visit'
      AND d_lat IS NOT NULL AND d_lon IS NOT NULL
      AND EXTRACT(dow FROM TRY_CAST(event_start AS TIMESTAMP)) IN (<dow_set>)
      AND EXTRACT(hour FROM TRY_CAST(event_start AS TIMESTAMP)) BETWEEN <h_lo> AND <h_hi>
    HAVING COUNT(*) >= 3

  Decode time words for Case B:
    Saturday → dow=6,  Sunday → dow=0,  Monday=1 ... Friday=5
    weekend → dow IN (0,6)
    weekday → dow IN (1,2,3,4,5)
    morning  → hour 6-11
    afternoon → hour 12-17
    evening / dinner → hour 18-22
    lunch → hour 11-14

For point or route mode: DO NOT emit admin_regions or trajectory zone SQL.

WRONG examples (do NOT do these):
  ✗ "good karaoke bar" → NEVER `WHERE name_en_norm LIKE '%karaoke%'`
    (karaoke is a category, not a place. Use point.)
  ✗ "Japanese food tonight" → NEVER `LIKE '%japan%'`
    (cuisine, not a place. Use point.)
  ✗ "ramen tonight" → point (no place name + no dow → point, not zone)
  ✗ "quick bite this afternoon" → point (no dow + no place → point)

────────────────────────────────────────────
OUTPUT JSON:
{{
  "intent": {{
    "spatial_mode": "route" | "point" | "zone",
    "category": "<one whitelist item>" | null,
    "time_bucket": "morning"|"lunch"|"afternoon"|"dinner"|"evening"|"late_night"|null
  }},
  "anchors": [
    {{"role": "current" | "origin" | "destination" | "region_center",
      "lat": <number>, "lon": <number>,
      "source": "profile.top_pois[<index>]" | "sql_result_<id>",
      "reasoning": "<why this anchor>"}}
  ],
  "sql_queries": [
    {{"id":"sql_1", "purpose":"<what this SQL computes>", "sql":"<duckdb sql>"}}
  ]
}}

Output ONLY the JSON, no markdown.
"""


def plan_query(query: str, uid: str, traj: list, profile_cache: dict | None = None) -> dict:
    if profile_cache is not None and uid in profile_cache:
        profile = profile_cache[uid]
    else:
        profile = build_profile(uid, traj)
        if profile_cache is not None:
            profile_cache[uid] = profile

    text = compact_profile(profile)
    prompt = _render_planner_prompt(META_PROMPT_TMPL).format(profile_text=text, cat_list=_current_cat_list(), query=query)

    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            response_format={"type":"json_object"},
            **_chat_kwargs(LLM_MODEL),
            messages=[
                {"role":"system","content":"You are a careful deterministic JSON planner."},
                {"role":"user","content": prompt},
            ],
        )
        plan = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[meta_planner] LLM failed: {str(e)[:150]}")
        if profile.get("spatial_clusters"):
            c = profile["spatial_clusters"][0]
            plan = {
                "intent": {"spatial_mode":"point","category":None,"time_bucket":None},
                "anchors":[{"role":"current","lat":c["center"][0],"lon":c["center"][1],
                            "source":"profile.spatial_clusters[0]","reasoning":"fallback"}],
                "sql_queries": [],
            }
        else:
            plan = {"intent":{"spatial_mode":"none","category":None,"time_bucket":None},
                    "anchors":[], "sql_queries":[]}

    plan["_profile"] = profile
    return plan
