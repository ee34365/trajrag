import json
import re

def _strip_md_fence(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()

from openai import OpenAI
from constants import OPENAI_KEY, OPENAI_BASE_URL, LLM_MODEL, PREF_TOP_N


def _is_reasoning_model(m: str) -> bool:
    m = (m or "").lower()
    return ("gpt-5" in m
            or "o1" in m or "o3" in m or "o4" in m
            or "gemini-2.5-pro" in m or "gemini-3" in m
            or ("claude" in m and ("opus" in m or "thinking" in m))
            or "qwen3.5" in m or "qwen3.6" in m or "qwen3.7" in m
            or "deepseek-v4-pro" in m or "deepseek-r1" in m
            or "deepseek-reasoner" in m)

def _is_qwen_reasoning(m: str) -> bool:
    m = (m or "").lower()
    return "qwen3.5" in m or "qwen3.6" in m or "qwen3.7" in m or "deepseek-v4-pro" in m

def _chat_kwargs(model: str) -> dict:
    import os as _os
    if "11434" in (_os.environ.get("OPENAI_BASE_URL") or ""):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if _is_qwen_reasoning(model):
        return {"extra_body": {"reasoning": {"enabled": False}}}
    if _is_reasoning_model(model):
        return {"extra_body": {"reasoning": {"effort": "low"}}}
    return {}



_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()


INTENT_SYSTEM = f"""You are a query parser for personalized POI retrieval. Output STRICT JSON only.

Schema:
{{
  "target": {{"entity_type": "poi", "category": string|null, "categories": [string], "keywords": [string]}},
  "spatial": {{"mode": "route" | "point" | "zone" | "none",
              "anchor": "home"|"work"|"current"|null,
              "route_direction": "home_to_work"|"work_to_home"|null,
              "zone_name": string|null,
              "raw_span": string | null}},
  "temporal": {{"time_bucket": string|null, "day_hint": string|null}},
  "must_conditions": [{{"field": string, "operator": string, "value": string | null, "source_span": string}}],
  "preference_conditions": [{{"field": string, "operator": "prefer"|"avoid", "value": string, "source_span": string}}],
  "evidence_requirements": [{{"type": "point"|"route"|"zone"|"preference", "name": string, "arguments": object}}],
  "scoring_objective": ["spatial","temporal","semantic","preference"]
}}

CRITICAL PRIORITY ORDER: First check Rule #1 (route). If ANY route phrase matches, mode MUST be "route" — IGNORE all time-word defaults.
If no route phrase matches, check Rule #2 (zone). Only fall back to Rule #3 (point) when neither route nor zone applies.

Rules for spatial.mode:
1. "route" — query mentions ANY commute path phrase (HIGHEST PRIORITY — overrides time-word defaults):
   Trigger phrases (case-insensitive, anywhere in query):
   - "on my way home" / "on my way to work" / "on my way to/from <place>"
   - "along my route" / "along my way" / "along the way home"
   - "on my commute" / "during my commute" / "on the commute"
   - "heading home" / "heading to work" / "head home from work"
   - "after work" + any movement word (going, stop, swing by, pick up, hit) → likely route home
   - "before work" + any movement word → likely route to work
   - "on my drive" / "while driving" / "stop at...while driving"
   - "on the way back" / "on my way back home"
   - "swing by ... on my way" / "hit ... on my way"
   route_direction inference:
   - "to work" / "before work" / "morning commute" / "heading to work" → home_to_work
   - "home" / "after work" / "evening commute" / "heading home" / "way back" → work_to_home

2. "zone" — query SEARCHES FOR results INSIDE a named Japanese region.
   The named-region triggers are:
   - 区 / Ward (e.g. "Naka Ward", "Setagaya")
   - 市 / City (e.g. "Ishioka", "Funabashi", "Kyoto")
   - 県 / Prefecture (e.g. "Aichi", "Hyogo")
   - 府 / 都 (e.g. "Osaka", "Tokyo")
   - "X area" (e.g. "Ichinomiya area", "Takasago area")

   ★ CRITICAL DISTINCTION for "in / at / around / near <place>":
   
   (a) SPEAKER-LOCATION HINT — the speaker declares they ARE currently in
       the place; they want a nearby POI, not a search inside the place.
       → mode="point", anchor="current" (NOT zone)
       Examples:
         "I'm in Noda and need to fill up my tank" → point/current
         "I'm out in Hadano this afternoon"        → point/current
         "while I'm out in <place>"                → point/current
         "in <place>, where can I..."              → point/current (speaker already there)
   
   (b) SEARCH-REGION SCOPE — the query asks for a POI whose location is
       INSIDE the named place; the speaker is not necessarily there.
       → mode="zone"
       Examples:
         "any parks in Setagaya?"                  → zone
         "recommend a hotel in Takasago"           → zone
         "clinic in Naka Ward for a checkup"       → zone
         "groceries in Itami tomorrow"             → zone
         "car dealership in <place> this weekend"  → zone
   
   Rule of thumb: if the sentence CAN be paraphrased as "I am currently
   located in X, therefore find nearby Y", it is (a) → point/current.
   If it paraphrases as "find Y that is located in X", it is (b) → zone.
   
   Whenever (b) triggers, set spatial_mode="zone" and put the recognized
   place name in zone_name (use the English form as it appears in the query).

3. "point" (FALLBACK — only when neither route nor zone applies):
   - anchor = "current" — when query has time word ("tonight", "this afternoon", "this evening", "today", "later", "now", "right now") AND no route phrase from Rule #1
   - WARNING: time words alone do NOT override Rule #1. If "on my way" + "this evening" both appear → mode=route, not point.
   - anchor = "home" if explicit "near home"/"at home"/"close to home"
   - anchor = "work" if explicit "near work"/"at the office"
   - For implicit local queries like "good sushi for dinner", "doctor that can see me tonight", "supermarket this evening" → DEFAULT point + current

4. "none" — ONLY if query explicitly asks for city-wide / global search:
   - "best ramen in Tokyo", "anywhere in Japan", "all over the country"
   - This should be RARE. If unsure, choose "point" with anchor="current".

target.category and target.categories MUST be chosen ONLY from the AMap category list below.
You may ONLY output labels that appear verbatim in the lists below — nothing else exists in the database.

★ TWO-LEVEL RULE:
Some categories have SUBTYPES (regional cuisines). When the need names a specific cuisine, output the
SUBTYPE **and its PARENT** (many places are only tagged with the parent). When generic, output the PARENT.
  "I want hotpot"        -> ["hotpot_restaurant", "chinese_restaurant"]
  "some Sichuan food"    -> ["sichuan_restaurant", "chinese_restaurant"]
  "sushi / Japanese"     -> ["japanese_restaurant", "foreign_restaurant"]
  "just some Chinese"    -> ["chinese_restaurant"]
  "grab fast food"       -> ["fast_food_restaurant"]
  "a coffee"             -> ["cafe"]

CATEGORIES WITH SUBTYPES (output subtype + parent when the cuisine is specific):
  chinese_restaurant  [subtypes: anhui_restaurant, beijing_restaurant, cantonese_restaurant, chaozhou_restaurant, chinese_vegetarian_restaurant, fujian_restaurant, halal_restaurant, hotpot_restaurant, hubei_restaurant, hunan_restaurant, jiangsu_restaurant, northeastern_restaurant, northwestern_restaurant, seafood_restaurant, shandong_restaurant, shanghai_restaurant, sichuan_restaurant, taiwanese_restaurant, time_honored_restaurant, yunnan_guizhou_restaurant, zhejiang_restaurant]
  foreign_restaurant  [subtypes: american_restaurant, brazilian_restaurant, british_restaurant, french_restaurant, german_restaurant, indian_restaurant, italian_restaurant, japanese_restaurant, korean_restaurant, mediterranean_restaurant, mexican_restaurant, other_asian_restaurant, portuguese_restaurant, russian_restaurant, steakhouse, thai_vietnamese_restaurant, western_restaurant]

FLAT CATEGORIES (use as-is):
agency, airport, art_gallery, atm, auto_parts_store, baby_service, bakery, bank, bathhouse, car_maintenance, car_rental, car_repair, car_wash, charging_station, cinema_theater, clinic, clothing_store, cold_drink_shop, company, convenience_store, convention_center, cosmetics_store, courier_service, dessert_shop, driving_school, electronics_store, exhibition_hall, factory, flower_pet_market, gas_station, general_market, government_office, hair_salon, home_building_market, hostel, hotel, industrial_park, info_center, insurance_company, job_market, laundry, law_enforcement, leisure_venue, lottery_outlet, media_agency, office_building, park, photo_studio, post_office, repair_shop, research_institute, residential_area, resort, school, shopping_mall, social_organization, specialized_hospital, specialty_store, sporting_goods_store, sports_venue, stationery_store, supermarket, tax_office, tea_house, tourist_attraction, train_station, training_institution, travel_agency

MAPPING RULE: if the user's need is phrased with a word NOT in the lists, map it to the CLOSEST listed
label — do NOT output the unlisted word. Examples of needs that must be mapped:
  spa / massage / sauna / 养生   -> bathhouse
  makeup / nail / hair / 美甲     -> hair_salon
  gym / fitness / yoga            -> sports_venue
  parcel / shipping / express     -> courier_service
  museum / exhibition / gallery   -> exhibition_hall  (or art_gallery)
  printing / photo / 冲印         -> photo_studio
  phone/computer repair / 维修    -> repair_shop
IMPORTANT: do NOT default to "casual_dining" for ordinary meals. casual_dining is ONLY for genuinely casual/light venues (snack bars, diners). For a normal Chinese meal use "chinese_restaurant" (+ cuisine subtype if specific); for fast food use "fast_food_restaurant"; for coffee use "cafe".
Output null ONLY if the query names no place-type at all (pure chit-chat). If the query mentions ANY kind of venue/shop/service (bar, cafe, locker, restaurant, salon...), you MUST map it to the closest listed category — NEVER output null just because the wording is indirect. Never output a label that is not printed above.

Always emit a "categories" array (1-4 entries); copy categories[0] into the "category" field.
If nothing fits, output null.

temporal.time_bucket: one of [breakfast, lunch, afternoon, dinner, evening, late_night, morning].
temporal.day_hint: Monday/Tuesday/.../weekend if mentioned.

evidence_requirements rules:
- mode="point" → ALWAYS include {{"type":"point","name":"<anchor>_location","arguments":{{"anchor":"<anchor>"}}}}
- mode="route" → include {{"type":"route","name":"<dir>_route","arguments":{{"direction":"<dir>"}}}}
- mode="zone" → include {{"type":"zone","name":"<zone>","arguments":{{"zone_name":"<zone>","time_bucket":<bucket>}}}}
- If target.category set → also include preference evidence: {{"type":"preference","name":"category_preference","arguments":{{"category":<cat>,"top_n":{PREF_TOP_N}}}}}

Output ONLY the JSON object. No markdown."""


_INTENT_CACHE = {}
_NORM_QUERY = [""]



def _smart_anchor_from_temporal(intent: dict, query: str) -> str:
    q = query.lower()
    bucket = (intent.get("temporal", {}).get("time_bucket") or "").lower()
    day = (intent.get("temporal", {}).get("day_hint") or "").lower()
    is_weekend = "weekend" in day or "saturday" in day or "sunday" in day or "weekend" in q
    if bucket in {"dinner", "evening", "late_night", "night"} or "tonight" in q or "evening" in q:
        return "home"
    if not is_weekend and bucket in {"morning", "lunch", "afternoon", "weekday_morning", "weekday_lunch"}:
        return "work"
    return "current"

def parse_intent(query: str) -> dict:
    if query in _INTENT_CACHE:
        return _INTENT_CACHE[query]

    try:
        resp = _client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=1200,
            timeout=60,
            response_format={"type": "json_object"},
            **_chat_kwargs(LLM_MODEL),
            messages=[
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        intent = json.loads(_strip_md_fence(resp.choices[0].message.content))
        _NORM_QUERY[0] = query
        intent = _normalize(intent, query)
    except Exception as e:
        print(f"[parser] LLM failure: {str(e)[:100]} → fallback")
        intent = _fallback_intent(query)

    _INTENT_CACHE[query] = intent
    return intent


def _normalize(intent: dict, query: str) -> dict:
    intent.setdefault("target", {})
    intent["target"].setdefault("entity_type", "poi")
    intent["target"].setdefault("category", None)
    intent["target"].setdefault("categories", [])
    intent["target"].setdefault("keywords", [])
    _cats = intent["target"].get("categories") or []
    if not isinstance(_cats, list):
        _cats = [str(_cats)]
    _cats = [str(c).strip() for c in _cats if c]
    _primary = intent["target"].get("category")
    if _primary and _primary not in _cats:
        _cats = [_primary] + _cats
    elif not _primary and _cats:
        intent["target"]["category"] = _cats[0]
    intent["target"]["categories"] = _cats

    intent.setdefault("spatial", {})
    intent["spatial"].setdefault("mode", "none")
    intent["spatial"].setdefault("anchor", None)
    intent["spatial"].setdefault("route_direction", None)
    intent["spatial"].setdefault("raw_span", None)

    intent.setdefault("must_conditions", [])
    intent.setdefault("preference_conditions", [])
    intent.setdefault("evidence_requirements", [])
    intent.setdefault("scoring_objective", ["spatial", "temporal", "semantic", "preference"])

    if intent["spatial"].get("mode") not in {"route", "point", "zone", "none"}:
        intent["spatial"]["mode"] = "point"
    _sp = intent["spatial"]
    if _sp.get("route_direction") and _sp.get("mode") != "route":
        print(f"[parser] _normalize: route_direction='{_sp.get('route_direction')}' but mode='{_sp.get('mode')}' → forcing mode=route")
        _sp["mode"] = "route"
        if "spatial" not in intent.get("scoring_objective", []):
            intent.setdefault("scoring_objective", []).append("spatial")
    if intent["spatial"].get("mode") == "none":
        intent["spatial"]["mode"] = "point"
        intent["spatial"]["anchor"] = _smart_anchor_from_temporal(intent, _NORM_QUERY[0])
    if intent["spatial"].get("mode") == "point" and not intent["spatial"].get("anchor"):
        intent["spatial"]["anchor"] = _smart_anchor_from_temporal(intent, _NORM_QUERY[0])

    mode = intent["spatial"]["mode"]
    have_types = {r.get("type") for r in intent.get("evidence_requirements", [])}

    if mode == "point" and "point" not in have_types:
        anchor = intent["spatial"].get("anchor") or "current"
        intent["evidence_requirements"].append(
            {"type": "point", "name": f"{anchor}_location", "arguments": {"anchor": anchor}}
        )
    elif mode == "route" and "route" not in have_types:
        direction = intent["spatial"].get("route_direction") or "home_to_work"
        intent["spatial"]["route_direction"] = direction
        intent["evidence_requirements"].append(
            {"type": "route", "name": f"{direction}_route", "arguments": {"direction": direction}}
        )
    elif mode == "zone" and "zone" not in have_types:
        bucket = _infer_time_bucket(intent) or "lunch"
        zname = intent["spatial"].get("zone_name") or None
        zargs = {"time_bucket": bucket}
        if zname:
            zargs["zone_name"] = zname
        intent["evidence_requirements"].append(
            {"type": "zone", "name": "activity_zone", "arguments": zargs}
        )

    cat = intent.get("target", {}).get("category")
    if cat and "preference" not in have_types:
        intent["evidence_requirements"].append(
            {"type": "preference", "name": "category_preference", "arguments": {"category": cat, "top_n": PREF_TOP_N}}
        )

    return intent


def _infer_time_bucket(intent: dict) -> str | None:
    for c in intent.get("must_conditions", []):
        if c.get("field") == "opening_hours" and c.get("value"):
            return str(c["value"])
    for c in intent.get("preference_conditions", []):
        v = str(c.get("value", "")).lower()
        for b in ["breakfast", "lunch", "dinner", "evening", "morning", "afternoon", "late_night"]:
            if b in v:
                return b
    return None


def _fallback_intent(query: str) -> dict:
    return {
        "target": {"entity_type": "poi", "category": None, "keywords": []},
        "spatial": {"mode": "none", "anchor": None, "route_direction": None, "raw_span": None},
        "must_conditions": [],
        "preference_conditions": [],
        "evidence_requirements": [],
        "scoring_objective": ["semantic"],
    }
