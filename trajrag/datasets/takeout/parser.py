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

_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()
import os as _os_mt
_MAXTOK = int(_os_mt.environ.get("TRAJRAG_MAXTOK_INTENT", "700"))


import os as _os_nr
_NO_REASON_KW = ({"extra_body": {"reasoning": {"enabled": False}}}
                 if _os_nr.environ.get("TRAJRAG_NO_REASONING") else {})



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

2. "zone" — query mentions a Japanese named region (any of the below triggers):
   - 区 / Ward (e.g. "Naka Ward", "Setagaya")
   - 市 / City (e.g. "Ishioka", "Funabashi", "Kyoto")
   - 県 / Prefecture (e.g. "Aichi", "Hyogo")
   - 府 / 都 (e.g. "Osaka", "Tokyo")
   - "X area" (e.g. "Ichinomiya area", "Takasago area")
   - "in / around / at / near + named place"
   Whenever ANY of these trigger, set spatial_mode="zone" and put the
   recognized place name in zone_name (use the English form as it appears in the query).

3. "point" (FALLBACK — only when neither route nor zone applies):
   - anchor = "current" — when query has time word ("tonight", "this afternoon", "this evening", "today", "later", "now", "right now") AND no route phrase from Rule #1
   - WARNING: time words alone do NOT override Rule #1. If "on my way" + "this evening" both appear → mode=route, not point.
   - anchor = "home" if explicit "near home"/"at home"/"close to home"
   - anchor = "work" if explicit "near work"/"at the office"
   - For implicit local queries like "good sushi for dinner", "doctor that can see me tonight", "supermarket this evening" → DEFAULT point + current

4. "none" — ONLY if query explicitly asks for city-wide / global search:
   - "best ramen in Tokyo", "anywhere in Japan", "all over the country"
   - This should be RARE. If unsure, choose "point" with anchor="current".

target.category and target.categories MUST be selected from this whitelist (Google Places primaryType).

CRITICAL RULE: Always emit a "categories" array with 1-4 entries representing PLAUSIBLE TYPES for the query:
- categories[0] = your top guess (also copy to "category" field for backward compat)
- categories[1..N] = alternate types the user might also accept (annotators may have labeled differently)

EXAMPLES:
  "quiet place to read"     → categories=["library","cafe","book_store","park"]
  "breakfast spot"          → categories=["cafe","fast_food_restaurant","bakery","restaurant"]
  "dermatology clinic"      → categories=["skin_care_clinic","doctor","hospital","dental_clinic"]
  "dental checkup"          → categories=["dentist","dental_clinic","doctor","hospital"]
  "supermarket for grocery" → categories=["supermarket","grocery_store","food_store","convenience_store"]
  "historic site"           → categories=["historical_landmark","historical_place","tourist_attraction","museum"]
  "convenience store"       → categories=["convenience_store"]
  "gas station"             → categories=["gas_station"]

Each entry MUST be from this whitelist (Google Places primaryType):
convenience_store, shopping_mall, train_station, japanese_restaurant, supermarket, premise, drugstore, grocery_store, rest_stop, restaurant, park, discount_store, hotel, subway_station, ramen_restaurant, bus_stop, gas_station, place_of_worship, store, home_improvement_store, post_office, transit_station, cafe, fast_food_restaurant, clothing_store, condominium_complex, bank, sushi_restaurant, hospital, chinese_restaurant, school, electronics_store, bakery, corporate_office, community_center, doctor, car_dealer, government_office, city_hall, food_store, parking, book_store, museum, hair_salon, department_store, local_government_office, university, market, pharmacy, library, coffee_shop, primary_school, public_bath, wholesaler, italian_restaurant, dessert_shop, cemetery, car_repair, home_goods_store, confectionery, dental_clinic, event_venue, tourist_attraction, barber_shop, steak_house, auto_parts_store, dentist, bar, sporting_goods_store, japanese_inn, athletic_field, amusement_center, liquor_store, bicycle_store, sports_activity_location, real_estate_agency, sports_complex, beauty_salon, concert_hall, meal_takeaway, cell_phone_store, general_contractor, karaoke, hamburger_restaurant, consultant, preschool, natural_feature, furniture_store, spa, seafood_restaurant, golf_course, internet_cafe, gym, apartment_building, airport, storage, pizza_restaurant, shoe_store, historical_landmark, ferry_terminal, funeral_home, convention_center, international_airport, bus_station, fire_station, aquarium, zoo, indian_restaurant, farm, dessert_restaurant, ranch, stadium, car_rental, butcher_shop, historical_place, pet_store, playground, atm, asian_grocery_store, apartment_complex, korean_restaurant, electrician, thai_restaurant, resort_hotel, donut_shop, veterinary_care, arena, campground, observation_deck, massage, amusement_park, ice_cream_shop, insurance_agency, finance, buffet_restaurant, video_arcade, wedding_venue, vietnamese_restaurant, pub, nail_salon, painter, botanical_garden, performing_arts_theater, yoga_studio, telecommunications_service_provider, water_park, car_wash, wellness_center, tourist_information_center, chocolate_shop, tea_house, sports_club, neighborhood_police_station, vegan_restaurant, rv_park, point_of_interest, food_court, philharmonic_hall, deli, lawyer, french_restaurant, cafeteria, warehouse_store, postal_code, catering_service, bagel_shop, swimming_pool, food, cultural_center, plaza, national_park, visitor_center, courier_service, lodging, hardware_store, public_bathroom, fishing_pond, tour_agency, sauna, mexican_restaurant, skin_care_clinic, meal_delivery, housing_complex, garden, health, ski_resort, hostel, turkish_restaurant, laundry, brazilian_restaurant, jewelry_store, florist
If query maps to no whitelist item, output null. Examples:
- "mall after pharmacy" → category="shopping_mall"
- "coffee on the way" → category="cafe"
- "quick bite" → category="fast_food_restaurant"
- "grab a snack" → category="convenience_store"
- "sushi for dinner" → category="sushi_restaurant"

temporal.time_bucket: one of [breakfast, lunch, afternoon, dinner, evening, late_night, morning].
temporal.day_hint: Monday/Tuesday/.../weekend if mentioned.

evidence_requirements rules:
- mode="point" → ALWAYS include {{"type":"point","name":"<anchor>_location","arguments":{{"anchor":"<anchor>"}}}}
- mode="route" → include {{"type":"route","name":"<dir>_route","arguments":{{"direction":"<dir>"}}}}
- mode="zone" → include {{"type":"zone","name":"<zone>","arguments":{{"zone_name":"<zone>","time_bucket":<bucket>}}}}
- If target.category set → also include preference evidence: {{"type":"preference","name":"category_preference","arguments":{{"category":<cat>,"top_n":{PREF_TOP_N}}}}}

Output ONLY the JSON object. No markdown."""


_INTENT_CACHE = {}
import threading as _pth
_PARSER_LOCK = _pth.Lock()



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
    import os as _os
    if _os.environ.get("TRAJRAG_ABLATE", "").lower() == "noplanner":
        _it = _fallback_intent(query)
        _it["query"] = query
        return _it
    with _PARSER_LOCK:
        if query in _INTENT_CACHE:
            return _INTENT_CACHE[query]

    try:
        resp = _client.chat.completions.create(
            **_NO_REASON_KW,
            model=LLM_MODEL,
            temperature=0.0,
            max_tokens=_MAXTOK,
            timeout=20,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": INTENT_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        _raw = resp.choices[0].message.content or ""
        intent = json.loads(_strip_md_fence(_raw))
        if not isinstance(intent, dict):
            raise TypeError(f"expected JSON object, got {type(intent).__name__}")
        for _k in ("target", "spatial", "temporal"):
            if _k in intent and not isinstance(intent[_k], dict):
                intent[_k] = {}
        intent = _normalize(intent, query)
    except Exception as e:
        _snip = (locals().get("_raw") or "")[:160].replace("\n", " ")
        print(f"[parser] LLM failure: {type(e).__name__}: {str(e)[:80]} "
              f"| model={LLM_MODEL} | raw[:160]={_snip!r} → fallback")
        intent = _fallback_intent(query)

    with _PARSER_LOCK:
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
        intent["spatial"]["anchor"] = _smart_anchor_from_temporal(intent, query)
    if intent["spatial"].get("mode") == "point" and not intent["spatial"].get("anchor"):
        intent["spatial"]["anchor"] = _smart_anchor_from_temporal(intent, query)

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


def _chat_kwargs(model=None, **kw):
    return {}
