import os

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL = os.environ.get("TRAJRAG_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("TRAJRAG_EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = os.environ.get("TRAJRAG_LOCAL_EMBED", "intfloat/multilingual-e5-base")
USE_LOCAL_EMBED = os.environ.get("TRAJRAG_USE_LOCAL_EMBED", "1") == "1"
EMBED_DIM = int(os.environ.get("TRAJRAG_EMBED_DIM", "768"))
BASE_WEIGHTS = {
    "route": {"spa": 0.40, "tem": 0.10, "sem": 0.35, "pref": 0.15},
    "point": {"spa": 0.45, "tem": 0.10, "sem": 0.35, "pref": 0.10},
    "zone":  {"spa": 0.30, "tem": 0.20, "sem": 0.35, "pref": 0.15},
    "none":  {"spa": 0.30, "tem": 0.15, "sem": 0.40, "pref": 0.15},
}

R_ROUTE_KM = 5.0
R_POINT_KM = 3.0
ZONE_ALPHA = 1.5

CANDIDATE_TOPK_TOTAL = int(os.environ.get("TRAJRAG_CANDIDATE_TOPK_TOTAL", 1000))
CANDIDATE_TOPK_SPATIAL = int(os.environ.get("TRAJRAG_CANDIDATE_TOPK_SPATIAL", 700))
CANDIDATE_TOPK_SEMANTIC = int(os.environ.get("TRAJRAG_CANDIDATE_TOPK_SEMANTIC", 700))
CANDIDATE_TOPK_PREFERENCE = int(os.environ.get("TRAJRAG_CANDIDATE_TOPK_PREFERENCE", 300))
ROUTE_PREFILTER_MULTIPLIER = 4


USE_LLM_RERANK = os.environ.get("TRAJRAG_USE_LLM_RERANK", "0").lower() in {"1", "true", "yes", "y"}
LLM_RERANK_TOP_K = int(os.environ.get("TRAJRAG_LLM_RERANK_TOP_K", 20))

PREF_TOP_N = 5
PREF_MIN_SUPPORT = 3

CATEGORY_ALIASES = {
    "cafe": ["cafe", "coffee_shop", "bakery", "tea_house"],
    "coffee_shop": ["cafe", "coffee_shop", "bakery", "tea_house"],
    "tea_house": ["tea_house", "cafe"],
    "restaurant": ["restaurant", "food", "meal_takeaway", "fast_food_restaurant", "japanese_restaurant", "chinese_restaurant", "italian_restaurant", "ramen_restaurant", "sushi_restaurant", "cafeteria"],
    "fast_food_restaurant": ["fast_food_restaurant", "restaurant", "meal_takeaway", "food", "convenience_store", "hamburger_restaurant"],
    "japanese_restaurant": ["japanese_restaurant", "restaurant", "ramen_restaurant", "sushi_restaurant", "japanese_inn"],
    "italian_restaurant": ["italian_restaurant", "pizza_restaurant", "restaurant"],
    "chinese_restaurant": ["chinese_restaurant", "restaurant"],
    "ramen_restaurant": ["ramen_restaurant", "japanese_restaurant", "restaurant"],
    "sushi_restaurant": ["sushi_restaurant", "japanese_restaurant", "restaurant"],
    "pizza_restaurant": ["pizza_restaurant", "italian_restaurant", "restaurant"],
    "korean_restaurant": ["korean_restaurant", "restaurant"],
    "vietnamese_restaurant": ["vietnamese_restaurant", "restaurant"],
    "thai_restaurant": ["thai_restaurant", "restaurant"],
    "indian_restaurant": ["indian_restaurant", "restaurant"],
    "vegan_restaurant": ["vegan_restaurant", "restaurant"],
    "mexican_restaurant": ["mexican_restaurant", "restaurant"],
    "buffet_restaurant": ["buffet_restaurant", "restaurant"],
    "hamburger_restaurant": ["hamburger_restaurant", "fast_food_restaurant", "restaurant"],
    "steak_house": ["steak_house", "restaurant"],
    "seafood_restaurant": ["seafood_restaurant", "restaurant"],
    "deli": ["deli", "restaurant", "meal_takeaway"],
    "supermarket": ["supermarket", "grocery_store", "food_store", "convenience_store", "discount_store", "market"],
    "grocery_store": ["grocery_store", "supermarket", "food_store", "convenience_store", "market"],
    "food_store": ["food_store", "supermarket", "grocery_store", "market"],
    "convenience_store": ["convenience_store", "supermarket", "grocery_store", "fast_food_restaurant", "meal_takeaway"],
    "discount_store": ["discount_store", "supermarket", "grocery_store"],
    "market": ["market", "supermarket", "grocery_store", "food_court"],
    "doctor": ["doctor", "hospital", "dental_clinic", "dentist", "skin_care_clinic", "medical_lab", "wellness_center", "veterinary_care", "health"],
    "hospital": ["hospital", "doctor", "medical_lab", "health", "dental_clinic"],
    "dentist": ["dentist", "dental_clinic", "doctor", "hospital"],
    "dental_clinic": ["dental_clinic", "dentist", "doctor", "hospital"],
    "skin_care_clinic": ["skin_care_clinic", "doctor", "hospital", "spa", "beauty_salon"],
    "medical_lab": ["medical_lab", "doctor", "hospital"],
    "veterinary_care": ["veterinary_care", "doctor", "pet_store"],
    "pharmacy": ["pharmacy", "drugstore", "doctor"],
    "drugstore": ["drugstore", "pharmacy", "convenience_store"],
    "beauty_salon": ["beauty_salon", "hair_salon", "spa", "nail_salon", "barber_shop"],
    "hair_salon": ["hair_salon", "beauty_salon", "barber_shop"],
    "barber_shop": ["barber_shop", "hair_salon", "beauty_salon"],
    "nail_salon": ["nail_salon", "beauty_salon"],
    "spa": ["spa", "beauty_salon", "sauna", "wellness_center", "massage"],
    "gym": ["gym", "yoga_studio", "sports_complex", "sports_activity_location", "wellness_center", "athletic_field"],
    "yoga_studio": ["yoga_studio", "gym", "wellness_center"],
    "shopping_mall": ["shopping_mall", "department_store", "store", "discount_store"],
    "department_store": ["department_store", "shopping_mall", "store"],
    "clothing_store": ["clothing_store", "shoe_store", "department_store", "store"],
    "electronics_store": ["electronics_store", "store", "cell_phone_store", "home_improvement_store"],
    "book_store": ["book_store", "library", "store"],
    "bookstore": ["book_store", "library", "store"],
    "library": ["library", "book_store"],
    "gas_station": ["gas_station", "car_repair", "auto_parts_store"],
    "train_station": ["train_station", "subway_station", "bus_station", "transit_station"],
    "subway_station": ["subway_station", "train_station", "transit_station"],
    "parking": ["parking", "rv_park"],
    "park": ["park", "tourist_attraction", "garden", "playground", "botanical_garden", "national_park", "natural_feature", "plaza"],
    "garden": ["garden", "park", "botanical_garden"],
    "tourist_attraction": ["tourist_attraction", "park", "tourist_information_center", "museum", "historical_landmark"],
    "bar": ["bar", "pub", "night_club", "karaoke"],
    "pub": ["pub", "bar"],
    "karaoke": ["karaoke", "bar", "pub", "video_arcade", "amusement_center"],
    "amusement_center": ["amusement_center", "video_arcade", "karaoke", "amusement_park"],
    "hotel": ["hotel", "lodging", "resort_hotel", "japanese_inn", "hostel"],
    "lodging": ["lodging", "hotel", "japanese_inn", "hostel"],
    "bank": ["bank", "atm", "finance"],
    "atm": ["atm", "bank"],
    "post_office": ["post_office", "courier_service"],
    "car_dealer": ["car_dealer", "car_repair", "auto_parts_store", "car_rental"],
    "car_repair": ["car_repair", "gas_station", "auto_parts_store"],
}

TIME_BUCKETS = {
    "breakfast":         (6, 10),
    "lunch":             (11, 14),
    "afternoon":         (14, 18),
    "dinner":            (17, 21),
    "evening":           (18, 22),
    "late_night":        (22, 30),
    "morning":           (6, 12),
    "weekday_morning":   (6, 10),
    "weekday_lunch":     (11, 14),
    "weekday_afternoon": (14, 18),
    "weekday_evening":   (18, 23),
    "weekend_morning":   (6, 12),
    "weekend_afternoon": (12, 18),
    "weekend_evening":   (18, 23),
    "open_now":          None,
}
