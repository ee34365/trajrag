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

CATEGORY_ALIASES = {}

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
