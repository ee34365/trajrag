# TrajRAG

Method code for **TrajRAG**: trajectory retrieval-augmented geospatial query answering
(event-level mobility representation → intent grounding → deterministic geometric
retrieval → entropy-weighted multi-signal scoring).

The two evaluation corpora (**Takeout**, Google-Places taxonomy / JP brands; and
**GeoLife**, AMap taxonomy / CN brands) run **one shared algorithm** — only the dataset
lexicons differ (brand dictionary, category whitelist / aliases, prompt localization).
They are selected at run time with a `--dataset` / `TRAJRAG_DATASET` switch.

> Datasets and POI stores are **not** included (size / licensing). Set the required API
> keys via environment variables (below).

## Layout

```
trajrag_env.py                 # dataset switch: setup("takeout"|"geolife")
trajrag/
  core/                        # shared, identical for both datasets
    scoring.py                 #   per-candidate signals (spa/tem/sem/pref) + EWM fusion
    profile.py profile_compact.py
    utils.py llm_rerank.py
  datasets/
    takeout/                   # dataset-specific (Google types, JP brands)
      constants.py parser.py evidence.py meta_planner.py candidate.py pipeline.py
      anchor_v2_module.py  eval_v5.py            # Takeout eval entry
    geolife/                   # dataset-specific (AMap types, CN brands)
      constants.py parser.py evidence.py meta_planner.py candidate.py pipeline.py
weighting.py                   # entropy-weight fusion used for the reported numbers
dump_signals.py                # run full pipeline, dump per-candidate signals
final_326.py final_326_newbl.py abl_now.py       # main table / ablation aggregation
eval_baselines_v4.py           # baselines (SD, ST, Naive RAG, GeoLLM, Spatial-RAG, SemaSK, ...)
figures/fig2_overview.pdf
```

Reproduction path for the reported numbers: `dump_signals.py` runs the full pipeline
(`pipeline.score_query`, via the `trajrag_env` dataset switch) and caches per-candidate
signals; `weighting.py` fuses them (`W.evaluate(recs, "ewm", 0.5)`); `final_326.py` /
`final_326_newbl.py` / `abl_now.py` aggregate into the paper's tables. Data paths default
to `./data/...` (see `## Environment`) — point them at your own copies of the two corpora
and POI stores.

`core` vs `datasets` diff: `scoring.py` differs only by the brand dictionary (the shared
copy carries both JP+CN brands); `profile.py`/`utils.py` are byte-identical; the six
`datasets/*` modules hold the per-corpus taxonomy and prompts.

## Usage

```python
import trajrag_env
trajrag_env.setup("geolife")            # or "takeout"; or export TRAJRAG_DATASET=geolife
from pipeline import score_query, build_conn_from_traj   # resolves to the chosen dataset
```

Command-line scripts honour `TRAJRAG_DATASET` (e.g. `dump_signals.py`) or take
`--dataset {takeout,geolife}` (e.g. `eval_baselines_v4.py`).

## Environment

```bash
export OPENAI_API_KEY="..."        # LLM + embeddings (or OpenAI-compatible endpoint)
export OPENAI_BASE_URL="..."       # optional
export ANTHROPIC_API_KEY="..."     # optional, some baselines
export TRAJRAG_DATASET=takeout     # or geolife
```

Encoder: `intfloat/multilingual-e5-base`. Backbone: `gpt-4o-mini`.
Spatial execution uses DuckDB (`INSTALL spatial; LOAD spatial;`).

## Scoring

Four bounded signals per candidate — spatial, temporal, semantic, preference — with the
category match folded into the semantic signal (`α=0.5`), fused by the Entropy Weight
Method (per-query weights derived from the candidate set) and ranked by
`S(p) = Σ_g w_g · s̃_g(p)`.
