from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from constants import OPENAI_BASE_URL

from constants import OPENAI_KEY, LLM_MODEL
from parser import _chat_kwargs  

_client = OpenAI(api_key=OPENAI_KEY, base_url=OPENAI_BASE_URL) if OPENAI_BASE_URL else OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else OpenAI()


SYSTEM_PROMPT = """You are a careful POI reranker for personalized geospatial retrieval.

You will receive:
1. A user query.
2. A compact trajectory-evidence summary.
3. A numbered list of candidate POIs that were already ranked by deterministic scoring.

Your task:
- Select exactly ONE candidate that best answers the query.
- Use the candidate information and trajectory evidence only.
- Do not invent new POIs.
- Do not select a candidate not in the numbered list.
- Return STRICT JSON only.

Output schema:
{
  "selected_index": integer,
  "reason": string
}

The selected_index must be 1-based, matching the numbered list.
Keep the reason concise."""


def _safe_str(x: Any, max_len: int = 160) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def _summarize_evidence(R_tau: Dict[str, Any]) -> str:
    parts: List[str] = []

    point = R_tau.get("point") if isinstance(R_tau, dict) else None
    if point:
        parts.append(
            f"Point anchor: {point.get('anchor_name','point')} "
            f"at ({point.get('lat')}, {point.get('lon')})."
        )

    route = R_tau.get("route") if isinstance(R_tau, dict) else None
    if route:
        n = len(route.get("points") or [])
        parts.append(f"Route evidence: {route.get('route_name','route')} with {n} waypoints; source={route.get('source','unknown')}.")

    zone = R_tau.get("zone") if isinstance(R_tau, dict) else None
    if zone:
        parts.append(
            f"Zone evidence: {zone.get('zone_name','zone')} centered at "
            f"({zone.get('center_lat')}, {zone.get('center_lon')}) with radius {zone.get('radius_km')} km "
            f"from {zone.get('support_size','unknown')} visits."
        )

    pref = R_tau.get("preference") if isinstance(R_tau, dict) else None
    if pref:
        desc = _safe_str(pref.get("anonymized_description"), 220)
        if desc:
            parts.append(f"Preference evidence: {desc}")

    return "\n".join(parts) if parts else "No trajectory evidence is available."


def _candidate_line(i: int, p: Dict[str, Any]) -> str:
    name = _safe_str(p.get("name"), 80)
    cat = _safe_str(p.get("category") or p.get("primaryType"), 60)
    addr = _safe_str(p.get("address"), 120)
    desc = _safe_str(p.get("description"), 120)
    score = p.get("_score")
    score_s = f"{float(score):.4f}" if isinstance(score, (int, float)) else "NA"
    signals = p.get("_signals", {})
    sig_s = ", ".join(f"{k}={float(v):.3f}" for k, v in signals.items() if isinstance(v, (int, float)))
    return (
        f"{i}. name={name} | category={cat} | address={addr} | "
        f"description={desc} | deterministic_score={score_s} | signals=[{sig_s}]"
    )


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def llm_rerank(
    query: str,
    R_tau: Dict[str, Any],
    ranked: List[Dict[str, Any]],
    top_k: int = 20,
    model: str = LLM_MODEL,
    timeout: int = 30,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    if not ranked or top_k <= 1:
        return ranked

    top = ranked[: min(top_k, len(ranked))]
    evidence_summary = _summarize_evidence(R_tau)
    cand_text = "\n".join(_candidate_line(i, p) for i, p in enumerate(top, 1))

    user_prompt = f"""User query:
{query}

Trajectory evidence:
{evidence_summary}

Candidate POIs:
{cand_text}

Select the single best candidate. Return strict JSON only."""

    try:
        resp = _client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=180,
            timeout=timeout,
            response_format={"type": "json_object"},
            **_chat_kwargs(model),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        data = _parse_json_response(resp.choices[0].message.content or "")
        selected = int(data.get("selected_index", 0))
        reason = _safe_str(data.get("reason"), 300)
    except Exception as e:
        if verbose:
            print(f"[LLM rerank] failed: {str(e)[:120]}")
        return ranked

    if selected < 1 or selected > len(top):
        if verbose:
            print(f"[LLM rerank] invalid selected_index={selected}")
        return ranked

    selected_zero = selected - 1
    chosen = dict(top[selected_zero])
    chosen["_llm_rerank_selected"] = True
    chosen["_llm_rerank_reason"] = reason
    chosen["_llm_rerank_original_rank"] = selected

    new_top = [chosen] + [p for j, p in enumerate(top) if j != selected_zero]
    out = new_top + ranked[len(top):]
    if verbose:
        print(f"[LLM rerank] selected top-{selected}: {chosen.get('name','?')} | reason={reason}")
    return out
