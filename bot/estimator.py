"""
Probability estimator.

Uses GPT-4o to combine all gathered evidence and produce a calibrated
probability estimate for a binary market outcome (YES / NO).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore[assignment,misc]

from config import settings
from data.sources import ExternalEvidence


# ── Result model ───────────────────────────────────────────────────────────────

@dataclass
class ProbabilityEstimate:
    """Output of one LLM estimation pass."""
    yes_probability: float          # 0.0 – 1.0
    confidence: str                 # "low" | "medium" | "high"
    reasoning: str
    sources_used: list[str]

    @property
    def no_probability(self) -> float:
        return 1.0 - self.yes_probability


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are an expert probabilistic forecaster with deep knowledge in geopolitics,
economics, sports, science, and current events. Your task is to estimate the
probability that a binary prediction market question resolves YES.

Rules:
1. Carefully read all provided evidence (news, other prediction market probabilities,
   Wikipedia background).
2. Weight evidence by recency and reliability. Prefer hard data over speculation.
3. Account for base rates and reference classes.
4. Your estimate must be a number strictly between 0.01 and 0.99 — avoid extreme
   probabilities unless the evidence is overwhelming.
5. Return ONLY a JSON object with exactly these keys:
   {
     "yes_probability": <float 0.01-0.99>,
     "confidence": "<low|medium|high>",
     "reasoning": "<2-4 sentence explanation>",
     "sources_used": ["news", "metaculus", "manifold", "wikipedia"]
     // include only the sources that meaningfully influenced your estimate
   }
6. Do NOT include any prose outside the JSON object.
""".strip()


# ── Estimator ──────────────────────────────────────────────────────────────────

class ProbabilityEstimator:
    """
    Calls the OpenAI chat API to produce a probability estimate for a
    given evidence bundle.
    """

    def __init__(self) -> None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        groq_key = settings.groq_api_key.get_secret_value()
        openai_key = settings.openai_api_key.get_secret_value()

        if groq_key:
            # Groq: free tier, OpenAI-compatible API
            self._client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            self._model = settings.groq_model
            logger.info("LLM: using Groq ({}) — free tier", self._model)
        elif openai_key:
            self._client = OpenAI(api_key=openai_key)
            self._model = settings.openai_model
            logger.info("LLM: using OpenAI ({})", self._model)
        else:
            raise RuntimeError(
                "No LLM API key configured. Set GROQ_API_KEY (free) or OPENAI_API_KEY."
            )

        self._quota_exceeded = False

    def estimate(self, evidence: ExternalEvidence) -> Optional[ProbabilityEstimate]:
        """
        Returns a ProbabilityEstimate, or None if the LLM call fails.
        """
        # If quota was already exhausted this cycle, skip immediately
        if self._quota_exceeded:
            return None

        user_message = (
            f"{evidence.to_context_string()}\n\n"
            "Based on the evidence above, what is the probability that this "
            "market resolves YES? Return your answer strictly as JSON."
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,          # low temperature for calibration
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            return self._parse(raw, evidence.question)
        except Exception as exc:
            err_str = str(exc)
            if "insufficient_quota" in err_str or "429" in err_str:
                self._quota_exceeded = True
                logger.error(
                    "OpenAI quota exceeded — no credits in account. "
                    "Add credits at platform.openai.com/billing to enable AI analysis. "
                    "Skipping remaining markets this cycle."
                )
            else:
                logger.error("LLM call failed for '{}': {}", evidence.question[:60], exc)
            return None

    # ── Aggregated estimate (combines LLM + peer markets) ─────────────────────

    def aggregate(
        self,
        llm_estimate: ProbabilityEstimate,
        evidence: ExternalEvidence,
    ) -> float:
        """
        Blend the LLM estimate with peer-market signals using a simple
        weighted average.  Peer markets get lower weight when the LLM has
        high confidence; higher weight when confidence is low.
        """
        weights = {
            "llm": {"low": 0.50, "medium": 0.65, "high": 0.80},
        }
        llm_w = weights["llm"].get(llm_estimate.confidence, 0.65)
        remaining = 1.0 - llm_w

        peers = []
        if evidence.metaculus_probability is not None:
            peers.append(evidence.metaculus_probability)
        if evidence.manifold_probability is not None:
            peers.append(evidence.manifold_probability)

        if not peers:
            return llm_estimate.yes_probability

        peer_mean = sum(peers) / len(peers)
        blended = llm_w * llm_estimate.yes_probability + remaining * peer_mean

        logger.debug(
            "Blended probability for '{}': LLM={:.1%} peers={:.1%} → {:.1%}",
            evidence.question[:60],
            llm_estimate.yes_probability,
            peer_mean,
            blended,
        )
        return round(blended, 4)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str, question: str) -> Optional[ProbabilityEstimate]:
        try:
            data = json.loads(raw)
            prob = float(data["yes_probability"])
            prob = max(0.01, min(0.99, prob))   # clamp to safe range
            return ProbabilityEstimate(
                yes_probability=prob,
                confidence=data.get("confidence", "medium"),
                reasoning=data.get("reasoning", ""),
                sources_used=data.get("sources_used", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            # Try to salvage a number from the raw text as fallback
            match = re.search(r'"yes_probability"\s*:\s*([0-9.]+)', raw)
            if match:
                prob = max(0.01, min(0.99, float(match.group(1))))
                logger.warning(
                    "Partial LLM parse for '{}' — extracted {:.1%}", question[:60], prob
                )
                return ProbabilityEstimate(
                    yes_probability=prob,
                    confidence="low",
                    reasoning="Partial parse",
                    sources_used=[],
                )
            logger.error("Could not parse LLM response for '{}': {}", question[:60], exc)
            return None
