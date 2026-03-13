"""
External data sources used to estimate the "true" probability of a market.

Sources implemented:
 1. NewsAPI        — recent headline sentiment & event evidence
 2. Metaculus      — crowd probability from another prediction market
 3. Manifold       — another prediction market for cross-reference
 4. Wikipedia      — background context for the LLM
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import requests
from loguru import logger

from config import settings


# ── Domain models ──────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    description: str
    source: str
    published_at: str
    url: str


@dataclass
class ExternalEvidence:
    """All evidence collected for a single market question."""
    question: str
    news_items: list[NewsItem] = field(default_factory=list)
    metaculus_probability: Optional[float] = None
    manifold_probability: Optional[float] = None
    wikipedia_summary: Optional[str] = None

    def to_context_string(self) -> str:
        """Serialise evidence into a text block for the LLM prompt."""
        lines: list[str] = [f"## Market question\n{self.question}\n"]

        if self.metaculus_probability is not None:
            lines.append(f"### Metaculus crowd probability\n{self.metaculus_probability:.1%}\n")

        if self.manifold_probability is not None:
            lines.append(f"### Manifold Markets probability\n{self.manifold_probability:.1%}\n")

        if self.news_items:
            lines.append("### Recent news headlines")
            for item in self.news_items[:8]:
                lines.append(f"- [{item.source}] {item.title}: {item.description or ''}")
            lines.append("")

        if self.wikipedia_summary:
            lines.append(f"### Wikipedia background\n{self.wikipedia_summary}\n")

        return "\n".join(lines)


# ── Source clients ─────────────────────────────────────────────────────────────

class NewsAPIClient:
    """Fetches recent news from newsapi.org."""

    BASE = "https://newsapi.org/v2/everything"

    def __init__(self) -> None:
        self._key = settings.news_api_key.get_secret_value()
        self._session = requests.Session()

    def fetch(self, query: str, max_articles: int = 10) -> list[NewsItem]:
        if not self._key:
            logger.debug("NEWS_API_KEY not set — skipping NewsAPI")
            return []

        params = {
            "q": self._truncate_query(query),
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": max_articles,
            "apiKey": self._key,
        }
        try:
            resp = self._session.get(self.BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            return [
                NewsItem(
                    title=a.get("title", ""),
                    description=a.get("description", ""),
                    source=a.get("source", {}).get("name", ""),
                    published_at=a.get("publishedAt", ""),
                    url=a.get("url", ""),
                )
                for a in articles
                if a.get("title")
            ]
        except Exception as exc:
            logger.warning("NewsAPI fetch failed for '{}': {}", query[:60], exc)
            return []

    @staticmethod
    def _truncate_query(q: str, max_len: int = 100) -> str:
        """NewsAPI has a query length limit; keep the most important words."""
        words = re.sub(r"[^\w\s]", "", q).split()
        result = ""
        for w in words:
            if len(result) + len(w) + 1 > max_len:
                break
            result = f"{result} {w}".strip()
        return result or q[:max_len]


class MetaculusClient:
    """Fetches community probability from Metaculus."""

    BASE = "https://www.metaculus.com/api2/questions/"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "polymarket-bot/1.0"

    def find_probability(self, question: str) -> Optional[float]:
        """
        Search Metaculus for the most relevant question and return its
        community probability.  Returns None if no match found.
        """
        params = {
            "search": question[:100],
            "status": "open",
            "order_by": "-activity",
            "limit": 3,
        }
        try:
            resp = self._session.get(self.BASE, params=params, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None

            # Pick the first result that has a community prediction
            for r in results:
                cp = (
                    r.get("community_prediction", {})
                    .get("full", {})
                    .get("q2")
                )
                if cp is not None:
                    logger.debug("Metaculus match: '{}' → {:.1%}", r.get("title", "?"), cp)
                    return float(cp)
        except Exception as exc:
            logger.warning("Metaculus lookup failed: {}", exc)
        return None


class ManifoldClient:
    """Fetches market probability from Manifold Markets."""

    BASE = "https://api.manifold.markets/v0"

    def __init__(self) -> None:
        self._session = requests.Session()

    def find_probability(self, question: str) -> Optional[float]:
        """Search Manifold and return the probability of the best matching market."""
        try:
            resp = self._session.get(
                f"{self.BASE}/search-markets",
                params={"term": question[:100], "limit": 3},
                timeout=10,
            )
            resp.raise_for_status()
            markets = resp.json()
            for m in markets:
                if m.get("outcomeType") == "BINARY" and m.get("probability") is not None:
                    prob = float(m["probability"])
                    logger.debug("Manifold match: '{}' → {:.1%}", m.get("question", "?"), prob)
                    return prob
        except Exception as exc:
            logger.warning("Manifold lookup failed: {}", exc)
        return None


class WikipediaClient:
    """Fetches a Wikipedia article summary for background context."""

    BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "polymarket-bot/1.0"

    def get_summary(self, topic: str) -> Optional[str]:
        """Return the extract for the closest matching Wikipedia article."""
        # Use the search API to find the best article title first
        try:
            search_resp = self._session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": topic[:100],
                    "format": "json",
                    "srlimit": 1,
                },
                timeout=10,
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("query", {}).get("search", [])
            if not results:
                return None

            title = results[0]["title"]
            summary_resp = self._session.get(
                f"{self.BASE}/{requests.utils.quote(title)}",
                timeout=10,
            )
            summary_resp.raise_for_status()
            extract = summary_resp.json().get("extract", "")
            # Truncate to avoid bloating the LLM context
            return extract[:1200] if extract else None
        except Exception as exc:
            logger.warning("Wikipedia lookup failed for '{}': {}", topic[:60], exc)
            return None


# ── Aggregator ─────────────────────────────────────────────────────────────────

class DataAggregator:
    """
    Collects evidence from all available sources for a single market question.
    Failures in individual sources are silently swallowed so one broken
    integration never stops the whole pipeline.
    """

    def __init__(self) -> None:
        self._news = NewsAPIClient()
        self._metaculus = MetaculusClient()
        self._manifold = ManifoldClient()
        self._wikipedia = WikipediaClient()

    def collect(self, question: str, description: str = "") -> ExternalEvidence:
        evidence = ExternalEvidence(question=question)

        # Use both the question and the description for news search
        search_query = question if len(question) <= 120 else question[:120]

        evidence.news_items = self._news.fetch(search_query)
        evidence.metaculus_probability = self._metaculus.find_probability(question)
        evidence.manifold_probability = self._manifold.find_probability(question)

        # Derive a short topic from the question for Wikipedia
        topic = self._extract_topic(question)
        evidence.wikipedia_summary = self._wikipedia.get_summary(topic)

        logger.debug(
            "Evidence for '{}': {} news, metaculus={}, manifold={}, wiki={}",
            question[:60],
            len(evidence.news_items),
            f"{evidence.metaculus_probability:.1%}" if evidence.metaculus_probability else "N/A",
            f"{evidence.manifold_probability:.1%}" if evidence.manifold_probability else "N/A",
            "yes" if evidence.wikipedia_summary else "no",
        )
        return evidence

    @staticmethod
    def _extract_topic(question: str) -> str:
        """
        Heuristic: strip common question prefixes and return the core noun phrase
        so Wikipedia searches are more accurate.
        """
        prefixes = (
            "will ", "who will ", "when will ", "what will ",
            "how many ", "does ", "is ", "are ", "was ",
        )
        q = question.strip().rstrip("?")
        lower = q.lower()
        for pref in prefixes:
            if lower.startswith(pref):
                q = q[len(pref):]
                break
        return q[:80]
