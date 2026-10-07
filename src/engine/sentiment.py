"""Sentiment in [-1, 1]. FinBERT when available, finance-lexicon fallback otherwise.
score = P(positive) - P(negative) for FinBERT."""
import re
POS = set("beat beats surge surges soar soars record growth profit profits gain gains upgrade upgraded strong rally rebound approve approved approval launch launches innovative expands boost bullish outperform win wins".split())
NEG = set("miss misses plunge plunges slump slumps loss losses default defaults downgrade downgraded weak crash collapse fraud probe lawsuit recall layoffs cuts bankruptcy bankrupt sanctions war invasion attack breach hack outage fear fears panic selloff bearish warn warns warning shortfall write-off writedown delisting".split())
NEGATORS = {"no", "not", "never", "without", "fails", "failed"}

class SentimentModel:
    def __init__(self, backend: str = "auto"):
        self.backend, self.pipe = "lexicon", None
        if backend in ("auto", "finbert"):
            try:
                from transformers import pipeline
                self.pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True, max_length=256)
                self.backend = "finbert"
            except Exception:
                if backend == "finbert":
                    raise

    def score(self, text: str) -> tuple[float, float]:
        """returns (score in [-1,1], confidence in [0,1])"""
        if self.pipe is not None:
            out = self.pipe([text])[0]
            p = {d["label"].lower(): d["score"] for d in out}
            return round(p.get("positive", 0) - p.get("negative", 0), 4), round(max(p.values()), 4)
        toks = re.findall(r"[a-z\-]+", text.lower())
        pos = neg = 0
        for i, t in enumerate(toks):
            flip = i > 0 and toks[i - 1] in NEGATORS
            if t in POS: pos, neg = (pos, neg + 1) if flip else (pos + 1, neg)
            elif t in NEG: pos, neg = (pos + 1, neg) if flip else (pos, neg + 1)
        return round((pos - neg) / (pos + neg + 1), 4), round(min(1.0, (pos + neg) / 3), 4)
