"""Risk Engine pipeline: raw text records -> structured risk signals (JSONL).
Run:  python -m src.engine.pipeline --inputs data/sample_news.jsonl data/sample_tweets.jsonl --out data/signals.jsonl"""
import argparse, json, re
from datetime import datetime
from .sentiment import SentimentModel
from .events import classify_event
from .entities import extract_entities, sectors_of
from .impact import score_impact

STOP = set("the a an of to in on for and as at by with from is are was be its it this that after amid over".split())
def _tokens(t): return {w for w in re.findall(r"[a-z]{3,}", t.lower()) if w not in STOP}
def _jaccard(a, b): return len(a & b) / max(1, len(a | b))

class RiskEngine:
    """Stateful so it can run in streaming/replay mode. Clusters near-duplicate stories
    (same event type, Jaccard>=0.3 or shared ticker, within 48h) -> cluster_size feeds impact."""
    def __init__(self, backend="auto", window_h=48):
        self.sent = SentimentModel(backend)
        self.clusters, self.window_h, self._n = [], window_h, 0

    def _assign_cluster(self, ts, event, tickers, toks):
        for c in self.clusters:
            if c["event"] != event or (ts - c["ts"]).total_seconds() > self.window_h * 3600: continue
            if _jaccard(toks, c["toks"]) >= 0.3 or (tickers and set(tickers) & c["tickers"]):
                c["size"] += 1; c["tickers"] |= set(tickers); c["toks"] |= toks; c["ts"] = ts
                return c
        self._n += 1
        c = {"id": f"C{self._n:04d}", "event": event, "ts": ts, "size": 1, "tickers": set(tickers), "toks": set(toks)}
        self.clusters.append(c); return c

    def process(self, rec: dict) -> dict:
        text, ts = rec["text"], datetime.fromisoformat(rec["timestamp"])
        s, s_conf = self.sent.score(text)
        event, e_conf, terms = classify_event(text)
        tickers = rec.get("tickers") or extract_entities(text)
        cl = self._assign_cluster(ts, event, tickers, _tokens(text))
        impact, why = score_impact(event, s, rec["source"], cl["size"], text)
        return {"id": rec["id"], "timestamp": rec["timestamp"], "source": rec["source"], "text": text,
                "entities": tickers, "sectors": sectors_of(tickers),
                "sentiment_score": s, "sentiment_confidence": s_conf, "sentiment_backend": self.sent.backend,
                "event_type": event, "event_confidence": e_conf, "matched_terms": terms,
                "impact_score": impact, "impact_breakdown": why,
                "cluster_id": cl["id"], "cluster_size": cl["size"]}

def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f: recs += [json.loads(l) for l in f if l.strip()]
    return sorted(recs, key=lambda r: r["timestamp"])   # time-ordered replay

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "finbert", "lexicon"])
    a = ap.parse_args()
    eng = RiskEngine(a.backend)
    with open(a.out, "w", encoding="utf-8") as f:
        n = 0
        for r in load(a.inputs):
            f.write(json.dumps(eng.process(r)) + "\n"); n += 1
    print(f"{n} signals -> {a.out} (sentiment backend: {eng.sent.backend})")
