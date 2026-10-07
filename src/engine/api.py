"""uvicorn src.engine.api:app --reload   (reads data/signals.jsonl)"""
import json, os
from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import RiskEngine

app = FastAPI(title="NLP Risk Engine")
SIGNALS = os.environ.get("SIGNALS_FILE", "data/signals.jsonl")
_engine = None
def engine():
    global _engine
    _engine = _engine or RiskEngine()
    return _engine

class Doc(BaseModel):
    text: str
    source: str = "news"
    timestamp: str = "2026-01-01T00:00:00"

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/signals")
def signals(min_impact: float = 0, event_type: str | None = None, ticker: str | None = None, limit: int = 200):
    with open(SIGNALS, encoding="utf-8") as f: rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r["impact_score"] >= min_impact
            and (not event_type or r["event_type"] == event_type) and (not ticker or ticker in r["entities"])]
    return rows[-limit:]

@app.post("/analyze")
def analyze(d: Doc):
    return engine().process({"id": "adhoc", "text": d.text, "source": d.source, "timestamp": d.timestamp})
