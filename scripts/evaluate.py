"""Evaluate the engine against labeled data. Produces docs/eval_results.json + docs/eval_results.md.
Usage (from repo root):
  python scripts/evaluate.py --sentiment-csv data/eval/all-data.csv --events-csv data/eval/events_labeled.csv
Sentiment CSV: Kaggle 'Sentiment Analysis for Financial News' (no header, latin-1: label,text) or any CSV with sentiment/text columns.
Events CSV: columns text,event_type."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src.engine.sentiment import SentimentModel
from src.engine.events import classify_event

def load_sentiment(path, n, seed=42):
    df = pd.read_csv(path, header=None, encoding="latin-1")
    if str(df.iloc[0, 0]).strip().lower() in ("sentiment", "label"):
        df.columns = [str(c).strip().lower() for c in df.iloc[0]]; df = df.iloc[1:]
    else:
        df.columns = ["sentiment", "text"] + list(df.columns[2:])
    df = df[["sentiment", "text"]].dropna()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[df["sentiment"].isin(["positive", "negative", "neutral"])]
    if len(df) > n:  # stratified sample
        df = df.groupby("sentiment", group_keys=False).apply(lambda g: g.sample(min(len(g), n // 3), random_state=seed))
    return df.reset_index(drop=True)

def eval_sentiment(df, backend):
    m = SentimentModel(backend)
    preds = [lab for _, _, lab in m.score_batch(df["text"].tolist())]
    return m.backend, {
        "n": len(df), "accuracy": round(accuracy_score(df["sentiment"], preds), 4),
        "macro_f1": round(f1_score(df["sentiment"], preds, average="macro"), 4),
        "per_class": classification_report(df["sentiment"], preds, output_dict=True, zero_division=0)}

def eval_events(path):
    df = pd.read_csv(path).dropna(subset=["text", "event_type"])
    df["event_type"] = df["event_type"].str.strip()
    preds = [classify_event(t)[0] for t in df["text"]]
    labels = sorted(set(df["event_type"]) | set(preds))
    majority = df["event_type"].value_counts().idxmax()
    return {"n": len(df), "accuracy": round(accuracy_score(df["event_type"], preds), 4),
            "macro_f1": round(f1_score(df["event_type"], preds, average="macro", zero_division=0), 4),
            "majority_baseline_accuracy": round((df["event_type"] == majority).mean(), 4),
            "per_class": classification_report(df["event_type"], preds, output_dict=True, zero_division=0),
            "labels": labels, "confusion_matrix": confusion_matrix(df["event_type"], preds, labels=labels).tolist(),
            "errors": [{"text": t, "true": y, "pred": p} for t, y, p in zip(df["text"], df["event_type"], preds) if y != p][:40]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment-csv"); ap.add_argument("--events-csv")
    ap.add_argument("--n", type=int, default=1500); ap.add_argument("--out", default="docs/eval_results")
    a = ap.parse_args(); res = {}
    if a.sentiment_csv:
        df = load_sentiment(a.sentiment_csv, a.n); res["sentiment"] = {}
        for be in ("lexicon", "auto"):
            name, r = eval_sentiment(df, be); res["sentiment"][name] = r
            print(f"[sentiment/{name}] n={r['n']} acc={r['accuracy']} macroF1={r['macro_f1']}")
    if a.events_csv:
        res["events"] = eval_events(a.events_csv); e = res["events"]
        print(f"[events/rules] n={e['n']} acc={e['accuracy']} macroF1={e['macro_f1']} (majority baseline {e['majority_baseline_accuracy']})")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2)
    md = ["| Task | Model | N | Accuracy | Macro-F1 |", "|---|---|---|---|---|"]
    for be, r in res.get("sentiment", {}).items(): md.append(f"| Sentiment | {be} | {r['n']} | {r['accuracy']} | {r['macro_f1']} |")
    if "events" in res: md.append(f"| Event type | rules | {res['events']['n']} | {res['events']['accuracy']} | {res['events']['macro_f1']} |")
    open(a.out + ".md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
