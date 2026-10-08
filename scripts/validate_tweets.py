"""Out-of-domain validation: does engine sentiment on REAL tweets relate to subsequent stock returns?
FinBERT was NOT trained on these tweets, so this is a clean test (unlike Financial PhraseBank).
Usage: python scripts/validate_tweets.py --csv reduced_dataset-release.csv --n 3000 --export-jsonl data/eval/tweets_sample.jsonl
Metrics per horizon: Spearman rho, directional hit rate (|s|>0.2) vs base rate, top-minus-bottom quintile return spread (Welch t).
Baselines: the dataset's own TEXTBLOB_POLARITY and LSTM_POLARITY columns."""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy import stats
from src.engine.sentiment import SentimentModel

HORIZONS = ["1_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
def clean(t): return re.sub(r"\s+", " ", re.sub(r"http\S+|@\w+", "", str(t))).strip()

def metrics(s, r):
    s, r = np.asarray(s, float), np.asarray(r, float); ok = ~(np.isnan(s) | np.isnan(r)); s, r = s[ok], r[ok]
    rho, p = stats.spearmanr(s, r)
    m = np.abs(s) > 0.2 if np.abs(s).max() > 0.2 else np.ones_like(s, bool)
    hit = float((np.sign(s[m]) == np.sign(r[m])).mean()) if m.sum() else float("nan")
    base = float(max((r[m] > 0).mean(), (r[m] < 0).mean())) if m.sum() else float("nan")
    q = pd.qcut(pd.Series(s).rank(method="first"), 5, labels=False)
    top, bot = r[q == 4], r[q == 0]; t = stats.ttest_ind(top, bot, equal_var=False)
    return {"n": int(len(s)), "spearman": round(float(rho), 4), "p_value": round(float(p), 4), "n_directional": int(m.sum()),
            "hit_rate": round(hit, 4), "majority_base_rate": round(base, 4),
            "q5_minus_q1": round(float(top.mean() - bot.mean()), 5), "t_stat": round(float(t.statistic), 2)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="docs/tweet_validation"); ap.add_argument("--export-jsonl")
    a = ap.parse_args()
    df = pd.read_csv(a.csv, encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["TWEET"]); df["text"] = df["TWEET"].map(clean); df = df[df.text.str.len() > 15]
    df = df.sample(min(a.n, len(df)), random_state=42).reset_index(drop=True)
    print(f"{len(df)} tweets; date range {df['DATE'].min()} .. {df['DATE'].max()}")
    if a.export_jsonl:
        os.makedirs(os.path.dirname(a.export_jsonl) or ".", exist_ok=True)
        with open(a.export_jsonl, "w", encoding="utf-8") as f:
            for i, r in df.sort_values("DATE").iterrows():
                f.write(json.dumps({"id": f"tw{i:05d}", "source": "twitter", "timestamp": pd.to_datetime(r["DATE"]).isoformat(),
                                    "text": r["text"], "tickers": [str(r["STOCK"])] if "STOCK" in df and pd.notna(r["STOCK"]) else []}) + "\n")
        print("exported ->", a.export_jsonl)
    model = SentimentModel("auto"); df["engine"] = [x[0] for x in model.score_batch(df["text"].tolist())]
    cols = {f"engine ({model.backend})": "engine", "TextBlob (dataset)": "TEXTBLOB_POLARITY", "LSTM (dataset)": "LSTM_POLARITY"}
    res, md = {}, ["| Horizon | Scorer | N | Spearman | p | Hit rate | Base rate | Q5-Q1 return | t |", "|---|---|---|---|---|---|---|---|---|"]
    for h in [h for h in HORIZONS if h in df]:
        for name, c in cols.items():
            if c not in df: continue
            m = metrics(df[c], pd.to_numeric(df[h], errors="coerce")); res[f"{h}|{name}"] = m
            md.append(f"| {h} | {name} | {m['n']} | {m['spearman']} | {m['p_value']} | {m['hit_rate']} | {m['majority_base_rate']} | {m['q5_minus_q1']} | {m['t_stat']} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=2); open(a.out + ".md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
