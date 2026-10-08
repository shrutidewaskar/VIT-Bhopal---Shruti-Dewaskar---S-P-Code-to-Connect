"""Create data/eval/events_to_label.csv: random real headlines + tweets for you to label with an event type.
Allowed labels: Geopolitical, Macroeconomic, Credit Event, Merger/Acquisition, Product Launch, Regulatory,
Cyber/Operational, Natural Disaster/Pandemic, Earnings, Other.
Usage: python scripts/make_label_sheet.py --news data/eval/all-data.csv --tweets reduced_dataset-release.csv"""
import argparse, os, re
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--news", required=True); ap.add_argument("--tweets")
ap.add_argument("--n-news", type=int, default=110); ap.add_argument("--n-tweets", type=int, default=40); a = ap.parse_args()
news = pd.read_csv(a.news, header=None, encoding="latin-1", names=["sentiment", "text"]).dropna().sample(a.n_news, random_state=7)
rows = [{"text": t, "source": "news"} for t in news.text]
if a.tweets:
    tw = pd.read_csv(a.tweets, on_bad_lines="skip").dropna(subset=["TWEET"])
    tw = tw[tw.TWEET.str.len() > 40].sample(a.n_tweets, random_state=7)
    rows += [{"text": re.sub(r"\s+", " ", re.sub(r"http\S+", "", t)).strip(), "source": "twitter"} for t in tw.TWEET]
df = pd.DataFrame(rows); df["pre_label"] = ""; df["event_type"] = ""
os.makedirs("data/eval", exist_ok=True); df.to_csv("data/eval/events_to_label.csv", index=False)
print(f"{len(df)} rows -> data/eval/events_to_label.csv  (fill event_type; then save the finished file as data/eval/events_labeled.csv with columns text,event_type)")
