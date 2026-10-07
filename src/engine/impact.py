"""Transparent 1-10 impact score:
 impact = clip( prior[event] * (0.7+0.3*|sentiment|) * source_credibility * (1+0.12*ln(cluster_size)) + severity_bonus , 1, 10)
Every term is returned so a reviewer can audit any score."""
import math, re
PRIOR = {"Credit Event": 8.5, "Geopolitical": 8.0, "Natural Disaster/Pandemic": 8.0, "Macroeconomic": 6.5,
         "Cyber/Operational": 6.5, "Regulatory": 6.0, "Merger/Acquisition": 5.0, "Earnings": 4.5,
         "Product Launch": 3.0, "Other": 2.0}
CRED = {"news": 1.0, "twitter": 0.7}
SEVERE = re.compile(r"collapse|default|invasion|bank run|bankrupt|crash|emergency|halt|plunge|systemic|contagion", re.I)

def score_impact(event: str, sentiment: float, source: str, cluster_size: int, text: str) -> tuple[float, dict]:
    prior = PRIOR.get(event, 2.0)
    mag = 0.7 + 0.3 * abs(sentiment)
    cred = CRED.get(source, 0.8)
    boost = 1 + 0.12 * math.log(max(cluster_size, 1))
    bonus = min(1.5, 0.75 * len(set(m.lower() for m in SEVERE.findall(text))))
    raw = prior * mag * cred * boost + bonus
    val = round(min(10.0, max(1.0, raw)), 2)
    return val, {"event_prior": prior, "sentiment_magnitude": round(mag, 3), "source_credibility": cred,
                 "cluster_boost": round(boost, 3), "severity_bonus": bonus}
