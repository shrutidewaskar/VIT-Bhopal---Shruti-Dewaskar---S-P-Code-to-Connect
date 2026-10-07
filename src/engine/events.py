"""Rule-weighted event classifier. Fully explainable: returns the matched terms."""
import re
RULES = {
 "Geopolitical": [r"\bwar\b", r"invasion", r"sanction", r"missile", r"military", r"tensions?", r"ceasefire", r"embargo", r"geopolitic", r"troops", r"conflict", r"blockade"],
 "Macroeconomic": [r"\binflation\b", r"\bcpi\b", r"interest rates?", r"rate (hike|cut)", r"\bfed\b", r"central bank", r"\bgdp\b", r"recession", r"unemployment", r"jobs report", r"yield curve", r"treasury yields?"],
 "Credit Event": [r"\bdefault", r"downgrad", r"bankrupt", r"insolven", r"bank run", r"liquidity crisis", r"missed (a )?payment", r"restructur", r"credit rating", r"write-?off", r"bailout", r"bank (collapse|failure)", r"systemic (risk|stress)", r"debt covenant", r"deposit(s)? (outflow|flight)"],
 "Merger/Acquisition": [r"acquir", r"acquisition", r"\bmerger", r"takeover", r"buyout", r"\bbid for\b", r"divest", r"spin-?off"],
 "Product Launch": [r"launch", r"unveil", r"introduc", r"new (model|product|chip|service)", r"rolls? out", r"release[sd]?"],
 "Regulatory": [r"regulator", r"\bsec\b", r"antitrust", r"\bfine[sd]?\b", r"probe", r"investigation", r"lawsuit", r"compliance", r"ban(s|ned)?\b", r"\bsue[sd]?\b"],
 "Cyber/Operational": [r"cyber", r"\bhack", r"data breach", r"ransomware", r"outage", r"recall", r"supply chain", r"strike\b", r"plant shut"],
 "Natural Disaster/Pandemic": [r"earthquake", r"hurricane", r"flood", r"pandemic", r"outbreak", r"wildfire", r"typhoon", r"lockdown"],
 "Earnings": [r"earnings", r"quarterly (results|profit)", r"\beps\b", r"revenue (beat|miss|rose|fell)", r"guidance", r"\bq[1-4]\b"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RULES.items()}

def classify_event(text: str) -> tuple[str, float, list[str]]:
    scores, hits = {}, {}
    for cat, pats in _COMPILED.items():
        m = [p.search(text).group(0).lower() for p in pats if p.search(text)]
        if m: scores[cat], hits[cat] = float(len(m)), m
    if not scores:
        return "Other", 0.3, []
    top = max(scores, key=scores.get)
    conf = (scores[top] / sum(scores.values())) * min(1.0, 0.5 + 0.25 * scores[top])
    return top, round(conf, 3), hits[top]
