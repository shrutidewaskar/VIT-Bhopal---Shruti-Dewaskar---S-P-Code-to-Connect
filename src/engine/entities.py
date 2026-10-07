"""Company/ticker + sector detection via alias matching (transparent, no black box)."""
import re
ALIASES = {
 "AAPL": ["apple", "iphone"], "MSFT": ["microsoft"], "GOOGL": ["google", "alphabet"],
 "AMZN": ["amazon"], "META": ["meta platforms", "facebook"], "NVDA": ["nvidia"],
 "TSLA": ["tesla"], "NFLX": ["netflix"], "JPM": ["jpmorgan", "jp morgan"],
 "GS": ["goldman"], "BAC": ["bank of america"], "C": ["citigroup", "citibank"],
 "WFC": ["wells fargo"], "XOM": ["exxon"], "CVX": ["chevron"], "PFE": ["pfizer"],
 "JNJ": ["johnson & johnson"], "BA": ["boeing"], "KO": ["coca-cola", "coca cola"], "WMT": ["walmart"],
}
SECTORS = {
 "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "AMZN": "Consumer",
 "META": "Technology", "NVDA": "Technology", "TSLA": "Consumer", "NFLX": "Communication",
 "JPM": "Financials", "GS": "Financials", "BAC": "Financials", "C": "Financials", "WFC": "Financials",
 "XOM": "Energy", "CVX": "Energy", "PFE": "Healthcare", "JNJ": "Healthcare",
 "BA": "Industrials", "KO": "Consumer Staples", "WMT": "Consumer Staples",
}
_PATTERNS = {t: re.compile(r"\b(" + "|".join(re.escape(a) for a in al) + r")\b", re.I) for t, al in ALIASES.items()}
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")

def extract_entities(text: str) -> list[str]:
    found = {t for t, p in _PATTERNS.items() if p.search(text)}
    found |= {m for m in _CASHTAG.findall(text) if m in ALIASES}
    return sorted(found)

def sectors_of(tickers: list[str]) -> list[str]:
    return sorted({SECTORS[t] for t in tickers if t in SECTORS})
