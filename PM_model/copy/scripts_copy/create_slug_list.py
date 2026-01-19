import pandas as pd

# List of Polymarket event slugs
slugs = [
    "nvda-above-on-january-16-2026",
    "aapl-above-on-january-16-2026",
    "googl-above-on-january-16-2026",
    "msft-above-on-january-16-2026",
    "meta-above-on-january-16-2026",
    "amzn-above-on-january-16-2026",
    "tsla-above-on-january-16-2026",
    "pltr-above-on-january-16-2026",
    "open-above-on-january-16-2026",
    "nflx-above-on-january-16-2026"
]
tickers = [
    "NVDA",
    "AAPL",
    "GOOGL",
    "MSFT",
    "META",
    "AMZN",
    "TSLA",
    "PLTR",
    "OPEN",
    "NFLX"
]

df = pd.DataFrame({"slug": slugs, "ticker": tickers})
df.to_csv("../data/polymarket_target_markets.csv", index=False)
