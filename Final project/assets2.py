import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs
from pathlib import Path
import numpy as np

DATA_DIR = Path("data_IS")
files = sorted(DATA_DIR.glob("data2_*.parquet"))

data2 = {}
for file in files:
    q = file.stem.replace("data2_", "")
    data2[q] = pd.read_parquet(file)


q, df = next(iter(data2.items()))

def normalize_group2_quarter(df):
    df = df.copy()

    # datetime jako index
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime")

    df = df.sort_index()

    return df

data2_norm = {}

for q, df in data2.items():
    data2_norm[q] = normalize_group2_quarter(df)

q, df = next(iter(data2_norm.items()))

def clean_group2_quarter(df):
    df = df.copy()

    hour = df.index.hour
    weekday = df.index.weekday

  
    is_daily_break = (hour == 16)

    is_saturday = (weekday == 5)

    is_sunday_preopen = (weekday == 6) & (hour < 18)

    is_friday_postclose = (weekday == 4) & (hour >= 16)

    df = df[~(
        is_daily_break |
        is_saturday |
        is_sunday_preopen |
        is_friday_postclose
    )]

    return df

data2_clean = {}

for q, df in data2_norm.items():
    data2_clean[q] = clean_group2_quarter(df)

#Is it profitable to trade individually?

assets = ["AUD", "CAD", "XAU", "XAG"]

def compute_log_returns(data2_clean):
    return np.log(data2_clean / data2_clean.shift(1))

autocorr = {}

for asset in assets:
    autocorr[asset] = {}

    for q, df in data2_clean.items():
        r = compute_log_returns(df[asset]).dropna()
        autocorr[asset][q] = r.autocorr(lag=1)

autocorr_df = pd.DataFrame(autocorr)

autocorr_df.to_csv('autocorr_df.csv', index=False)

# Let's analyze the stability of the autocorrelations
stability = {}

for asset in assets:
    signs = np.sign(autocorr_df[asset])
    stability[asset] = {
        "mean_autocorr": autocorr_df[asset].mean(),
        "std_autocorr": autocorr_df[asset].std(),
        "negative_ratio": (signs < 0).mean(), 
        "positive_ratio": (signs > 0).mean()
    }

stability_df = pd.DataFrame(stability).T

stability_df.to_csv('stability_df.csv', index=False)

# Let's analyze the average volatility of the assets
vol = {}

for asset in assets:
    vols = []
    for q, df in data2_clean.items():
        r = compute_log_returns(df)[asset].dropna()
        vols.append(r.std())
    vol[asset] = np.mean(vols)

vol_df = pd.DataFrame.from_dict(vol, orient='index', columns=['avg_volatility'])

vol_df.to_csv('vol_df.csv', index=False)

#Spread

spreads = [
    ("XAU", "XAG"),
    ("AUD", "CAD")
]

spread_autocorr = {}

for a, b in spreads:
    name = f"{a}-{b}"
    spread_autocorr[name] = {}
    
    for q, df in data2_clean.items():
        r = compute_log_returns(df)
        spread = (r[a] - r[b]).dropna()
        spread_autocorr[name][q] = spread.autocorr(lag=1)

spread_autocorr_df = pd.DataFrame(spread_autocorr)
spread_autocorr_df.to_csv('spread_autocorr_df.csv', index=False)

spread_stability = {}

for spread in spread_autocorr_df.columns:
    signs = np.sign(spread_autocorr_df[spread])
    spread_stability[spread] = {
        "mean_autocorr": spread_autocorr_df[spread].mean(),
        "std_autocorr": spread_autocorr_df[spread].std(),
        "negative_ratio": (signs < 0).mean(),
        "positive_ratio": (signs > 0).mean()
    }

spread_stability_df = pd.DataFrame(spread_stability).T

spread_stability_df.to_csv('spread_stability_df.csv', index=False)

spread_vol = {}

for a, b in spreads:
    name = f"{a}-{b}"
    vols = []
    
    for q, df in data2_clean.items():
        r = compute_log_returns(df)
        spread = (r[a] - r[b]).dropna()
        vols.append(spread.std())
        
    spread_vol[name] = np.mean(vols)

spread_vol_df = pd.DataFrame.from_dict(spread_vol, orient='index', columns=['avg_volatility'])
spread_vol_df.to_csv('spread_vol_df.csv', index=False)

corrs = {}

for a, b in spreads:
    vals = []
    for q, df in data2_clean.items():
        r = compute_log_returns(df)[[a, b]].dropna()
        vals.append(r[a].corr(r[b]))
    corrs[f"{a}-{b}"] = np.mean(vals)

corrs_df = pd.DataFrame.from_dict(corrs, orient='index', columns=['avg_correlation'])
corrs_df.to_csv('corrs_df.csv', index=False)

#portfelio analysis

portfolios = {
    "FX": ["AUD", "CAD"],
    "Metals": ["XAU", "XAG"],
    "All": ["AUD", "CAD", "XAU", "XAG"]
}

portfolio_autocorr = {}

for name, assets_p in portfolios.items():
    portfolio_autocorr[name] = {}
    
    for q, df in data2_clean.items():
        r = compute_log_returns(df)[assets_p]
        port_ret = r.mean(axis=1).dropna()   # równe wagi
        portfolio_autocorr[name][q] = port_ret.autocorr(lag=1)

portfolio_autocorr_df = pd.DataFrame(portfolio_autocorr)
portfolio_autocorr_df.to_csv('portfolio_autocorr_df.csv', index=False)

portfolio_stability = {}

for p in portfolio_autocorr_df.columns:
    signs = np.sign(portfolio_autocorr_df[p])
    portfolio_stability[p] = {
        "mean_autocorr": portfolio_autocorr_df[p].mean(),
        "std_autocorr": portfolio_autocorr_df[p].std(),
        "negative_ratio": (signs < 0).mean(),
        "positive_ratio": (signs > 0).mean()
    }

pd.DataFrame(portfolio_stability).T

portfolio_stability_df = pd.DataFrame(portfolio_stability).T
portfolio_stability_df.to_csv('portfolio_stability_df.csv', index=False)

portfolio_vol = {}

for name, assets_p in portfolios.items():
    vols = []
    for q, df in data2_clean.items():
        r = compute_log_returns(df)[assets_p]
        port_ret = r.mean(axis=1).dropna()
        vols.append(port_ret.std())
    portfolio_vol[name] = np.mean(vols)

portfolio_vol_df = pd.DataFrame.from_dict(portfolio_vol, orient='index', columns=['avg_volatility'])
portfolio_vol_df.to_csv('portfolio_vol_df.csv', index=False)