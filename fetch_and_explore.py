"""WTI crude oil price analysis: events, inflation, dollar/gold correlation, and forecasting."""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Quiet Prophet's Stan backend logging
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

# Global chart style, set once for all figures
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9

# Every figure is written to figures/ so the results are readable in the repo
# without installing the dependencies and running this script.
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


def save(name):
    """Write the current figure to figures/<name>.png."""
    plt.tight_layout()
    plt.savefig(FIGURES / f"{name}.png", dpi=150)


# ===== WTI monthly oil price (FRED: WTISPLC) =====
oil = pd.read_csv("oil_price.csv")
oil.columns = ["date", "price"]
oil["date"] = pd.to_datetime(oil["date"])
oil["price"] = pd.to_numeric(oil["price"], errors="coerce")
oil = oil.set_index("date")
oil = oil.loc["1970":]                    # drop the flat price-controlled era, keep 1973/79 shocks
print(oil.isna().sum())

# ===== CPI (FRED: CPIAUCSL) for inflation adjustment =====
cpi = pd.read_csv("cpi.csv")
cpi.columns = ["date", "cpi"]
cpi["date"] = pd.to_datetime(cpi["date"])
cpi["cpi"] = pd.to_numeric(cpi["cpi"], errors="coerce")
cpi["cpi"] = cpi["cpi"].ffill()           # fill the single missing month
cpi = cpi.set_index("date")

# Real price: convert every month to latest-month dollars
oil["cpi"] = cpi["cpi"]
base_cpi = cpi["cpi"].iloc[-1]
oil["real_price"] = oil["price"] * base_cpi / oil["cpi"]

print("Top 5 by NOMINAL price:")
print(oil["price"].sort_values(ascending=False).head(5).round(1))
print("Top 5 by REAL price:")
print(oil["real_price"].sort_values(ascending=False).head(5).round(1))

# ===== Broad US Dollar Index (FRED: DTWEXBGS), daily -> monthly mean =====
dollar = pd.read_csv("dollar_index.csv")
dollar.columns = ["date", "dollar"]
dollar["date"] = pd.to_datetime(dollar["date"])
dollar["dollar"] = pd.to_numeric(dollar["dollar"], errors="coerce")
dollar = dollar.set_index("date")
dollar_monthly = dollar["dollar"].resample("MS").mean()
oil["dollar"] = dollar_monthly            # NaN before 2006 (no data)

# Oil vs dollar correlation: on levels, and on more-robust monthly returns
corr = oil["price"].corr(oil["dollar"])
print("Correlation (oil price vs dollar):", round(corr, 3))
corr_pct = oil["price"].pct_change().corr(oil["dollar"].pct_change())
print("Correlation (monthly % change):", round(corr_pct, 3))

# Rolling 36-month correlation to test whether the relationship is stable over time
oil["roll_corr"] = oil["price"].pct_change().rolling(36).corr(oil["dollar"].pct_change())

# Regression of oil returns on dollar returns: slope = sensitivity, R^2 = explanatory power
reg = oil[["price", "dollar"]].pct_change().dropna() * 100
x = reg["dollar"]
y = reg["price"]
slope, intercept = np.polyfit(x, y, 1)
r2 = x.corr(y) ** 2                        # for simple linear regression, R^2 = corr^2
print(f"Regression: oil% = {slope:.2f} * dollar% + ({intercept:.2f})")
print(f"R-squared: {r2:.3f}  (dollar explains {r2*100:.0f}% of oil's monthly variance)")

# ===== Gold (Yahoo Finance GC=F, daily -> monthly mean, stored in gold.csv) =====
# FRED's LBMA gold series was removed in 2022; sourced from Yahoo instead (see README)
gold = pd.read_csv("gold.csv")
gold["date"] = pd.to_datetime(gold["date"])
gold["gold"] = pd.to_numeric(gold["gold"], errors="coerce")
gold = gold.set_index("date")
oil["gold"] = gold["gold"]                # NaN before 2000; no interpolation, keep data honest

# Oil vs gold: level correlation looks real, but returns correlation is ~0 (spurious/trend-driven)
corr_gold = oil["price"].corr(oil["gold"])
corr_gold_pct = oil["price"].pct_change().corr(oil["gold"].pct_change())
print("Correlation (oil vs gold):", round(corr_gold, 3))
print("Correlation (oil vs gold, monthly % change):", round(corr_gold_pct, 3))

# Events table
events = pd.read_csv("events.csv")
events["date"] = pd.to_datetime(events["date"])

# ===== Biggest monthly moves + automatic event matching =====
oil["pct_change"] = oil["price"].pct_change() * 100
oil["abs_change"] = oil["pct_change"].abs()
biggest_moves = oil.sort_values("abs_change", ascending=False).head(10)

# Attach the nearest event within 180 days to each large move
matched = pd.merge_asof(
    biggest_moves.sort_index().reset_index(),
    events.sort_values("date"),
    on="date",
    direction="nearest",
    tolerance=pd.Timedelta("180D"),
)
matched = matched.sort_values("abs_change", ascending=False)
print(matched[["date", "price", "pct_change", "event"]])

# ===== Event study: 1990 Gulf War (index price to 100 at event month) =====
event_date = pd.Timestamp("1990-08-01")
window = oil.loc[event_date - pd.DateOffset(months=6):event_date + pd.DateOffset(months=6)].copy()
window["indexed"] = window["price"] / window.loc[event_date, "price"] * 100

fig, ax = plt.subplots()
ax.plot(window.index, window["indexed"], marker="o", color="#0b5394", linewidth=1.8)
ax.fill_between(window.index, 100, window["indexed"],
                where=window["indexed"] >= 100, color="#0b5394", alpha=0.12)  # shade the rise
ax.axvline(event_date, color="crimson", linestyle="--", linewidth=1.2)
ax.axhline(100, color="gray", linestyle=":", linewidth=1)

# Annotate the peak gain
peak_date = window["indexed"].idxmax()
peak_val = window["indexed"].max()
ax.annotate(f"Peak +{peak_val - 100:.0f}%", xy=(peak_date, peak_val),
            xytext=(10, 12), textcoords="offset points", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#333333"))
ax.annotate("Iraq invades Kuwait", xy=(event_date, window["indexed"].min()),
            xytext=(6, 0), textcoords="offset points",
            rotation=90, va="bottom", fontsize=8, color="crimson")

ax.set_title("Event Study: Gulf War 1990 (event month = 100)")
ax.set_xlabel("Date")
ax.set_ylabel("Price (event month = 100)")
save("event-study-gulf-war-1990")

# ===== Event study: five crises overlaid on a common axis =====
study_events = {
    "1990-08-01": "Gulf War 1990",
    "2008-07-01": "GFC 2008",
    "2020-03-01": "COVID 2020",
    "2022-02-01": "Ukraine 2022",
    "2026-03-01": "US-Iran 2026",
}

plt.figure()
for ev_str, name in study_events.items():
    ev = pd.Timestamp(ev_str)
    w = oil.loc[ev - pd.DateOffset(months=6):ev + pd.DateOffset(months=6)].copy()
    w["indexed"] = w["price"] / w.loc[ev, "price"] * 100
    months = (w.index.year - ev.year) * 12 + (w.index.month - ev.month)   # months from event
    plt.plot(months, w["indexed"], marker="o", linewidth=1.8, label=name)

plt.axvline(0, color="gray", linestyle="--")
plt.axhline(100, color="gray", linestyle=":")
plt.title("Event Study Comparison (event month = 100)")
plt.xlabel("Months from event")
plt.ylabel("Price (event month = 100)")
plt.legend(title="Event")
save("event-study-comparison")

# ===== Nominal vs real price =====
fig, ax = plt.subplots()
ax.plot(oil.index, oil["price"], color="#999999", linewidth=1.3, label="Nominal (current dollars)")
ax.plot(oil.index, oil["real_price"], color="#0b5394", linewidth=1.6, label="Real (2026 dollars)")

# Annotate the all-time real high (2008)
rp_date = oil["real_price"].idxmax()
rp_val = oil["real_price"].max()
ax.annotate(f"All-time real high\n{rp_date.year}: ${rp_val:.0f}",
            xy=(rp_date, rp_val), xytext=(-120, -10), textcoords="offset points",
            fontsize=9, fontweight="bold", color="#0b5394",
            arrowprops=dict(arrowstyle="->", color="#0b5394"))

ax.set_title("WTI Oil: Nominal vs Real Price")
ax.set_xlabel("Year")
ax.set_ylabel("Price (USD per barrel)")
ax.legend()
save("nominal-vs-real-price")

# ===== Oil vs Dollar Index (dual-axis, 2006+) =====
d = oil.loc["2006":]
fig, ax1 = plt.subplots()
ax1.plot(d.index, d["price"], color="tab:blue", label="Oil price")
ax1.set_xlabel("Year")
ax1.set_ylabel("Oil price (USD/barrel)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()                          # shared x-axis, second y-axis for the dollar
ax2.plot(d.index, d["dollar"], color="tab:green", label="Dollar index")
ax2.set_ylabel("Broad Dollar Index", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

# Merge both axes' handles into one legend
lines = ax1.get_lines() + ax2.get_lines()
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="lower right")
ax1.set_title(f"Oil vs US Dollar Index (2006-2026, r={corr:.2f})")
save("oil-vs-dollar-index")

# ===== Rolling correlation over time =====
fig, ax = plt.subplots()
ax.plot(oil.index, oil["roll_corr"], color="#8e44ad", linewidth=1.6)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
# Shade positive (red) vs negative (blue) regimes
ax.fill_between(oil.index, 0, oil["roll_corr"],
                where=oil["roll_corr"] >= 0, color="crimson", alpha=0.15)
ax.fill_between(oil.index, 0, oil["roll_corr"],
                where=oil["roll_corr"] < 0, color="#0b5394", alpha=0.15)
ax.set_ylim(-1, 1)
ax.set_title("Rolling 36-Month Correlation: Oil vs Dollar (% change)")
ax.set_xlabel("Year")
ax.set_ylabel("Correlation")
save("rolling-correlation-oil-dollar")

# ===== Regression scatter: one point per month + fit line =====
fig, ax = plt.subplots()
ax.scatter(x, y, s=14, alpha=0.5, color="#0b5394")
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, slope * xs + intercept, color="crimson", linewidth=2,
        label=f"y = {slope:.2f}x + ({intercept:.2f})")
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_title(f"Oil vs Dollar Monthly Returns (R² = {r2:.2f})")
ax.set_xlabel("Dollar index monthly % change")
ax.set_ylabel("Oil price monthly % change")
ax.legend()
save("regression-oil-vs-dollar-returns")

# ===== Oil vs Gold (dual-axis, 2000+) =====
g = oil.loc["2000":]
fig, ax1 = plt.subplots()
ax1.plot(g.index, g["price"], color="#0b5394", label="Oil price")
ax1.set_xlabel("Year")
ax1.set_ylabel("Oil price (USD/barrel)", color="#0b5394")
ax1.tick_params(axis="y", labelcolor="#0b5394")

ax2 = ax1.twinx()
ax2.plot(g.index, g["gold"], color="#d4a017", label="Gold price")
ax2.set_ylabel("Gold (USD/oz)", color="#d4a017")
ax2.tick_params(axis="y", labelcolor="#d4a017")

lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")
# Title makes the point: level correlation looks real, returns correlation is ~0 (spurious)
ax1.set_title(f"Oil vs Gold (level r={corr_gold:.2f}, but return r={corr_gold_pct:.2f})")
save("oil-vs-gold")

# ===== Main chart: full price panorama with event annotations =====
fig, ax = plt.subplots()
ax.plot(oil.index, oil["price"], color="#0b5394", linewidth=1.6)
ax.set_title("WTI Crude Oil Price with Major Events (1970-2026)", pad=12)
ax.set_xlabel("Year")
ax.set_ylabel("Price (USD per barrel)")

ymax = oil["price"].max()
ax.set_ylim(0, ymax * 1.08)                # minimal headroom so the line fills the chart
for _, row in events.iterrows():
    ax.axvline(row["date"], color="crimson", linestyle="--", alpha=0.35, linewidth=1)
    ax.text(row["date"], ymax * 0.50, row["event"],
            rotation=90, fontsize=9, color="#333333",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
save("price-history-with-events")

# ===== ARIMA 12-month forecast (1986+ monthly prices) =====
oil_ts = oil.loc["1986":, "price"].asfreq("MS")
arima_fit = ARIMA(oil_ts, order=(1, 1, 1)).fit()   # d=1 differences out the trend
fc = arima_fit.get_forecast(steps=12)
mean_fc = fc.predicted_mean
ci = fc.conf_int()
print("ARIMA 12-month forecast (USD/barrel):")
print(mean_fc.round(1))

# ===== Prophet forecast for comparison =====
# Prophet's Stan backend can fail to load on some Windows setups; guard it so the script
# always produces the ARIMA forecast and adds Prophet only when its backend is available.
prophet_future = None
try:
    prophet_df = oil_ts.reset_index()
    prophet_df.columns = ["ds", "y"]       # Prophet requires columns named ds/y
    m = Prophet(interval_width=0.95)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=12, freq="MS")
    prophet_fc = m.predict(future).set_index("ds")
    prophet_future = prophet_fc.loc[mean_fc.index]
    print("Prophet 12-month forecast (USD/barrel):")
    print(prophet_future["yhat"].round(1))
except Exception as e:
    print("Prophet skipped (backend unavailable on this environment):", type(e).__name__)

# ===== Forecast chart: ARIMA alone, or ARIMA vs Prophet if available =====
fig, ax = plt.subplots()
recent = oil_ts.loc["2015":]
ax.plot(recent.index, recent, color="#0b5394", label="History")

ax.plot(mean_fc.index, mean_fc, color="crimson", linewidth=2, label="ARIMA forecast")
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="crimson", alpha=0.12)

if prophet_future is not None:
    ax.plot(prophet_future.index, prophet_future["yhat"], color="#2e8b57", linewidth=2, label="Prophet forecast")
    ax.fill_between(prophet_future.index, prophet_future["yhat_lower"], prophet_future["yhat_upper"],
                    color="#2e8b57", alpha=0.12)
    ax.set_title("Oil Price 12-Month Forecast: ARIMA vs Prophet")
else:
    ax.set_title("ARIMA(1,1,1): 12-Month WTI Oil Price Forecast")

ax.set_xlabel("Year")
ax.set_ylabel("Price (USD per barrel)")
ax.legend(loc="upper left")
save("arima-forecast")

plt.show()
