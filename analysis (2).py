"""
Tokyo electricity demand analysis (Python, pandas, NumPy).

Data: tokyo_hourly_clean.csv.gz - 39,744 hourly records, 2022-04-01 to 2026-10-12 (JST)
  * TEPCO Power Grid "でんき予報": area demand, TEPCO day-ahead forecast, supply capacity, solar output (GW)
  * Japan Meteorological Agency: hourly temperature, humidity, sunshine (Tokyo station + Kanto average)

Run:  python analysis.py      (all tables and charts are written next to this file)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
DATA = HERE / "tokyo_hourly_clean.csv.gz"

# Chart palette (validated categorical order: blue, orange, aqua, yellow) on a near-white surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
SEASON_COLORS = {"Winter": BLUE, "Summer": ORANGE, "Spring": AQUA, "Autumn": YELLOW}
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#2a78d6", "#0d366b"])

COMFORT_C = 16.0      # temperature with the strongest "distance from comfort" correlation (see relationships)
HEAT_MAX_C = 16.0     # fit heating slope below this
COOL_MIN_C = 22.0     # fit cooling slope above this


# --------------------------------------------------------------------------------------
# 0. Load
# --------------------------------------------------------------------------------------
def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df["weekday"] = pd.Categorical(df["weekday"], ordered=True, categories=[
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    df["season"] = pd.Categorical(df["season"], ordered=True, categories=["Winter", "Spring", "Summer", "Autumn"])
    df["day_type"] = np.where(df["is_workday"], "Workday", "Weekend/holiday")
    df["net_demand_gw"] = df["demand_gw"] - df["solar_gw"]          # demand left for non-solar plants
    df["forecast_error_gw"] = df["demand_gw"] - df["tepco_forecast_gw"]
    df["abs_pct_error"] = (df["forecast_error_gw"].abs() / df["demand_gw"] * 100)
    edges = list(range(-4, 42, 2))
    df["temp_band"] = pd.cut(df["temp_tokyo_c"], bins=edges, right=False,
                             labels=[f"{a} to {a + 2}" for a in edges[:-1]])
    return df


def daily(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.normalize())
    out = pd.DataFrame({
        "peak_gw": g["demand_gw"].max(),
        "peak_hour": g["demand_gw"].idxmax().dt.hour,
        "low_gw": g["demand_gw"].min(),
        "mean_gw": g["demand_gw"].mean(),
        "mean_temp_c": g["temp_tokyo_c"].mean(),
        "max_temp_c": g["temp_tokyo_c"].max(),
        "is_workday": g["is_workday"].first(),
        "season": g["season"].first(),
    })
    out.index.name = "date"
    return out


# --------------------------------------------------------------------------------------
# 1. Basic statistics
# --------------------------------------------------------------------------------------
MAIN = ["demand_gw", "tepco_forecast_gw", "supply_gw", "usage_rate_pct", "solar_gw",
        "temp_tokyo_c", "rh_tokyo_pct", "sun_tokyo_h"]


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    s = df[MAIN].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    s["iqr"] = s["75%"] - s["25%"]
    s["cv_pct"] = s["std"] / s["mean"] * 100
    s["skew"] = df[MAIN].skew()
    s["kurtosis"] = df[MAIN].kurt()
    s["missing"] = df[MAIN].isna().sum()
    return s.round(2)


# --------------------------------------------------------------------------------------
# 2. Highest / lowest values
# --------------------------------------------------------------------------------------
def extremes(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    rows = []
    for col, label in [("demand_gw", "Demand"), ("temp_tokyo_c", "Tokyo temperature"),
                       ("usage_rate_pct", "Usage rate (demand / supply)"), ("solar_gw", "Solar output")]:
        for kind, sub in [("highest", df.nlargest(n, col)), ("lowest", df.nsmallest(n, col))]:
            if col in ("usage_rate_pct", "solar_gw") and kind == "lowest":
                continue  # lowest usage/solar are uninformative (night-time zeros)
            for rank, (ts, r) in enumerate(sub.iterrows(), 1):
                rows.append({"measure": label, "kind": kind, "rank": rank,
                             "datetime": ts.strftime("%Y-%m-%d %H:00"), "value": round(float(r[col]), 2),
                             "demand_gw": round(float(r["demand_gw"]), 2),
                             "temp_tokyo_c": round(float(r["temp_tokyo_c"]), 1),
                             "day_type": r["day_type"], "holiday": bool(r["is_holiday"])})
    return pd.DataFrame(rows)


def yearly_records(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("year")
    out = pd.DataFrame({
        "hours": g.size(),
        "max_demand_gw": g["demand_gw"].max(),
        "max_demand_at": g["demand_gw"].idxmax().dt.strftime("%Y-%m-%d %H:00"),
        "min_demand_gw": g["demand_gw"].min(),
        "max_temp_c": g["temp_tokyo_c"].max(),
        "hours_35c_plus": g["temp_tokyo_c"].apply(lambda s: int((s >= 35).sum())),
        "max_solar_gw": g["solar_gw"].max(),
        "hours_usage_95pct_plus": g["usage_rate_pct"].apply(lambda s: int((s >= 95).sum())),
    })
    return out.round(2)


# --------------------------------------------------------------------------------------
# 3. Group and compare
# --------------------------------------------------------------------------------------
def by_season_hour(df):
    return df.pivot_table(index="hour", columns="season", values="demand_gw", aggfunc="mean", observed=True).round(2)


def by_day_type(df):
    t = df.groupby("day_type")["demand_gw"].agg(["count", "mean", "median", "max", "min"]).round(2)
    prof = df.pivot_table(index="hour", columns="day_type", values="demand_gw", aggfunc="mean")
    gap = prof["Workday"] - prof["Weekend/holiday"]
    return t, gap.round(2)


def by_weekday(df):
    return df.groupby("weekday", observed=True)["demand_gw"].agg(["mean", "max", "min"]).round(2)


def by_month(df):
    return df.groupby("month").agg(
        mean_demand_gw=("demand_gw", "mean"), max_demand_gw=("demand_gw", "max"),
        mean_supply_gw=("supply_gw", "mean"), mean_temp_c=("temp_tokyo_c", "mean"),
        mean_solar_gw=("solar_gw", "mean"),
        hours_usage_95pct_plus=("usage_rate_pct", lambda s: int((s >= 95).sum()))).round(2)


def peak_hour_by_season(d: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(d["season"], d["peak_hour"], normalize="index") * 100
    top = ct.apply(lambda r: r.nlargest(2).index.tolist(), axis=1)
    return pd.DataFrame({"most_common_peak_hours": top,
                         "share_of_days_pct": ct.apply(lambda r: round(r.nlargest(2).sum(), 1), axis=1)})


# --------------------------------------------------------------------------------------
# 4. Trends
# --------------------------------------------------------------------------------------
def monthly_series(df):
    m = df.resample("MS").agg({"demand_gw": "mean", "solar_gw": "mean", "temp_tokyo_c": "mean"})
    m["solar_max_gw"] = df["solar_gw"].resample("MS").max()
    m["demand_12m_avg"] = m["demand_gw"].rolling(12, min_periods=12).mean()
    return m.round(3)


def summer_by_year(df):
    s = df[df["month"].isin([7, 8])]
    g = s.groupby("year")
    return pd.DataFrame({
        "mean_temp_c": g["temp_tokyo_c"].mean(),
        "hours_35c_plus": g["temp_tokyo_c"].apply(lambda x: int((x >= 35).sum())),
        "mean_demand_gw": g["demand_gw"].mean(),
        "peak_demand_gw": g["demand_gw"].max(),
    }).round(2)


def solar_trend(df):
    g = df.groupby("year")["solar_gw"]
    out = pd.DataFrame({"max_solar_gw": g.max(), "p99_solar_gw": g.quantile(0.99)})
    yrs = out.index.values.astype(float)
    slope = np.polyfit(yrs, out["p99_solar_gw"].values, 1)[0]
    return out.round(2), round(float(slope), 3)


# --------------------------------------------------------------------------------------
# 5. Relationships
# --------------------------------------------------------------------------------------
def correlations(df):
    cols = {"demand_gw": "Demand", "temp_tokyo_c": "Temperature", "dist_from_comfort_c": "|Temp - 16 °C|",
            "rh_tokyo_pct": "Humidity", "sun_tokyo_h": "Sunshine", "solar_gw": "Solar output",
            "hour": "Hour of day", "is_workday": "Workday"}
    d = df.assign(dist_from_comfort_c=(df["temp_tokyo_c"] - COMFORT_C).abs(),
                  is_workday=df["is_workday"].astype(int))
    return d[list(cols)].corr().rename(index=cols, columns=cols).round(2)


def comfort_point_scan(df):
    """Correlation of demand with |temperature - x| for each candidate comfort temperature x."""
    xs = np.arange(10, 24.5, 0.5)
    r = [df["demand_gw"].corr((df["temp_tokyo_c"] - x).abs()) for x in xs]
    return pd.Series(np.round(r, 3), index=xs, name="r")


def temperature_slopes(df):
    """GW per °C on workday daytime hours (10:00-17:00), fitted separately on the cold and hot side."""
    w = df[df["is_workday"] & df["hour"].between(10, 17)]
    cold = w[w["temp_tokyo_c"] <= HEAT_MAX_C]
    hot = w[w["temp_tokyo_c"] >= COOL_MIN_C]
    hb, ha = np.polyfit(cold["temp_tokyo_c"], cold["demand_gw"], 1)
    cb, ca = np.polyfit(hot["temp_tokyo_c"], hot["demand_gw"], 1)
    r2 = lambda x, y, b, a: 1 - np.sum((y - (b * x + a)) ** 2) / np.sum((y - y.mean()) ** 2)  # noqa: E731
    return {
        "heating_gw_per_c": round(-hb, 2), "heating_r2": round(r2(cold.temp_tokyo_c, cold.demand_gw, hb, ha), 2),
        "heating_n": len(cold), "heating_fit": (hb, ha),
        "cooling_gw_per_c": round(cb, 2), "cooling_r2": round(r2(hot.temp_tokyo_c, hot.demand_gw, cb, ca), 2),
        "cooling_n": len(hot), "cooling_fit": (cb, ca),
        "cooling_vs_heating": round(cb / -hb, 2),
    }


def humidity_effect(df):
    """On hot workday afternoons, does humidity add demand at the same temperature? (multiple regression)"""
    b = df[df["is_workday"] & df["hour"].between(12, 16) & df["temp_tokyo_c"].between(28, 34)]
    b = b.dropna(subset=["rh_tokyo_pct"])
    X = np.column_stack([np.ones(len(b)), b["temp_tokyo_c"], b["rh_tokyo_pct"]])
    coef, *_ = np.linalg.lstsq(X, b["demand_gw"].to_numpy(), rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((b["demand_gw"] - pred) ** 2) / np.sum((b["demand_gw"] - b["demand_gw"].mean()) ** 2)
    return {"hours": len(b), "gw_per_c": round(coef[1], 2), "gw_per_10pct_rh": round(coef[2] * 10, 2),
            "r2": round(float(r2), 2), "simple_r_rh": round(b["demand_gw"].corr(b["rh_tokyo_pct"]), 2)}


def solar_relationships(df):
    mid = df[df["hour"].between(11, 13)]
    prof = df.groupby("hour")[["demand_gw", "net_demand_gw", "solar_gw"]].mean()
    spring = df[df["month"].isin([4, 5]) & df["is_workday"]].groupby("hour")[["demand_gw", "net_demand_gw"]].mean()
    return {
        "r_solar_vs_kanto_sunshine_midday": round(mid["solar_gw"].corr(mid["sun_kanto_h"]), 2),
        "profile": prof.round(2),
        "spring_profile": spring.round(2),
        "noon_demand_gw": round(prof.loc[12, "demand_gw"], 1),
        "noon_net_gw": round(prof.loc[12, "net_demand_gw"], 1),
        "net_peak_hour": int(prof["net_demand_gw"].idxmax()),
        "demand_peak_hour": int(prof["demand_gw"].idxmax()),
        "spring_noon_net_gw": round(spring.loc[12, "net_demand_gw"], 1),
        "spring_evening_net_gw": round(spring["net_demand_gw"].loc[17:19].max(), 1),
    }


def forecast_accuracy(df):
    by_band = df.groupby("temp_band", observed=True).agg(hours=("abs_pct_error", "size"),
                                                         mape_pct=("abs_pct_error", "mean"))
    by_band = by_band[by_band["hours"] >= 100]
    by_year = df.groupby("year")["abs_pct_error"].mean()
    return {"mape_pct": round(df["abs_pct_error"].mean(), 2),
            "bias_gw": round(df["forecast_error_gw"].mean(), 3),
            "by_band": by_band.round(2), "by_year": by_year.round(2)}


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, title, sub):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_daily_profile(prof: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=SURFACE)
    _style(ax)
    for season in ["Winter", "Summer", "Spring", "Autumn"]:
        y = prof[season]
        ax.plot(prof.index, y, color=SEASON_COLORS[season], linewidth=2, label=season)
        ax.text(23.3, y.iloc[-1], season, color=INK, fontsize=9, va="center")
    for season, h in [("Summer", 14), ("Winter", 9), ("Winter", 18)]:
        ax.plot(h, prof.loc[h, season], "o", color=SEASON_COLORS[season], markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.3, 25.2)
    ax.set_xlabel("Hour of day (JST)", color=INK_2, fontsize=9)
    ax.set_ylabel("Average demand (GW)", color=INK_2, fontsize=9)
    _title(ax, "Summer demand peaks mid-afternoon; winter has morning and evening peaks",
           "Average demand by hour, Apr 2022 – Oct 2026. Dots mark each season's typical peak hour.")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=4)
    _save(fig, "fig1_daily_profile_by_season.png")


def chart_temperature(df, slopes):
    w = df[df["is_workday"] & df["hour"].between(10, 17)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    _style(ax, grid=None)
    hb = ax.hexbin(w["temp_tokyo_c"], w["demand_gw"], gridsize=45, cmap=SEQUENTIAL, mincnt=1, linewidths=0.2,
                   edgecolors=SURFACE)
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("Hours", color=INK_2, fontsize=8.5)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    b, a = slopes["heating_fit"]
    x = np.linspace(w["temp_tokyo_c"].min(), HEAT_MAX_C, 20)
    ax.plot(x, b * x + a, color=INK, linewidth=2)
    ax.text(x[2], b * x[2] + a + 3.5, f"Colder: +{slopes['heating_gw_per_c']} GW per °C", color=INK, fontsize=9)
    b, a = slopes["cooling_fit"]
    x = np.linspace(COOL_MIN_C, w["temp_tokyo_c"].max(), 20)
    ax.plot(x, b * x + a, color=INK, linewidth=2)
    ax.text(27.5, b * 32 + a, f"Hotter: +{slopes['cooling_gw_per_c']} GW per °C", color=INK, fontsize=9,
            ha="right", va="center")
    ax.set_xlabel("Tokyo temperature (°C)", color=INK_2, fontsize=9)
    ax.set_ylabel("Demand (GW)", color=INK_2, fontsize=9)
    _title(ax, f"A degree of heat adds {slopes['cooling_vs_heating']:.1f}x as much demand as a degree of cold",
           "Workday hours 10:00–17:00. Lines: linear fits below 16 °C and above 22 °C.")
    _save(fig, "fig2_demand_vs_temperature.png")


def chart_trends(m: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1.4, 1]})
    for ax in (a1, a2):
        _style(ax)
    a1.plot(m.index, m["demand_gw"], color=BLUE, linewidth=2, marker="o", markersize=3.5,
            markeredgecolor=SURFACE, markeredgewidth=1)
    a1.plot(m.index, m["demand_12m_avg"], color=INK_2, linewidth=1.5)
    last = m["demand_12m_avg"].dropna()
    a1.text(last.index[0], last.iloc[0] + 0.5, "12-month average", color=INK_2, fontsize=8.5, ha="left",
            va="bottom")
    a1.set_ylabel("Avg demand (GW)", color=INK_2, fontsize=9)
    _title(a1, "Demand peaks every summer and winter; the yearly level stays near 32 GW",
           "Top: monthly average demand and its 12-month rolling average. "
           "Bottom: highest solar output in each month.")
    a2.bar(m.index, m["solar_max_gw"], width=24, color=ORANGE)
    a2.set_ylabel("Max solar (GW)", color=INK_2, fontsize=9)
    a2.set_ylim(0, 19)
    _save(fig, "fig3_monthly_trends.png")


def chart_solar(prof: pd.DataFrame, title_values: dict):
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=SURFACE)
    _style(ax)
    ax.fill_between(prof.index, prof["net_demand_gw"], prof["demand_gw"], color=ORANGE, alpha=0.12, linewidth=0)
    ax.plot(prof.index, prof["demand_gw"], color=BLUE, linewidth=2, label="Total demand")
    ax.plot(prof.index, prof["net_demand_gw"], color=ORANGE, linewidth=2, label="Demand minus solar")
    ax.text(8.6, prof.loc[9, "demand_gw"] + 0.3, "Total demand", color=INK, fontsize=9, ha="right")
    ax.text(14.3, prof.loc[14, "net_demand_gw"], "Demand minus solar", color=INK, fontsize=9, ha="left",
            va="top")
    ax.text(12, (prof.loc[12, "demand_gw"] + prof.loc[12, "net_demand_gw"]) / 2, "covered by solar",
            ha="center", va="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.3, 25.2)
    ax.set_xlabel("Hour of day (JST)", color=INK_2, fontsize=9)
    ax.set_ylabel("Average (GW)", color=INK_2, fontsize=9)
    _title(ax, "Solar pushes the real peak into the evening (spring workdays)",
           f"Apr–May workdays. At noon solar leaves {title_values['spring_noon_net_gw']} GW for other plants; "
           f"by early evening it is {title_values['spring_evening_net_gw']} GW.")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, "fig4_solar_net_demand.png")


def chart_corr(c: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=40, ha="right", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(c)), c.index, fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    ax.set_xticks(np.arange(-0.5, len(c)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(c)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.outline.set_visible(False)
    ax.set_title("Correlations: demand tracks distance from 16 °C, not temperature itself",
                 loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    _save(fig, "fig5_correlation_matrix.png")


def chart_forecast(by_band: pd.DataFrame, overall: float):
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    _style(ax)
    x = range(len(by_band))
    ax.bar(x, by_band["mape_pct"], color=BLUE, width=0.78)
    ax.axhline(overall, color=INK_2, linewidth=1)
    ax.text(-0.4, overall + 0.03, f"All hours: {overall:.2f} %", color=INK_2, fontsize=8.5, ha="left")
    i = int(by_band["mape_pct"].values.argmax())
    ax.text(i, by_band["mape_pct"].iloc[i] + 0.03, f"{by_band['mape_pct'].iloc[i]:.2f} %", ha="center",
            fontsize=8.5, color=INK)
    ax.set_xticks(list(x), [s.replace(" to ", "–") for s in by_band.index.astype(str)], rotation=45, ha="right")
    ax.set_xlabel("Tokyo temperature (°C)", color=INK_2, fontsize=9)
    ax.set_ylabel("Mean absolute % error", color=INK_2, fontsize=9)
    _title(ax, "TEPCO's demand forecast is least accurate in hot weather",
           "Mean absolute % error of TEPCO's published forecast vs actual demand. Bands with < 100 hours left out.")
    _save(fig, "fig6_forecast_error_by_temperature.png")


# --------------------------------------------------------------------------------------
def run(save: bool = True) -> dict:
    df = load()
    d = daily(df)
    out = {
        "df": df, "daily": d,
        "basic_statistics": basic_statistics(df),
        "extremes": extremes(df),
        "yearly_records": yearly_records(df),
        "by_season_hour": by_season_hour(df),
        "by_month": by_month(df),
        "by_weekday": by_weekday(df),
        "peak_hour_by_season": peak_hour_by_season(d),
        "monthly": monthly_series(df),
        "summer_by_year": summer_by_year(df),
        "correlations": correlations(df),
        "comfort_scan": comfort_point_scan(df),
        "slopes": temperature_slopes(df),
        "humidity": humidity_effect(df),
        "solar": solar_relationships(df),
        "forecast": forecast_accuracy(df),
    }
    out["day_type"], out["workday_gap_by_hour"] = by_day_type(df)
    out["solar_trend"], out["solar_p99_slope_gw_per_year"] = solar_trend(df)
    if save:
        out["basic_statistics"].to_csv(HERE / "basic_statistics.csv")
        out["extremes"].to_csv(HERE / "highest_lowest_values.csv", index=False)
        out["yearly_records"].to_csv(HERE / "yearly_records.csv")
        out["by_month"].to_csv(HERE / "demand_by_month.csv")
        out["by_season_hour"].to_csv(HERE / "demand_by_hour_and_season.csv")
        out["correlations"].to_csv(HERE / "correlation_matrix.csv")
        chart_daily_profile(out["by_season_hour"])
        chart_temperature(df, out["slopes"])
        chart_trends(out["monthly"])
        chart_solar(out["solar"]["spring_profile"], out["solar"])
        chart_corr(out["correlations"])
        chart_forecast(out["forecast"]["by_band"], out["forecast"]["mape_pct"])
    return out


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 200)
    print(r["basic_statistics"][["mean", "std", "min", "50%", "max", "skew"]])
    print("\nSlopes:", {k: v for k, v in r["slopes"].items() if not k.endswith("_fit")})
    print("Humidity:", r["humidity"])
    print("Solar:", {k: v for k, v in r["solar"].items() if "profile" not in k})
    print("Forecast:", {k: v for k, v in r["forecast"].items() if k in ("mape_pct", "bias_gw")})
    print("Comfort point (max r):", r["comfort_scan"].idxmax(), r["comfort_scan"].max())
    print(r["peak_hour_by_season"])
    print(r["summer_by_year"])
    print(r["solar_trend"], r["solar_p99_slope_gw_per_year"])
    print(r["day_type"])
