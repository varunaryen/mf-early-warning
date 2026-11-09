"""Build the monthly fund x observation-date ML panel.

Inputs
  nav:   dict scheme_code -> pd.Series(NAV, index=DatetimeIndex)  (full MFAPI history)
  tri:   dict benchmark_name -> pd.Series(TRI, index=DatetimeIndex)
  funds: DataFrame [Fund, Category, AMC, Scheme Code, Benchmark]
All series are truncated at DATA_END before any calculation (no post-period data).
"""
import numpy as np
import pandas as pd

OBS_START = pd.Timestamp("2018-06-30")   # after SEBI recategorisation (circular Oct-2017, implemented by mid-2018)
ANN = 250
TRAIN_END = pd.Timestamp("2024-12-31")   # train on snapshots Jun-2018 .. Dec-2024; test = labelled snapshots from Jan-2025


def asof(s, d):
    """Last value on/before d (None if no observation)."""
    i = s.index.searchsorted(d, side="right") - 1
    return (s.index[i], s.iloc[i]) if i >= 0 else (None, None)


def shift_months(d, m):
    return d + pd.DateOffset(months=m)


def point_ret(s, start, end):
    ds, vs = asof(s, start)
    de, ve = asof(s, end)
    if ds is None or de is None or ds < s.index[0] or start < s.index[0]:
        return np.nan
    return ve / vs - 1


def ann(r, yrs):
    return (1 + r) ** (1 / yrs) - 1 if pd.notna(r) else np.nan


def daily_window(f, b, start, end):
    """Aligned daily returns of fund & benchmark on common dates within (start, end]."""
    j = pd.concat([f, b], axis=1, join="inner").loc[:end]
    j = j[j.index >= asof(j.iloc[:, 0], start)[0]] if start >= j.index[0] else None
    if j is None or len(j) < 2:
        return None
    return j.pct_change().dropna()


def max_dd(s):
    return (s / s.cummax() - 1).min()


def window_stats(f, b, start, end, tag):
    out = {}
    if start < f.index[0] or start < b.index[0]:
        return {k: np.nan for k in [f"Fund_Vol_{tag}", f"Bench_Vol_{tag}", f"TE_{tag}", f"Beta_{tag}",
                                    f"Fund_MaxDD_{tag}", f"Bench_MaxDD_{tag}"]}
    r = daily_window(f, b, start, end)
    fr, br = r.iloc[:, 0], r.iloc[:, 1]
    out[f"Fund_Vol_{tag}"] = fr.std(ddof=1) * np.sqrt(ANN)
    out[f"Bench_Vol_{tag}"] = br.std(ddof=1) * np.sqrt(ANN)
    out[f"TE_{tag}"] = (fr - br).std(ddof=1) * np.sqrt(ANN)
    out[f"Beta_{tag}"] = np.cov(fr, br, ddof=1)[0, 1] / br.var(ddof=1)
    fw = f.loc[asof(f, start)[0]:end]
    bw = b.loc[asof(b, start)[0]:end]
    out[f"Fund_MaxDD_{tag}"] = max_dd(fw)
    out[f"Bench_MaxDD_{tag}"] = max_dd(bw)
    return out


def monthly_series(s, months):
    return pd.Series([asof(s, m)[1] if asof(s, m)[0] is not None and m >= s.index[0] else np.nan for m in months],
                     index=months)


def build_panel(nav, tri, funds, DATA_END, only_latest=False):
    nav = {k: v.loc[:DATA_END] for k, v in nav.items()}
    tri = {k: v.loc[:DATA_END] for k, v in tri.items()}
    first = min(min(v.index[0] for v in nav.values()), min(v.index[0] for v in tri.values()))
    all_me = pd.date_range(first, DATA_END, freq="ME")
    obs_dates = all_me[all_me >= OBS_START]
    if DATA_END not in obs_dates:   # final live snapshot on the latest data date
        obs_dates = obs_dates.append(pd.DatetimeIndex([DATA_END]))
    if only_latest:
        obs_dates = pd.DatetimeIndex([DATA_END])
    rows = []
    for _, fd in funds.iterrows():
        f = nav[fd["Scheme Code"]]
        b = tri[fd["Benchmark"]]
        fm = monthly_series(f, all_me)
        bm = monthly_series(b, all_me)
        fmr, bmr = fm.pct_change(fill_method=None), bm.pct_change(fill_method=None)
        for t in obs_dates:
            row = {"Fund": fd["Fund"], "Category": fd["Category"], "AMC": fd["AMC"],
                   "Scheme Code": fd["Scheme Code"], "Benchmark": fd["Benchmark"], "Obs Date": t,
                   "Fund NAV Date Used": asof(f, t)[0], "Bench TRI Date Used": asof(b, t)[0]}
            # --- trailing returns
            for lbl, m in [("3M", 3), ("6M", 6), ("1Y", 12), ("2Y", 24), ("3Y", 36)]:
                st = shift_months(t, -m)
                fr, br = point_ret(f, st, t), point_ret(b, st, t)
                if m > 12:
                    fr, br = ann(fr, m / 12), ann(br, m / 12)
                row[f"Fund_Ret_{lbl}"] = fr
                row[f"Bench_Ret_{lbl}"] = br
                row[f"Excess_{lbl}"] = fr - br
            # --- risk (daily)
            for tag, m in [("1Y", 12), ("3Y", 36)]:
                row.update(window_stats(f, b, shift_months(t, -m), t, tag))
            row["IR_1Y"] = row["Excess_1Y"] / row["TE_1Y"] if row["TE_1Y"] else np.nan
            row["IR_3Y"] = row["Excess_3Y"] / row["TE_3Y"] if row["TE_3Y"] else np.nan
            row["Ret_per_Vol_1Y"] = row["Fund_Ret_1Y"] / row["Fund_Vol_1Y"]
            # --- monthly consistency / capture (last 36 / 12 completed months ending t)
            w36f, w36b = fmr.loc[:t].iloc[-36:], bmr.loc[:t].iloc[-36:]
            ok36 = w36f.notna().sum() == 36 and w36b.notna().sum() == 36
            w12f, w12b = w36f.iloc[-12:], w36b.iloc[-12:]
            row["Hit_Rate_12M"] = (w12f > w12b).mean() if w12f.notna().sum() == 12 else np.nan
            row["Hit_Rate_36M"] = (w36f > w36b).mean() if ok36 else np.nan
            if ok36:
                up, dn = w36b > 0, w36b < 0
                row["Up_Capture_3Y"] = w36f[up].mean() / w36b[up].mean() if up.any() else np.nan
                row["Down_Capture_3Y"] = w36f[dn].mean() / w36b[dn].mean() if dn.any() else np.nan
            else:
                row["Up_Capture_3Y"] = row["Down_Capture_3Y"] = np.nan
            row["Excess_1Y_Change_6M"] = row["Excess_1Y"] - (
                point_ret(f, shift_months(t, -18), shift_months(t, -6)) - point_ret(b, shift_months(t, -18), shift_months(t, -6)))
            row["Fund_Age_Yrs"] = (t - f.index[0]).days / 365.25
            # --- targets (future; never used as features)
            for lbl, m in [("1Y", 12), ("2Y", 24)]:
                end = shift_months(t, m)
                if end <= DATA_END:
                    fr, br = point_ret(f, t, end), point_ret(b, t, end)
                    row[f"Fwd_Fund_Ret_{lbl}"] = fr
                    row[f"Fwd_Bench_Ret_{lbl}"] = br
                    row[f"Fwd_Excess_{lbl}"] = fr - br
                    row[f"Target_Underperf_{lbl}"] = int(fr < br) if pd.notna(fr) and pd.notna(br) else np.nan
                else:
                    for k in ["Fwd_Fund_Ret_", "Fwd_Bench_Ret_", "Fwd_Excess_", "Target_Underperf_"]:
                        row[k + lbl] = np.nan
            rows.append(row)
    df = pd.DataFrame(rows)
    # --- cross-sectional (same date, same category) percentile ranks: 1 = best in category
    for c in ["Excess_1Y", "Excess_3Y", "IR_1Y"]:
        df[f"CatRank_{c}"] = df.groupby(["Obs Date", "Category"])[c].rank(pct=True)
    return df


def add_splits(df, TRAIN_END=TRAIN_END):
    """Chronological Train / Test split (no gap). Rows without a label = live scoring set."""
    t = df["Obs Date"]
    s1 = np.select([df["Target_Underperf_1Y"].isna(), t <= TRAIN_END], ["Score (no label yet)", "Train"], "Test")
    df["Split_1Y"] = s1
    return df
