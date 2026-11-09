"""Fund characteristics from AMFI: expense ratio (TER) and quarterly average AUM.
Point-in-time: at Obs Date t only values disclosed on or before t are used
(AUM: a quarter counts as public 1 month after quarter end)."""
import numpy as np
import pandas as pd
from . import features as F

AUM_LAG_MONTHS = 1


def clean_ter(ter):
    """Replace isolated one-month reporting spikes with the previous month's value."""
    ter = ter.sort_values(['Scheme Code', 'Date']).copy()
    g = ter.groupby('Scheme Code')['TER_Direct']
    prev, nxt = g.shift(1), g.shift(-1)
    spike = ((ter.TER_Direct - prev).abs() > 0.5) & ((ter.TER_Direct - nxt).abs() > 0.5) & ((prev - nxt).abs() < 0.25)
    ter.loc[spike, 'TER_Direct'] = prev[spike]
    return ter


def add_ter(df, ter):
    ter = ter.sort_values('Date')
    by = {c: g for c, g in ter.groupby('Scheme Code')}

    def asof(code, t):
        s = by.get(code)
        if s is None:
            return np.nan
        s = s[s['Date'] <= t]
        return s['TER_Direct'].iloc[-1] if len(s) else np.nan
    df['TER_Direct'] = [asof(c, t) for c, t in zip(df['Scheme Code'], df['Obs Date'])]
    df['TER_Imputed'] = df['TER_Direct'].isna().astype(int)
    med = df.groupby(['Obs Date', 'Category'])['TER_Direct'].transform('median')
    df['TER_Direct'] = df['TER_Direct'].fillna(med)
    df['TER_vs_Cat_Median'] = df['TER_Direct'] - df.groupby(['Obs Date', 'Category'])['TER_Direct'].transform('median')
    return df


def add_aum(df, aum, nav, data_end):
    aum = aum.sort_values('Quarter End')
    by = {c: g for c, g in aum.groupby('Scheme Code')}

    def asof(code, t):
        a = by.get(code)
        if a is None:
            return (np.nan, np.nan, pd.NaT)
        avail = a[a['Quarter End'] + pd.DateOffset(months=AUM_LAG_MONTHS) <= t]
        if not len(avail):
            return (np.nan, np.nan, pd.NaT)
        q = avail.iloc[-1]
        prev = avail[avail['Quarter End'] <= q['Quarter End'] - pd.DateOffset(months=12)]
        g = q['AUM_Cr'] / prev['AUM_Cr'].iloc[-1] - 1 if len(prev) else np.nan
        return (q['AUM_Cr'], g, q['Quarter End'])
    vals = [asof(c, t) for c, t in zip(df['Scheme Code'], df['Obs Date'])]
    df['AUM_Cr'] = [v[0] for v in vals]
    df['Log_AUM'] = np.log(df['AUM_Cr'])
    df['AUM_Growth_1Y'] = [v[1] for v in vals]
    df['AUM_Quarter_Used'] = [v[2] for v in vals]
    nf = []
    for code, g, qe in zip(df['Scheme Code'], df['AUM_Growth_1Y'], df['AUM_Quarter_Used']):
        nf.append(np.nan if pd.isna(g) else g - F.point_ret(nav[code].loc[:data_end], qe - pd.DateOffset(months=12), qe))
    df['Net_Flow_1Y'] = nf
    df['CatRank_AUM'] = df.groupby(['Obs Date', 'Category'])['AUM_Cr'].rank(pct=True)
    return df
