"""Panel build, training, testing and scoring - same design as the project notebook.
Train  : monthly snapshots Jun-2018 .. Dec-2024 (label = next-12-month result, known)
Test   : snapshots from Jan-2025 whose 12-month outcome is already known (chronological, untouched)
Score  : the latest data date (outcome over the next 12 months is the prediction)
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import chars as C
from . import data as D
from . import features as F

SEED = 42
TARGET = 'Target_Underperf_1Y'
PRICE_FEATURES = [
    'Fund_Ret_3M', 'Bench_Ret_3M', 'Excess_3M', 'Fund_Ret_6M', 'Bench_Ret_6M', 'Excess_6M',
    'Fund_Ret_1Y', 'Bench_Ret_1Y', 'Excess_1Y', 'Fund_Ret_2Y', 'Bench_Ret_2Y', 'Excess_2Y',
    'Fund_Ret_3Y', 'Bench_Ret_3Y', 'Excess_3Y',
    'Fund_Vol_1Y', 'Bench_Vol_1Y', 'TE_1Y', 'Beta_1Y', 'Fund_MaxDD_1Y', 'Bench_MaxDD_1Y',
    'Fund_Vol_3Y', 'Bench_Vol_3Y', 'TE_3Y', 'Beta_3Y', 'Fund_MaxDD_3Y', 'Bench_MaxDD_3Y',
    'IR_1Y', 'IR_3Y', 'Ret_per_Vol_1Y', 'Hit_Rate_12M', 'Hit_Rate_36M', 'Up_Capture_3Y', 'Down_Capture_3Y',
    'Excess_1Y_Change_6M', 'Fund_Age_Yrs', 'CatRank_Excess_1Y', 'CatRank_Excess_3Y', 'CatRank_IR_1Y']
CHAR_FEATURES = ['TER_Direct', 'TER_vs_Cat_Median', 'Log_AUM', 'AUM_Growth_1Y', 'Net_Flow_1Y', 'CatRank_AUM']
NUM_FEATURES = PRICE_FEATURES + CHAR_FEATURES
CAT_FEATURES = ['Category']
FEATURES = NUM_FEATURES + CAT_FEATURES

# tuned settings from the notebook (walk-forward CV on the training years)
MODELS = {
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10, max_features=0.3,
                                            class_weight='balanced', n_jobs=-1, random_state=SEED),
    'Logistic Regression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000),
}
BEST = 'Gradient Boosting'

RULES = [('Excess_1Y', lambda v: v < 0, 'Lagging its benchmark over the last 1 year'),
         ('Excess_3Y', lambda v: v < 0, 'Lagging its benchmark over 3 years (annualised)'),
         ('CatRank_Excess_1Y', lambda v: v <= 0.3, 'Bottom 30% of its category on 1-year excess return'),
         ('Hit_Rate_12M', lambda v: v < 0.5, 'Beat the benchmark in fewer than half of the last 12 months'),
         ('Down_Capture_3Y', lambda v: v > 1, 'Falls more than the index in down months (3Y)'),
         ('TE_1Y', lambda v: v > 0.06, 'High tracking error (more than 6%)'),
         ('TER_vs_Cat_Median', lambda v: v > 0.10, 'Costlier than category peers (TER > median + 0.10 pp)'),
         ('CatRank_AUM', lambda v: v >= 0.8, 'Among the largest funds in its category'),
         ('Net_Flow_1Y', lambda v: v > 0.30, 'Heavy net inflows (more than 30% of AUM in a year)')]


def make_pipe(model):
    pre = ColumnTransformer([('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), NUM_FEATURES),
                             ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_FEATURES)])
    return Pipeline([('pre', pre), ('model', clone(model))])


def warning_signs(row):
    out = []
    for col, test, label in RULES:
        v = row.get(col)
        if v is not None and pd.notna(v) and test(v):
            out.append(label)
    return out


def risk_tier(rank_pct):
    return pd.cut(rank_pct, [0, 0.2, 0.5, 0.8, 1.0], labels=['High', 'Elevated', 'Watch', 'Low']).astype(str)


# ------------------------------------------------------------------ data
def load_all(online=True, extra=None, progress=lambda *a: None):
    """Download/refresh everything. extra: list of dicts for user-added funds
    {Fund, Category, AMC, Scheme Code} (+ optional TER, AUM_Cr)."""
    funds = D.load_funds()
    progress('Loading index (TRI) data')
    tri, tri_info = D.get_tri(online)
    nav, src = {}, {}
    codes = list(funds['Scheme Code'])
    for i, code in enumerate(codes):
        progress(f'Fetching latest NAV {i + 1}/{len(codes)}')
        j, s = D.get_nav_json(code, online)
        nav[code] = D.nav_series(j)
        src[code] = s
    nav_last = pd.Series({c: s.index[-1] for c, s in nav.items()})
    data_end = min(D.today(), tri_info['tri_to'], nav_last.max())
    stale = [funds.set_index('Scheme Code').loc[c, 'Fund'] for c, d in nav_last.items() if (data_end - d).days > 5]
    return dict(funds=funds, nav=nav, tri=tri, tri_info=tri_info, nav_src=src, data_end=data_end,
                nav_last=nav_last, stale=stale, ter=C.clean_ter(D.load_ter()), aum=D.load_aum())


def build_panel(env, progress=lambda *a: None):
    progress('Building the monthly feature panel (2018 to today)')
    df = F.build_panel(env['nav'], env['tri'], env['funds'], env['data_end'])
    progress('Adding expense ratio and fund-size features')
    df = C.add_ter(df, env['ter'])
    df = C.add_aum(df, env['aum'], env['nav'], env['data_end'])
    # same row order as the notebook (Gradient Boosting's row subsampling depends on it)
    return F.add_splits(df.sort_values(['Obs Date', 'Fund']).reset_index(drop=True))


# ------------------------------------------------------------------ model
def metrics(y, p, thr=0.5, pred=None):
    pred = (p >= thr).astype(int) if pred is None else pred
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'ROC-AUC': roc_auc_score(y, p), 'Accuracy': accuracy_score(y, pred), 'Balanced Acc': balanced_accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0), 'Recall': recall_score(y, pred, zero_division=0),
            'F1': f1_score(y, pred, zero_division=0), 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}


def within_month_auc(d, p):
    a = []
    for _, g in d.assign(p=p).groupby('Obs Date'):
        if g[TARGET].nunique() == 2:
            a.append(roc_auc_score(g[TARGET], g['p']))
    return float(np.mean(a)) if a else np.nan


def train_test(df, progress=lambda *a: None):
    tr, te = df[df.Split_1Y == 'Train'], df[df.Split_1Y == 'Test']
    Xtr, ytr, Xte, yte = tr[FEATURES], tr[TARGET].astype(int), te[FEATURES], te[TARGET].astype(int)
    res, probs, fitted = {}, {}, {}
    for name, m in MODELS.items():
        progress(f'Training {name}')
        pipe = make_pipe(m).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        r = metrics(yte, p)
        r['Within-month AUC'] = within_month_auc(te, p)
        res[name], probs[name], fitted[name] = r, p, pipe
    mom = (-te['Excess_1Y']).rank(pct=True).to_numpy()
    r = metrics(yte, mom, pred=(te['Excess_1Y'] < 0).astype(int).to_numpy())
    r['Within-month AUC'] = within_month_auc(te, mom)
    res['Baseline: past-1Y laggard'] = r
    fpr, tpr, _ = roc_curve(yte, probs[BEST])
    test = te[['Obs Date', 'Fund', 'Category', 'Scheme Code', 'Excess_1Y', 'Fwd_Excess_1Y', TARGET]].assign(P=probs[BEST])
    progress('Refitting the selected model on all labelled months')
    lab = df[df[TARGET].notna()]
    final = make_pipe(MODELS[BEST]).fit(lab[FEATURES], lab[TARGET].astype(int))
    return dict(results=pd.DataFrame(res).T, test=test, roc=(fpr, tpr), final=final,
                n_train=len(tr), n_test=len(te), base_train=ytr.mean(), base_test=yte.mean(),
                train_span=(tr['Obs Date'].min(), tr['Obs Date'].max()), test_span=(te['Obs Date'].min(), te['Obs Date'].max()),
                n_lab=len(lab))


def score_latest(df, model, data_end):
    latest = df[df['Obs Date'] == data_end].copy()
    latest['P'] = model.predict_proba(latest[FEATURES])[:, 1]
    latest = latest.sort_values('P', ascending=False).reset_index(drop=True)
    latest.insert(0, 'Rank', range(1, len(latest) + 1))
    latest['Risk Tier'] = risk_tier(latest['Rank'] / len(latest))
    latest['Signs'] = latest.apply(lambda r: warning_signs(r), axis=1)
    latest['Warning Signs'] = latest['Signs'].str.len()
    return latest


def score_new_fund(env, model, fund, online=True):
    """Score a fund that is not in the 30-fund universe, against the 30 funds on the same date.
    fund: {Fund, Category, AMC, Scheme Code, TER (optional), AUM_Cr (optional)}"""
    code = int(fund['Scheme Code'])
    j, src = D.get_nav_json(code, online)
    s = D.nav_series(j)
    meta = j.get('meta', {})
    if s.index[-1] < env['data_end'] - pd.Timedelta(days=10):
        raise ValueError(f"Latest NAV for {code} is {s.index[-1]:%d-%b-%Y}; the scheme may be closed or merged.")
    if s.index[0] > env['data_end'] - pd.DateOffset(years=1):
        raise ValueError('The fund needs at least 1 year of NAV history to be scored.')
    nav = dict(env['nav'])
    nav[code] = s
    row = {'Fund': fund.get('Fund') or meta.get('scheme_name', str(code)), 'Category': fund['Category'],
           'AMC': fund.get('AMC') or meta.get('fund_house', ''), 'Scheme Code': code, 'Benchmark': D.BM[fund['Category']]}
    funds = env['funds']
    funds = pd.concat([funds[funds['Scheme Code'] != code], pd.DataFrame([row])], ignore_index=True)
    df = F.build_panel(nav, env['tri'], funds, env['data_end'], only_latest=True)
    ter, aum = env['ter'], env['aum']
    ter = ter[ter['Scheme Code'] != code]
    aum = aum[aum['Scheme Code'] != code]
    if fund.get('TER') is not None:
        ter = pd.concat([ter, pd.DataFrame([{'Scheme Code': code, 'Date': env['data_end'], 'TER_Direct': float(fund['TER'])}])])
    if fund.get('AUM_Cr') is not None:
        q = (env['data_end'] - pd.DateOffset(months=2)).to_period('Q').end_time.normalize()
        aum = pd.concat([aum, pd.DataFrame([{'Scheme Code': code, 'Quarter End': q, 'AUM_Cr': float(fund['AUM_Cr'])}])])
    df = C.add_ter(df, ter)
    df = C.add_aum(df, aum, nav, env['data_end'])
    out = score_latest(df, model, env['data_end'])
    missing = [c for c in PRICE_FEATURES if pd.isna(out.loc[out['Scheme Code'] == code, c].iloc[0])]
    return out, code, src, missing, s
