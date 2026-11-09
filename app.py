"""Mutual Fund Underperformance Early-Warning - web app
Run locally:  streamlit run app.py
Every day the first visit downloads the latest NAVs / index values, rebuilds the
feature panel, retrains the model (train <= Dec-2024, test = 2025 snapshots) and
scores today's data. Later visits that day reuse the cached result.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mfcore import data as D
from mfcore import engine as E

st.set_page_config(page_title='MF Early-Warning', page_icon='📉', layout='wide')
TIER_COLOR = {'High': '#C0392B', 'Elevated': '#E67E22', 'Watch': '#D4AC0D', 'Low': '#27AE60'}
pct = lambda v, d=1: '–' if v is None or pd.isna(v) else f'{v:.{d}%}'
spct = lambda v, d=1: '–' if v is None or pd.isna(v) else f'{v:+.{d}%}'


# ------------------------------------------------------------------ daily pipeline (cached)
@st.cache_resource
def _store():
    """Shared by all visitors: the day's results + a lock so only one rebuild runs at a time."""
    import threading
    return {'lock': threading.Lock(), 'day': None, 'out': None}


def daily_run(day: str, force=False):
    S = _store()
    with S['lock']:
        if force or S['day'] != day:
            box = st.status(f'Running the model for {day} (first visit today, takes about a minute)...', expanded=True)
            say = lambda m: box.write(m)
            env = E.load_all(online=True, progress=say)
            df = E.build_panel(env, progress=say)
            tt = E.train_test(df, progress=say)
            sc = E.score_latest(df, tt['final'], env['data_end'])
            box.update(label="Done - model trained, tested and today's scores are ready", state='complete', expanded=False)
            S['out'], S['day'] = (env, df, tt, sc), day
        return S['out']


day = f'{D.today():%Y-%m-%d}'
if st.session_state.pop('force', False):
    env, df, tt, sc = daily_run(day, force=True)
else:
    env, df, tt, sc = daily_run(day)
de = env['data_end']
horizon = de + pd.DateOffset(years=1)
res = tt['results']
best = res.loc[E.BEST]

# ------------------------------------------------------------------ header
st.title('📉 Mutual Fund Underperformance Early-Warning')
st.markdown(f'Predicts whether an Indian equity mutual fund (Direct, Growth) will **underperform its benchmark '
            f'Total Return Index over the next 12 months**, i.e. from **{de:%d %b %Y}** to **{horizon:%d %b %Y}**.')
c = st.columns(4)
c[0].metric('Run date (today)', f'{D.today():%d %b %Y}')
c[1].metric('Latest data used', f'{de:%d %b %Y}')
c[2].metric('Model', 'Grad. Boosting', help=E.BEST)
c[3].metric('Test ROC-AUC (2025 snapshots)', f"{best['ROC-AUC']:.3f}", help='0.5 = coin toss, 1.0 = perfect ranking')

with st.sidebar:
    st.header('Data status')
    st.write(f"**NAVs:** MFAPI, up to {env['nav_last'].max():%d %b %Y}")
    st.write(f"**Index TRI:** {env['tri_info']['source']}, official to {env['tri_info']['official_to']:%d %b %Y}")
    if env['tri_info']['proxy_days']:
        st.caption(f"Last {env['tri_info']['proxy_days']} day(s) of TRI extended with index-fund NAVs (published TRI not yet available).")
    st.write('**Expense ratio / AUM:** AMFI disclosures stored with the app')
    if env['stale']:
        st.warning('NAV older than 5 days for: ' + ', '.join(env['stale']))
    if st.button('🔄 Re-download data now'):
        st.session_state['force'] = True
        st.rerun()
    st.divider()
    st.caption('Student project (PGDM-BIFS, GIM). Not investment advice. Past performance does not guarantee future results.')

tab1, tab2, tab3, tab4 = st.tabs(['🔍 Check a fund', '📋 Early-warning list', '📊 Model performance', 'ℹ️ How it works'])


# ------------------------------------------------------------------ helpers
def fund_charts(code, bench_name, nav_series=None, label=''):
    f = (nav_series if nav_series is not None else env['nav'][code]).loc[:de]
    b = env['tri'][bench_name].loc[:de]
    start = de - pd.DateOffset(years=3)
    j = pd.concat([f, b], axis=1, join='inner').loc[start:]
    j.columns = ['Fund', bench_name]
    g = j / j.iloc[0] * 100
    fig = px.line(g, labels={'value': 'Value of Rs 100', 'index': '', 'variable': ''},
                  title='Growth of Rs 100 over the last 3 years')
    fig.update_layout(height=340, legend=dict(orientation='h', y=-0.2), margin=dict(t=50, b=10))
    m = pd.concat([f, b], axis=1, join='inner').resample('ME').last().dropna()
    roll = (m.iloc[:, 0].pct_change(12) - m.iloc[:, 1].pct_change(12)).loc[de - pd.DateOffset(years=5):].dropna()
    fig2 = go.Figure(go.Bar(x=roll.index, y=roll.values, marker_color=np.where(roll.values < 0, '#C0392B', '#27AE60')))
    fig2.update_layout(title='Rolling 1-year excess return vs benchmark (monthly)', yaxis_tickformat='.0%', height=340,
                       margin=dict(t=50, b=10))
    a, b2 = st.columns(2)
    a.plotly_chart(fig, width='stretch')
    b2.plotly_chart(fig2, width='stretch')


def show_result(r, universe, note=None):
    tier = r['Risk Tier']
    st.subheader(r['Fund'])
    st.caption(f"{r['Category']} · {r['AMC']} · scheme code {int(r['Scheme Code'])} · benchmark {r['Benchmark']}")
    k = st.columns(4)
    k[0].metric('P(underperform next 12M)', pct(r['P']))
    k[1].markdown(f"**Risk tier**<br><span style='font-size:1.9rem;font-weight:700;color:{TIER_COLOR[tier]}'>{tier}</span>",
                  unsafe_allow_html=True)
    k[2].metric('Rank (1 = riskiest)', f"{int(r['Rank'])} of {len(universe)}")
    k[3].metric('Warning signs', f"{int(r['Warning Signs'])} of 9")
    st.markdown(f"The model estimates a **{pct(r['P'], 0)}** chance that this fund returns less than **{r['Benchmark']}** "
                f"between {de:%d %b %Y} and {horizon:%d %b %Y}. Tiers are relative: High = riskiest 20% of the funds "
                f"scored today, Low = safest 20%.")
    if note:
        st.info(note)
    l, rr = st.columns([1, 1])
    with l:
        st.markdown('**Warning signs (simple rule checks)**')
        if r['Signs']:
            for s in r['Signs']:
                st.markdown(f'- ⚠️ {s}')
        else:
            st.markdown('- ✅ None of the 9 rules triggered')
    with rr:
        st.markdown('**Key numbers today**')
        tbl = pd.DataFrame({'Value': [spct(r['Excess_1Y']), spct(r['Excess_3Y']), f"{r['IR_1Y']:.2f}",
                                      pct(r['Hit_Rate_12M'], 0), f"{r['Down_Capture_3Y']:.2f}" if pd.notna(r['Down_Capture_3Y']) else '–',
                                      pct(r['TE_1Y']), f"{r['TER_Direct']:.2f}%", '–' if pd.isna(r['AUM_Cr']) else f"{r['AUM_Cr']:,.0f}"]},
                           index=['Excess return 1Y', 'Excess return 3Y (ann.)', 'Information ratio 1Y', 'Months beating index (last 12)',
                                  'Down-capture 3Y', 'Tracking error 1Y', 'Expense ratio (Direct)', 'AUM (Rs crore)'])
        st.dataframe(tbl, width='stretch')


# ------------------------------------------------------------------ tab 1
with tab1:
    mode = st.radio('Choose', ['One of the 30 project funds', 'Add any other fund (scheme code)'], horizontal=True)
    if mode.startswith('One'):
        names = sc.sort_values('Fund')['Fund'].tolist()
        pick = st.selectbox('Fund', names, index=None, placeholder='Select a fund')
        if st.button('▶ Run', type='primary', disabled=pick is None):
            st.session_state['res'] = ('own', pick)
    else:
        st.caption('Works for equity funds that should be judged against NIFTY 100 (large cap), NIFTY MIDCAP 150 (mid cap) '
                   'or NIFTY 500 (flexi / multi cap). Use the Direct-Growth scheme code from mfapi.in / AMFI.')
        q = st.text_input('Search by fund name (optional)', placeholder='e.g. Quant Flexi Cap Direct Growth')
        code_default = None
        if q:
            try:
                hits = [h for h in D.search_schemes(q) if 'direct' in h['schemeName'].lower() and 'growth' in h['schemeName'].lower()] or D.search_schemes(q)
                if hits:
                    sel = st.selectbox('Search results', hits[:30], format_func=lambda h: f"{h['schemeCode']} - {h['schemeName']}")
                    code_default = int(sel['schemeCode'])
                else:
                    st.warning('No schemes found.')
            except Exception as e:
                st.warning(f'Search is unavailable right now ({type(e).__name__}); type the scheme code instead.')
        a, b, c3, d4 = st.columns(4)
        code = a.number_input('Scheme code', min_value=100000, max_value=999999, step=1, value=code_default)
        cat = b.selectbox('Category', list(D.BM.keys()), index=2)
        ter = c3.number_input('Expense ratio % (optional)', min_value=0.0, max_value=3.0, value=None, step=0.01,
                              help='Direct plan TER. Blank = category median is used.')
        aum = d4.number_input('AUM Rs crore (optional)', min_value=1.0, value=None, step=100.0,
                              help='Latest fund size. Blank = treated as unknown.')
        if st.button('▶ Run', type='primary', disabled=not code):
            st.session_state['res'] = ('new', dict(**{'Scheme Code': int(code)}, Category=cat, TER=ter, AUM_Cr=aum))

    res_req = st.session_state.get('res')
    if res_req and res_req[0] == 'own' and not mode.startswith('One'):
        res_req = None
    if res_req and res_req[0] == 'new' and mode.startswith('One'):
        res_req = None
    if res_req:
        st.divider()
        if res_req[0] == 'own':
            r = sc[sc['Fund'] == res_req[1]].iloc[0]
            show_result(r, sc)
            fund_charts(r['Scheme Code'], r['Benchmark'])
            h = tt['test'][tt['test']['Fund'] == r['Fund']].copy()
            h['Predicted'] = h['P'].map(pct)
            h['Actually underperformed?'] = h[E.TARGET].map({1: 'Yes', 0: 'No'})
            h['Excess return next 12M'] = h['Fwd_Excess_1Y'].map(spct)
            h['Correct at 50%?'] = np.where((h['P'] >= 0.5) == (h[E.TARGET] == 1), '✅', '❌')
            st.markdown('**How the model did on this fund in the test period** (predictions made in 2025, outcomes now known)')
            st.dataframe(h.assign(Month=h['Obs Date'].dt.strftime('%b %Y'))[['Month', 'Predicted', 'Actually underperformed?',
                                                                            'Excess return next 12M', 'Correct at 50%?']],
                         hide_index=True, width='stretch')
        else:
            fd = res_req[1]
            try:
                with st.spinner('Downloading NAV history and scoring the fund...'):
                    out, code, src, missing, s = E.score_new_fund(env, tt['final'], fd)
                r = out[out['Scheme Code'] == code].iloc[0]
                note = []
                if code in set(env['funds']['Scheme Code']):
                    note.append('This scheme is one of the 30 project funds; result shown with your inputs.')
                if fd['TER'] is None:
                    note.append('Expense ratio not given, so the category median was used.')
                if fd['AUM_Cr'] is None:
                    note.append('AUM not given, so size-related features were filled with typical values.')
                if missing:
                    note.append(f'Fund is younger than 3 years: {len(missing)} long-window features were filled with typical values.')
                note.append('Rank and tier are relative to the 30 project funds plus this fund.')
                show_result(r, out, ' '.join(note))
                fund_charts(code, r['Benchmark'], nav_series=s)
            except Exception as e:
                st.error(f'Could not score scheme {fd["Scheme Code"]}: {e}')

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown(f'All 30 funds ranked by probability of underperforming over **{de:%d %b %Y} → {horizon:%d %b %Y}**.')
    t = sc[['Rank', 'Fund', 'Category', 'P', 'Risk Tier', 'Warning Signs', 'Excess_1Y', 'Excess_3Y', 'IR_1Y', 'TER_Direct', 'AUM_Cr']].copy()
    cats = st.multiselect('Category', list(D.BM.keys()), default=list(D.BM.keys()))
    t = t[t['Category'].isin(cats)]
    sty = (t.style.format({'P': '{:.1%}', 'Excess_1Y': '{:+.1%}', 'Excess_3Y': '{:+.1%}', 'IR_1Y': '{:.2f}', 'TER_Direct': '{:.2f}',
                           'AUM_Cr': '{:,.0f}'})
           .map(lambda v: f'color:{TIER_COLOR.get(v, "")};font-weight:700', subset=['Risk Tier'])
           .background_gradient(subset=['P'], cmap='Reds', vmin=0, vmax=1))
    st.dataframe(sty, hide_index=True, width='stretch', height=min(1100, 38 + 35 * len(t)),
                 column_config={'P': 'P(underperform 12M)', 'Excess_1Y': 'Excess 1Y', 'Excess_3Y': 'Excess 3Y (ann.)',
                                'IR_1Y': 'Info ratio 1Y', 'TER_Direct': 'TER %', 'AUM_Cr': 'AUM Rs cr'})
    out = sc.drop(columns=['Signs']).assign(Signs_list=sc['Signs'].str.join('; '))
    st.download_button('⬇ Download full scores (CSV)', out.to_csv(index=False).encode(), f'early_warning_scores_{de:%d%b%Y}.csv', 'text/csv')
    g = sc.groupby('Category')['P'].mean().reindex(list(D.BM.keys()))
    fig = px.bar(g, labels={'value': 'Average P(underperform)', 'Category': ''}, title='Average risk score by category', text_auto='.0%')
    fig.update_layout(showlegend=False, yaxis_tickformat='.0%', height=320)
    st.plotly_chart(fig, width='stretch')

# ------------------------------------------------------------------ tab 3
with tab3:
    ts, te_ = tt['train_span'], tt['test_span']
    st.markdown(f"""
- **Train:** {tt['n_train']:,} fund-months, snapshots {ts[0]:%b %Y} – {ts[1]:%b %Y} ({tt['base_train']:.1%} underperformed)
- **Test (never seen in training):** {tt['n_test']:,} fund-months, predictions made {te_[0]:%b %Y} – {te_[1]:%b %Y},
outcomes checked up to {te_[1] + pd.DateOffset(years=1):%b %Y} ({tt['base_test']:.1%} underperformed)
- **Live prediction:** {de:%d %b %Y} → {horizon:%d %b %Y} (final model refitted on all {tt['n_lab']:,} labelled rows)""")
    show = res[['ROC-AUC', 'Within-month AUC', 'Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1']].astype(float)
    st.dataframe(show.style.format('{:.3f}').highlight_max(axis=0, color='#D5F5E3'), width='stretch')
    st.caption('ROC-AUC: chance the model ranks a real underperformer above a real outperformer (0.5 = guessing). '
               'Within-month AUC: same, but only comparing funds on the same date. Baseline = "last year\'s laggard stays a laggard".')
    st.subheader(f'Confusion matrix – {E.BEST} on the Test set')
    thr = st.slider('Decision threshold (flag the fund if P ≥ threshold)', 0.05, 0.95, 0.50, 0.05)
    tst = tt['test']
    m = E.metrics(tst[E.TARGET].astype(int), tst['P'].to_numpy(), thr)
    l, r_ = st.columns([1, 1])
    cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', aspect='auto',
                    x=['Predicted: beats index', 'Predicted: underperforms'], y=['Actual: beat index', 'Actual: underperformed'])
    fig.update_layout(coloraxis_showscale=False, height=330, margin=dict(t=10))
    l.plotly_chart(fig, width='stretch')
    r_.markdown(f"""
| | |
|---|---|
| Accuracy | **{m['Accuracy']:.1%}** |
| Precision (flagged funds that really lagged) | **{m['Precision']:.1%}** |
| Recall (laggards that were caught) | **{m['Recall']:.1%}** |
| F1 | **{m['F1']:.3f}** |
| Funds flagged | **{int(m['TP'] + m['FP'])} of {len(tst)}** |
""")
    fpr, tpr = tt['roc']
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, name=f"{E.BEST} (AUC {best['ROC-AUC']:.3f})"),
                     go.Scatter(x=[0, 1], y=[0, 1], name='Random guess', line=dict(dash='dash', color='grey'))])
    fig.update_layout(title='ROC curve (Test set)', xaxis_title='False positive rate', yaxis_title='True positive rate', height=380)
    st.plotly_chart(fig, width='stretch')
    q = tst.assign(Q=tst['Obs Date'].dt.to_period('Q').astype(str)).groupby('Q').apply(
        lambda g: pd.Series({'Rows': len(g), 'Underperformed': g[E.TARGET].mean(),
                             'AUC': E.roc_auc_score(g[E.TARGET], g['P']) if g[E.TARGET].nunique() == 2 else np.nan}))
    st.markdown('**Results by prediction quarter**')
    st.dataframe(q.style.format({'Rows': '{:.0f}', 'Underperformed': '{:.0%}', 'AUC': '{:.3f}'}), width='stretch')

# ------------------------------------------------------------------ tab 4
with tab4:
    st.markdown(f"""
### What it does
For each fund it asks one yes/no question: **will the fund's return over the next 12 months be lower than its benchmark index (with dividends reinvested)?**
A machine-learning model answers with a probability.

### Data (refreshed automatically every day)
* **NAV history** of every fund: [mfapi.in](https://www.mfapi.in) (AMFI data).
* **Benchmark Total Return Indices** (NIFTY 100 / MIDCAP 150 / 500 TRI): niftyindices.com. If the latest days are not published yet,
  they are extended with the daily returns of a low-cost index fund on the same index.
* **Expense ratio (TER)** and **fund size (AUM)**: AMFI disclosures.

### How the model is built
1. **Monthly snapshots** from Jun-2018: on each month-end, 46 features are computed using only data available on that date
   (trailing returns, excess returns, volatility, tracking error, beta, drawdown, information ratio, hit-rate, up/down capture,
   rank within category, fund age, expense ratio, AUM, AUM growth, net flows, category).
2. **Label**: did the fund trail its index over the following 12 months?
3. **Chronological split**: train on snapshots up to Dec-2024; test on 2025 snapshots, whose 12-month outcomes are now known.
   No shuffling, so the test truly simulates predicting the future.
4. **Models compared**: Gradient Boosting (selected in the project by walk-forward cross-validation), Random Forest, Logistic Regression,
   and a simple "last year's laggard" baseline.
5. **Today's prediction**: the model is refitted on all labelled months and scores the latest date ({de:%d %b %Y}).
6. **Risk tiers** are relative ranks: High = riskiest 20%, Elevated = next 30%, Watch = next 30%, Low = safest 20%.

### Limitations
* Only 30 funds and about 8 years of history; one test year. Test AUC around 0.6–0.7 means the ranking is better than chance but far from certain.
* Probabilities are not calibrated; use the **ranking and tiers**, not the exact percentage.
* Survivorship: only funds that exist today are included.
* Not investment advice.
""")
