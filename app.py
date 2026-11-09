"""Mutual Fund Underperformance Early-Warning - web app
Run locally:  streamlit run app.py
The first visit each day downloads the latest NAVs / index values, rebuilds the
feature panel, retrains the model (train <= Dec-2024, test = 2025 snapshots) and
scores today's data. Later visits that day reuse the cached result.
"""
import threading

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mfcore import data as D
from mfcore import engine as E

st.set_page_config(page_title='MF Early-Warning', page_icon='📉', layout='wide', initial_sidebar_state='collapsed')

NAVY, BLUE, TEAL = '#0B2545', '#1F4E79', '#13817A'
TIER_COLOR = {'High': '#C0392B', 'Elevated': '#E67E22', 'Watch': '#C9A400', 'Low': '#1E8449'}
TIER_BG = {'High': '#FDEDEC', 'Elevated': '#FEF5E7', 'Watch': '#FEF9E7', 'Low': '#E9F7EF'}
TIERS = ['High', 'Elevated', 'Watch', 'Low']
pct = lambda v, d=1: '–' if v is None or pd.isna(v) else f'{v:.{d}%}'
spct = lambda v, d=1: '–' if v is None or pd.isna(v) else f'{v:+.{d}%}'

# ------------------------------------------------------------------ styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 55%, {TEAL} 100%); color: #fff; border-radius: 18px;
        padding: 26px 30px 22px; margin-bottom: 18px; box-shadow: 0 6px 24px rgba(11,37,69,.18); }}
.hero h1 {{ color: #fff; font-size: 2.0rem; font-weight: 800; margin: 0 0 6px 0; padding: 0; letter-spacing: -.5px; }}
.hero p {{ color: #DCE8F5; font-size: 1.0rem; margin: 0; }}
.hero .chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); border-radius: 999px;
        padding: 5px 12px; font-size: .82rem; color: #fff; }}
.kpi {{ background: #fff; border: 1px solid #E5E9F0; border-radius: 14px; padding: 14px 16px; height: 100%;
        box-shadow: 0 1px 3px rgba(16,24,40,.05); }}
.kpi .l {{ font-size: .78rem; color: #667085; text-transform: uppercase; letter-spacing: .4px; font-weight: 600; }}
.kpi .v {{ font-size: 1.75rem; font-weight: 800; color: {NAVY}; line-height: 1.25; margin-top: 4px; }}
.kpi .s {{ font-size: .8rem; color: #667085; margin-top: 2px; }}
.pill {{ display: inline-block; border-radius: 999px; padding: 3px 12px; font-weight: 700; font-size: .85rem; }}
.card {{ background: #fff; border: 1px solid #E5E9F0; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; }}
.card h4 {{ margin: 0 0 8px 0; color: {NAVY}; font-size: 1.02rem; }}
.sign {{ background: #FFF6F5; border-left: 4px solid #C0392B; padding: 7px 10px; border-radius: 6px; margin: 5px 0; font-size: .9rem; }}
.ok {{ background: #F0FAF4; border-left: 4px solid #1E8449; padding: 7px 10px; border-radius: 6px; margin: 5px 0; font-size: .9rem; }}
.gauge-note {{ font-size: .85rem; color: #475467; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E5E9F0; }}
.stTabs [data-baseweb="tab"] {{ background: #F5F7FA; border-radius: 10px 10px 0 0; padding: 8px 16px; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: {NAVY} !important; color: #fff !important; }}
.stTabs [aria-selected="true"] p {{ color: #fff !important; }}
div.stButton > button[kind="primary"] {{ background: {TEAL}; border: 0; border-radius: 10px; padding: .5rem 1.6rem; font-weight: 700; }}
.small {{ font-size: .82rem; color: #667085; }}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub='', color=None):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v' style='color:{color or NAVY}'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def pill(tier):
    return f"<span class='pill' style='background:{TIER_BG[tier]};color:{TIER_COLOR[tier]}'>{tier}</span>"


def style_fig(fig, h=340):
    fig.update_layout(height=h, margin=dict(t=48, b=10, l=10, r=10), font=dict(family='Inter'), plot_bgcolor='#fff',
                      paper_bgcolor='#fff', legend=dict(orientation='h', y=-0.18), title_font=dict(size=15, color=NAVY))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#EEF1F5')
    return fig


# ------------------------------------------------------------------ daily pipeline (shared by all visitors)
@st.cache_resource
def _store():
    return {'lock': threading.Lock(), 'day': None, 'out': None}


def daily_run(day: str, force=False):
    S = _store()
    if S['lock'].locked():
        wait = st.info('The daily data refresh is running for another visitor - this page will load in a moment...')
    else:
        wait = None
    with S['lock']:
        if wait:
            wait.empty()
        if force or S['day'] != day:
            box = st.status(f'Running the model for {day} - first visit today, about a minute...', expanded=True)
            say = lambda m: box.write(m)
            try:
                env = E.load_all(online=True, force=force, progress=say)
                df = E.build_panel(env, progress=say)
                tt = E.train_test(df, progress=say)
                sc = E.score_latest(df, tt['final'], env['data_end'])
                say('Explaining the model')
                extra = dict(imp=E.importance(tt['final']), dir=E.risk_direction(df), tiers=E.tier_backtest(tt['test']))
            except Exception as e:
                box.update(label='Refresh failed', state='error')
                if S['out'] is None:
                    st.error(f'The model could not be run right now ({type(e).__name__}: {e}). Please try again in a few minutes.')
                    st.stop()
                st.warning(f'Today\'s data refresh failed ({type(e).__name__}); showing the last successful run.')
                return S['out']
            box.update(label="Done - model trained, tested and today's scores are ready", state='complete', expanded=False)
            S['out'], S['day'], S['at'] = (env, df, tt, sc, extra), day, pd.Timestamp.now()
        return S['out']


day = f'{D.today():%Y-%m-%d}'
env, df, tt, sc, X = daily_run(day, force=st.session_state.pop('force', False))
de = env['data_end']
horizon = de + pd.DateOffset(years=1)
res = tt['results']
best = res.loc[E.BEST]
tb = X['tiers']

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class='hero'>
  <h1>📉 Mutual Fund Underperformance Early-Warning</h1>
  <p>Machine-learning scores for Indian equity funds (Direct, Growth): the chance each fund <b>trails its benchmark
  Total Return Index over the next 12 months</b>, from {de:%d %b %Y} to {horizon:%d %b %Y}.</p>
  <div class='chips'>
    <span class='chip'>📅 Run date {D.today():%d %b %Y}</span>
    <span class='chip'>🗂 Data to {de:%d %b %Y}</span>
    <span class='chip'>🤖 {E.BEST}</span>
    <span class='chip'>🎯 Test ROC-AUC {best['ROC-AUC']:.3f}</span>
    <span class='chip'>🧪 Backtest: High tier lagged {tb.loc['High', 'Underperformed']:.0%} vs Low tier {tb.loc['Low', 'Underperformed']:.0%}</span>
  </div>
</div>""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader('Data status')
    st.write(f"**NAVs:** mfapi.in, up to {env['nav_last'].max():%d %b %Y}")
    st.write(f"**Index TRI:** {env['tri_info']['source']}; official to {env['tri_info']['official_to']:%d %b %Y}")
    if env['tri_info']['proxy_days']:
        st.caption(f"Last {env['tri_info']['proxy_days']} day(s) of TRI extended with index-fund NAVs.")
    st.write('**Expense ratio / AUM:** AMFI disclosures stored with the app')
    if env['stale']:
        st.warning('NAV older than 5 days for: ' + ', '.join(env['stale']))
    last = _store().get('at')
    cool = last is not None and pd.Timestamp.now() - last < pd.Timedelta(minutes=15)
    if st.button('🔄 Re-download data now', width='stretch', disabled=cool,
                 help='Available 15 minutes after the last refresh' if cool else 'Downloads the latest NAVs and index values and reruns the model'):
        st.session_state['force'] = True
        st.rerun()
    st.divider()
    st.caption('Student project · PGDM-BIFS, Goa Institute of Management. Not investment advice.')

tabs = st.tabs(['🔍 Fund check', '⚖️ Compare funds', '💼 Portfolio check', '📋 Watchlist', '📊 Model & backtest', 'ℹ️ Method'])


# ------------------------------------------------------------------ shared views
def gauge(p, tier, universe):
    # colour bands = today's tier cut-offs (tiers are relative ranks, not fixed probability levels)
    q = universe['P'].sort_values(ascending=False).to_numpy()
    n = len(q)
    cut = lambda f: float(q[max(int(round(f * n)) - 1, 0)]) * 100
    hi, el, wa = cut(0.2), cut(0.5), cut(0.8)
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=p * 100, number={'suffix': '%', 'font': {'size': 40, 'color': TIER_COLOR[tier]}},
        gauge={'axis': {'range': [0, 100], 'ticksuffix': '%'}, 'bar': {'color': TIER_COLOR[tier], 'thickness': .28},
               'steps': [{'range': [0, wa], 'color': '#E9F7EF'}, {'range': [wa, el], 'color': '#FEF9E7'},
                         {'range': [el, hi], 'color': '#FEF5E7'}, {'range': [hi, 100], 'color': '#FDEDEC'}],
               'borderwidth': 0}))
    fig.update_layout(height=220, margin=dict(t=20, b=0, l=45, r=45), font=dict(family='Inter'))
    return fig


def price_charts(code, bench, s=None, years=3):
    f = (s if s is not None else env['nav'][code]).loc[:de]
    b = env['tri'][bench].loc[:de]
    j = pd.concat([f, b], axis=1, join='inner').loc[de - pd.DateOffset(years=years):]
    j.columns = ['Fund', bench]
    g = j / j.iloc[0] * 100
    fig = go.Figure([go.Scatter(x=g.index, y=g['Fund'], name='Fund', line=dict(color=BLUE, width=2.4)),
                     go.Scatter(x=g.index, y=g[bench], name=bench, line=dict(color='#98A2B3', width=1.8, dash='dot'))])
    fig.update_layout(title=f'Growth of Rs 100 · last {years} years', yaxis_title='Rs')
    m = pd.concat([f, b], axis=1, join='inner').resample('ME').last().dropna()
    roll = (m.iloc[:, 0].pct_change(12) - m.iloc[:, 1].pct_change(12)).loc[de - pd.DateOffset(years=5):].dropna()
    fig2 = go.Figure(go.Bar(x=roll.index, y=roll.values, marker_color=np.where(roll.values < 0, '#C0392B', '#1E8449')))
    fig2.update_layout(title='Rolling 1-year excess return vs index', yaxis_tickformat='.0%')
    dd = (f / f.cummax() - 1).loc[de - pd.DateOffset(years=years):]
    bd = (b / b.cummax() - 1).loc[de - pd.DateOffset(years=years):]
    fig3 = go.Figure([go.Scatter(x=dd.index, y=dd.values, name='Fund', fill='tozeroy', line=dict(color='#C0392B', width=1.4)),
                      go.Scatter(x=bd.index, y=bd.values, name=bench, line=dict(color='#98A2B3', width=1.4, dash='dot'))])
    fig3.update_layout(title='Drawdown from previous peak', yaxis_tickformat='.0%')
    return [style_fig(x, 320) for x in (fig, fig2, fig3)]


def show_fund(r, universe, note=None, s=None, own=True):
    tier = r['Risk Tier']
    st.markdown(f"### {r['Fund']} &nbsp;{pill(tier)}", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>{r['Category']} · {r['AMC']} · scheme code {int(r['Scheme Code'])} · "
                f"benchmark {r['Benchmark']}</div>", unsafe_allow_html=True)
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge(r['P'], tier, universe), width='stretch', config={'displayModeBar': False})
        st.markdown(f"<div class='gauge-note'>Chance of trailing <b>{r['Benchmark']}</b> from {de:%d %b %Y} to "
                    f"{horizon:%d %b %Y}. Band colours show today's tier cut-offs.</div>", unsafe_allow_html=True)
    with g2:
        k = st.columns(3)
        kpi(k[0], 'Rank (1 = riskiest)', f"{int(r['Rank'])}/{len(universe)}", 'among funds scored today')
        cat = universe[universe['Category'] == r['Category']]
        kpi(k[1], 'Rank in category', f"{int((cat['P'] > r['P']).sum() + 1)}/{len(cat)}", r['Category'])
        kpi(k[2], 'Warning signs', f"{int(r['Warning Signs'])}/9", 'simple rule checks',
            '#C0392B' if r['Warning Signs'] >= 3 else NAVY)
        k = st.columns(3)
        kpi(k[0], 'Excess return 1Y', spct(r['Excess_1Y']), 'fund minus index', '#C0392B' if r['Excess_1Y'] < 0 else '#1E8449')
        kpi(k[1], 'Excess return 3Y', spct(r['Excess_3Y']), 'annualised', '#C0392B' if r['Excess_3Y'] < 0 else '#1E8449')
        kpi(k[2], 'Expense ratio', f"{r['TER_Direct']:.2f}%", f"{r['TER_vs_Cat_Median']:+.2f} pp vs peers")
        # backtest context for this tier
        st.markdown(f"<div class='small' style='margin-top:10px'>📌 In the 2025 backtest, funds placed in the <b>{tier}</b> tier "
                    f"went on to underperform <b>{tb.loc[tier, 'Underperformed']:.0%}</b> of the time "
                    f"(all funds: {tt['base_test']:.0%}).</div>", unsafe_allow_html=True)
    if note:
        st.info(note)
    a, b = st.columns([1, 1.25])
    with a:
        st.markdown("<div class='card'><h4>⚠️ Warning signs</h4>" +
                    (''.join(f"<div class='sign'>{x}</div>" for x in r['Signs']) if r['Signs']
                     else "<div class='ok'>None of the 9 rules triggered</div>") + '</div>', unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><h4>🧠 Why this score: the model's most important factors</h4></div>", unsafe_allow_html=True)
        dv = E.drivers(r, universe, X['imp'], X['dir'])
        st.dataframe(dv, hide_index=True, width='stretch',
                     column_config={'Importance': st.column_config.ProgressColumn('Weight in model', format='%.3f',
                                                                                  min_value=0, max_value=float(X['imp'].max()))})
        st.caption('Effect = whether this fund\'s value sits on the side of its category median that went with more '
                   'underperformance in the training years. A guide, not an exact decomposition.')
    figs = price_charts(r['Scheme Code'], r['Benchmark'], s)
    c = st.columns(3)
    for col, fg in zip(c, figs):
        col.plotly_chart(fg, width='stretch', config={'displayModeBar': False})
    if own:
        h = tt['test'][tt['test']['Fund'] == r['Fund']].sort_values('Obs Date')
        if len(h):
            st.markdown("<div class='card'><h4>🕰 Track record on this fund (out-of-sample)</h4>"
                        "<div class='small'>Scores the model gave in 2025, before it saw the outcome. ● = actually underperformed, "
                        "○ = beat the index. The star is today's score.</div></div>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_hrect(y0=0.5, y1=1, fillcolor='#FDEDEC', opacity=.5, line_width=0)
            fig.add_trace(go.Scatter(x=h['Obs Date'], y=h['P'], mode='lines', line=dict(color=BLUE, width=2), name='Score'))
            fig.add_trace(go.Scatter(x=h['Obs Date'], y=h['P'], mode='markers', name='Outcome',
                                     marker=dict(size=12, symbol=np.where(h[E.TARGET] == 1, 'circle', 'circle-open'),
                                                 color=np.where(h[E.TARGET] == 1, '#C0392B', '#1E8449'), line=dict(width=2)),
                                     text=[f"next-12M excess {v:+.1%}" for v in h['Fwd_Excess_1Y']], hovertemplate='%{x|%b %Y}: %{y:.0%}<br>%{text}'))
            fig.add_trace(go.Scatter(x=[de], y=[r['P']], mode='markers', name='Today',
                                     marker=dict(size=18, symbol='star', color=TIER_COLOR[tier])))
            fig.update_layout(yaxis=dict(range=[0, 1], tickformat='.0%'), showlegend=False)
            c1, c2 = st.columns([2, 1])
            c1.plotly_chart(style_fig(fig, 300), width='stretch', config={'displayModeBar': False})
            hit = ((h['P'] >= 0.5) == (h[E.TARGET] == 1)).mean()
            c2.markdown(f"<div class='kpi'><div class='l'>Calls correct at 50%</div><div class='v'>{hit:.0%}</div>"
                        f"<div class='s'>{len(h)} monthly predictions in 2025</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='kpi' style='margin-top:10px'><div class='l'>Actually underperformed</div>"
                        f"<div class='v'>{int(h[E.TARGET].sum())}/{len(h)}</div><div class='s'>months</div></div>", unsafe_allow_html=True)
    rep = pd.DataFrame({'Item': ['Fund', 'Scheme code', 'Category', 'Benchmark', 'Data date', 'Horizon end', 'P(underperform)',
                                 'Risk tier', 'Rank', 'Warning signs'] + [E.NICE[c] for c in E.NUM_FEATURES],
                        'Value': [r['Fund'], int(r['Scheme Code']), r['Category'], r['Benchmark'], f'{de:%d-%b-%Y}',
                                  f'{horizon:%d-%b-%Y}', f"{r['P']:.4f}", tier, int(r['Rank']), '; '.join(r['Signs']) or 'none']
                                 + [E.fmt_feat(c, r[c]) for c in E.NUM_FEATURES]})
    st.download_button('⬇ Download this fund report (CSV)', rep.to_csv(index=False).encode(),
                       f"fund_report_{int(r['Scheme Code'])}_{de:%d%b%Y}.csv", 'text/csv')


# ------------------------------------------------------------------ 1. Fund check
with tabs[0]:
    mode = st.radio('Which fund?', ['One of the 30 project funds', 'Any other equity fund (by name or scheme code)'], horizontal=True)
    if mode.startswith('One'):
        c1, c2 = st.columns([4, 1])
        pick = c1.selectbox('Fund', sc.sort_values('Fund')['Fund'].tolist(), index=None, placeholder='Type to search the 30 funds',
                            label_visibility='collapsed')
        if c2.button('▶ Run', type='primary', disabled=pick is None, width='stretch'):
            st.session_state['res'] = ('own', pick)
    else:
        st.caption('Large cap funds are judged against NIFTY 100 TRI, mid cap against NIFTY MIDCAP 150 TRI, flexi/multi cap against '
                   'NIFTY 500 TRI. Choose the Direct - Growth plan.')
        q = st.text_input('Search by fund name', placeholder='e.g. Quant Flexi Cap')
        code_default = None
        if q:
            try:
                hits = D.search_schemes(q)
                dg = [h for h in hits if 'direct' in h['schemeName'].lower() and 'growth' in h['schemeName'].lower()]
                hits = dg or hits
                if hits:
                    sel = st.selectbox('Matching schemes', hits[:30], format_func=lambda h: f"{h['schemeName']}  ·  {h['schemeCode']}")
                    code_default = int(sel['schemeCode'])
                else:
                    st.warning('No schemes found - try fewer words.')
            except Exception as e:
                st.warning(f'Search is unavailable right now ({type(e).__name__}); type the scheme code instead.')
        a, b, c3, d4 = st.columns(4)
        code = a.number_input('Scheme code', min_value=100000, max_value=999999, step=1, value=code_default)
        guess = 2
        if code_default and q:
            n = sel['schemeName'].lower()
            if 'large & mid' in n or 'large and mid' in n or 'multi' in n or 'flexi' in n or 'focused' in n:
                guess = 2
            elif 'mid' in n and 'small' not in n:
                guess = 1
            elif 'large' in n or 'bluechip' in n or 'top 100' in n:
                guess = 0
        cat = b.selectbox('Category', list(D.BM.keys()), index=guess)
        ter = c3.number_input('Expense ratio % (optional)', min_value=0.0, max_value=3.0, value=None, step=0.01,
                              help='Direct plan TER. Blank = category median.')
        aum = d4.number_input('AUM Rs crore (optional)', min_value=1.0, value=None, step=100.0, help='Blank = treated as unknown.')
        if st.button('▶ Run', type='primary', disabled=not code):
            st.session_state['res'] = ('new', {'Scheme Code': int(code), 'Category': cat, 'TER': ter, 'AUM_Cr': aum})

    rq = st.session_state.get('res')
    if rq and ((rq[0] == 'own') != mode.startswith('One')):
        rq = None
    if not rq:
        st.markdown("<div class='card'><h4>How to use</h4>Pick a fund and press <b>Run</b>. You'll see the probability of "
                    "underperforming over the next 12 months, its risk tier, the warning signs, the factors behind the score, "
                    "charts and the model's past track record on that fund.</div>", unsafe_allow_html=True)
        c = st.columns(4)
        for col, t in zip(c, TIERS):
            n = int((sc['Risk Tier'] == t).sum())
            col.markdown(f"<div class='kpi' style='border-top:4px solid {TIER_COLOR[t]}'><div class='l'>{t} tier today</div>"
                         f"<div class='v' style='color:{TIER_COLOR[t]}'>{n} funds</div>"
                         f"<div class='s'>backtest: {tb.loc[t, 'Underperformed']:.0%} went on to lag</div></div>", unsafe_allow_html=True)
    elif rq[0] == 'own':
        st.divider()
        show_fund(sc[sc['Fund'] == rq[1]].iloc[0], sc)
    else:
        st.divider()
        fd = rq[1]
        try:
            with st.spinner('Downloading the NAV history and scoring the fund...'):
                out, code, src, missing, s = E.score_new_fund(env, tt['final'], fd)
            r = out[out['Scheme Code'] == code].iloc[0]
            note = []
            if code in set(env['funds']['Scheme Code']):
                note.append('This is one of the 30 project funds, so its known category and benchmark were used.')
            if r['TER_Imputed'] == 1:
                note.append('Expense ratio not available: category median used.')
            if pd.isna(r['AUM_Cr']):
                note.append('AUM not available: size features filled with typical values.')
            if missing:
                note.append(f'Short history: {len(missing)} long-window features filled with typical values.')
            note.append(f'Rank and tier are relative to the {len(out)} funds scored together.')
            show_fund(r, out, ' '.join(note), s=s, own=False)
        except Exception as e:
            st.error(f'Could not score scheme {fd["Scheme Code"]}: {e}')

# ------------------------------------------------------------------ 2. Compare
with tabs[1]:
    st.markdown('Compare up to **4 funds** side by side.')
    default = sc.sort_values('Rank')['Fund'].iloc[[0, -1]].tolist()
    sel = st.multiselect('Funds', sc.sort_values('Fund')['Fund'].tolist(), default=default, max_selections=4)
    if sel:
        sub = sc.set_index('Fund').loc[sel].reset_index()
        cols = st.columns(len(sub))
        for col, (_, r) in zip(cols, sub.iterrows()):
            col.markdown(f"<div class='kpi' style='border-top:5px solid {TIER_COLOR[r['Risk Tier']]}'>"
                         f"<div class='l'>{r['Category']}</div><div style='font-weight:700;color:{NAVY};min-height:44px'>{r['Fund']}</div>"
                         f"<div class='v' style='color:{TIER_COLOR[r['Risk Tier']]}'>{r['P']:.0%}</div>"
                         f"<div class='s'>{r['Risk Tier']} · rank {int(r['Rank'])}/30 · {int(r['Warning Signs'])} signs</div></div>",
                         unsafe_allow_html=True)
        rows = ['Excess_1Y', 'Excess_3Y', 'IR_1Y', 'Hit_Rate_12M', 'Fund_Vol_1Y', 'TE_1Y', 'Beta_3Y', 'Down_Capture_3Y',
                'Fund_MaxDD_3Y', 'TER_Direct', 'Log_AUM', 'Net_Flow_1Y', 'Fund_Age_Yrs']
        tbl = pd.DataFrame({r['Fund']: [E.fmt_feat(c, r[c]) for c in rows] for _, r in sub.iterrows()}, index=[E.NICE[c] for c in rows])
        st.dataframe(tbl, width='stretch')
        yrs = st.select_slider('Chart period', options=[1, 3, 5], value=3, format_func=lambda y: f'{y} years')
        fig = go.Figure()
        for _, r in sub.iterrows():
            f = env['nav'][r['Scheme Code']].loc[de - pd.DateOffset(years=yrs):de]
            fig.add_trace(go.Scatter(x=f.index, y=f / f.iloc[0] * 100, name=r['Fund']))
        for bname in sub['Benchmark'].unique():
            b = env['tri'][bname].loc[de - pd.DateOffset(years=yrs):de]
            fig.add_trace(go.Scatter(x=b.index, y=b / b.iloc[0] * 100, name=bname, line=dict(dash='dot', color='#98A2B3')))
        fig.update_layout(title=f'Growth of Rs 100 · last {yrs} years', yaxis_title='Rs')
        st.plotly_chart(style_fig(fig, 420), width='stretch')

# ------------------------------------------------------------------ 3. Portfolio
with tabs[2]:
    st.markdown('Enter the funds you hold (from the 30) and the amount in each. The app scores the **portfolio as a whole** '
                'and suggests lower-risk alternatives in the same category.')
    if 'pf' not in st.session_state:
        st.session_state['pf'] = pd.DataFrame({'Fund': sc.sort_values('Fund')['Fund'].iloc[[0, 9, 20]].tolist(), 'Amount (Rs)': [50000, 30000, 20000]})
    pf = st.data_editor(st.session_state['pf'], num_rows='dynamic', width='stretch', key='pf_ed',
                        column_config={'Fund': st.column_config.SelectboxColumn('Fund', options=sc.sort_values('Fund')['Fund'].tolist(), required=True),
                                       'Amount (Rs)': st.column_config.NumberColumn('Amount (Rs)', min_value=0, step=1000, format='%d')})
    pf = pf.dropna().groupby('Fund', as_index=False)['Amount (Rs)'].sum()
    pf = pf[pf['Amount (Rs)'] > 0]
    if len(pf):
        m = pf.merge(sc, on='Fund')
        w = m['Amount (Rs)'] / m['Amount (Rs)'].sum()
        wp = (w * m['P']).sum()
        hi = w[m['Risk Tier'].isin(['High', 'Elevated'])].sum()
        k = st.columns(4)
        kpi(k[0], 'Portfolio value', f"Rs {m['Amount (Rs)'].sum():,.0f}", f'{len(m)} funds')
        kpi(k[1], 'Weighted risk score', pct(wp, 0), 'avg P(underperform), by amount',
            TIER_COLOR['High'] if wp >= .7 else TIER_COLOR['Elevated'] if wp >= .55 else TIER_COLOR['Low'])
        kpi(k[2], 'In High / Elevated tiers', pct(hi, 0), 'share of money', '#C0392B' if hi > .5 else NAVY)
        kpi(k[3], 'Weighted expense ratio', f"{(w * m['TER_Direct']).sum():.2f}%", 'per year')
        c1, c2 = st.columns([1, 1.4])
        tiers = m.assign(w=w).groupby('Risk Tier')['w'].sum().reindex(TIERS).dropna()
        fig = go.Figure(go.Pie(labels=tiers.index, values=tiers.values, hole=.55, marker_colors=[TIER_COLOR[t] for t in tiers.index],
                               sort=False, textinfo='label+percent'))
        fig.update_layout(title='Money by risk tier', showlegend=False)
        c1.plotly_chart(style_fig(fig, 330), width='stretch')
        with c2:
            st.markdown("<div class='card'><h4>💡 Suggestions</h4></div>", unsafe_allow_html=True)
            flagged = m[m['Risk Tier'].isin(['High', 'Elevated'])].sort_values('P', ascending=False)
            if not len(flagged):
                st.success('No holding is in the High or Elevated tier.')
            for _, r in flagged.iterrows():
                alt = sc[(sc['Category'] == r['Category']) & (sc['Risk Tier'].isin(['Low', 'Watch'])) & (~sc['Fund'].isin(m['Fund']))]
                alt = alt.sort_values('P').head(2)
                alts = ', '.join(f"{a['Fund']} ({a['P']:.0%})" for _, a in alt.iterrows()) or 'none in the 30-fund list'
                st.markdown(f"<div class='sign'><b>{r['Fund']}</b> - {r['Risk Tier']} ({r['P']:.0%}). "
                            f"Lower-risk {r['Category'].lower()} peers: {alts}</div>", unsafe_allow_html=True)
        st.dataframe(m.assign(Weight=w).rename(columns={'P': 'P(underperform)', 'Excess_1Y': 'Excess 1Y'})[['Fund', 'Category', 'Amount (Rs)', 'Weight', 'P(underperform)', 'Risk Tier', 'Warning Signs', 'Excess 1Y']]
                     .sort_values('P(underperform)', ascending=False).style.format({'Weight': '{:.0%}', 'P(underperform)': '{:.1%}', 'Excess 1Y': '{:+.1%}', 'Amount (Rs)': '{:,.0f}'})
                     .map(lambda v: f'color:{TIER_COLOR.get(v, "")};font-weight:700', subset=['Risk Tier']),
                     hide_index=True, width='stretch')
        st.caption('Suggestions are model-based screens, not recommendations to buy or sell.')

# ------------------------------------------------------------------ 4. Watchlist
with tabs[3]:
    c1, c2, c3 = st.columns([2, 2, 1])
    cats = c1.multiselect('Category', list(D.BM.keys()), default=list(D.BM.keys()))
    tiers_sel = c2.multiselect('Risk tier', TIERS, default=TIERS)
    min_signs = c3.number_input('Min. warning signs', 0, 9, 0)
    t = sc[sc['Category'].isin(cats) & sc['Risk Tier'].isin(tiers_sel) & (sc['Warning Signs'] >= min_signs)]
    view = t.assign(AUM_Cr=t['AUM_Cr'].round(0))[['Rank', 'Fund', 'Category', 'P', 'Risk Tier', 'Warning Signs', 'Excess_1Y', 'Excess_3Y', 'IR_1Y', 'TER_Direct', 'AUM_Cr']]
    st.dataframe(view, hide_index=True, width='stretch', height=min(1100, 40 + 35 * len(view)),
                 column_config={'P': st.column_config.ProgressColumn('P(underperform 12M)', format='percent', min_value=0, max_value=1),
                                'Excess_1Y': st.column_config.NumberColumn('Excess 1Y', format='percent'),
                                'Excess_3Y': st.column_config.NumberColumn('Excess 3Y (ann.)', format='percent'),
                                'IR_1Y': st.column_config.NumberColumn('Info ratio 1Y', format='%.2f'),
                                'TER_Direct': st.column_config.NumberColumn('TER %', format='%.2f'),
                                'AUM_Cr': st.column_config.NumberColumn('AUM Rs cr', format='%,.0f')})
    out = sc.drop(columns=['Signs']).assign(Signs_list=sc['Signs'].str.join('; '))
    st.download_button('⬇ Download all scores (CSV)', out.to_csv(index=False).encode(), f'early_warning_scores_{de:%d%b%Y}.csv', 'text/csv')
    c1, c2 = st.columns(2)
    fig = px.scatter(sc, x='Excess_1Y', y='P', color='Risk Tier', color_discrete_map=TIER_COLOR, size='AUM_Cr', size_max=28,
                     hover_name='Fund', category_orders={'Risk Tier': TIERS},
                     labels={'Excess_1Y': 'Excess return last 1Y', 'P': 'P(underperform next 12M)'})
    fig.update_layout(title='Risk score vs last year\'s excess return (bubble = AUM)', xaxis_tickformat='.0%', yaxis_tickformat='.0%')
    fig.add_vline(x=0, line_dash='dot', line_color='#98A2B3')
    c1.plotly_chart(style_fig(fig, 400), width='stretch')
    g = sc.groupby('Category')['P'].agg(['mean', 'min', 'max']).reindex(list(D.BM.keys()))
    fig = go.Figure(go.Bar(x=g.index, y=g['mean'], marker_color=[BLUE, TEAL, NAVY], text=[f'{v:.0%}' for v in g['mean']], textposition='inside', insidetextanchor='start', textfont=dict(color='white', size=14),
                           error_y=dict(type='data', symmetric=False, array=g['max'] - g['mean'], arrayminus=g['mean'] - g['min'])))
    fig.update_layout(title='Average score by category (bars show min-max)', yaxis_tickformat='.0%', yaxis_range=[0, 1])
    c2.plotly_chart(style_fig(fig, 400), width='stretch')

# ------------------------------------------------------------------ 5. Model & backtest
with tabs[4]:
    ts, te_ = tt['train_span'], tt['test_span']
    k = st.columns(4)
    kpi(k[0], 'Training data', f"{tt['n_train']:,}", f"fund-months · {ts[0]:%b %Y} – {ts[1]:%b %Y}")
    kpi(k[1], 'Test data (unseen)', f"{tt['n_test']:,}", f"predictions {te_[0]:%b %Y} – {te_[1]:%b %Y}")
    kpi(k[2], 'Test ROC-AUC', f"{best['ROC-AUC']:.3f}", '0.5 = coin toss')
    kpi(k[3], 'Test accuracy', pct(best['Accuracy']), f"base rate {tt['base_test']:.0%} underperformed")
    st.markdown('')
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=tb.index, y=tb['Underperformed'], marker_color=[TIER_COLOR[t] for t in tb.index],
                           text=[f'{v:.0%}' for v in tb['Underperformed']], textposition='outside'))
    fig.add_hline(y=tt['base_test'], line_dash='dot', annotation_text=f"all funds {tt['base_test']:.0%}")
    fig.update_layout(title='Backtest: share that actually underperformed, by tier', yaxis_tickformat='.0%', yaxis_range=[0, 1])
    c1.plotly_chart(style_fig(fig, 360), width='stretch')
    fig = go.Figure(go.Bar(x=tb.index, y=tb['Avg_Excess'], marker_color=[TIER_COLOR[t] for t in tb.index],
                           text=[f'{v:+.1%}' for v in tb['Avg_Excess']], textposition='outside'))
    fig.update_layout(title='Backtest: average next-12M excess return, by tier', yaxis_tickformat='.1%')
    c2.plotly_chart(style_fig(fig, 360), width='stretch')
    st.caption('Each 2025 month the funds were tiered by the model score (no hindsight), then followed for 12 months. '
               'High = riskiest 20%, Low = safest 20%.')

    st.markdown('#### Model comparison on the test set')
    show = res[['ROC-AUC', 'Within-month AUC', 'Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1']].astype(float)
    st.dataframe(show.style.format('{:.3f}').highlight_max(axis=0, color='#D5F5E3'), width='stretch')
    st.caption('ROC-AUC: chance the model ranks a real underperformer above a real outperformer. Within-month AUC compares only '
               'funds on the same date. Baseline = "last year\'s laggard stays a laggard".')
    st.caption('Note: Gradient Boosting samples rows at random while training, so its test AUC depends on the random seed. '
               'In the Nov-2026 run, 20 different seeds gave test AUCs from 0.60 to 0.68 (average 0.64); this app uses the '
               'project seed (42). All seeds stayed well above the 0.5 coin-toss level and the 0.43 baseline.')

    st.markdown(f'#### Confusion matrix · {E.BEST}')
    thr = st.slider('Decision threshold - flag a fund when its score is at least', 0.05, 0.95, 0.50, 0.05)
    tst = tt['test']
    m = E.metrics(tst[E.TARGET].astype(int), tst['P'].to_numpy(), thr)
    l, r_ = st.columns([1.1, 1])
    cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
    lab = np.array([['True negative', 'False alarm'], ['Missed laggard', 'Caught laggard']])
    fig = go.Figure(go.Heatmap(z=cm, x=['Predicted: beats index', 'Predicted: underperforms'], y=['Actual: beat index', 'Actual: underperformed'],
                               colorscale=[[0, '#EAF2FB'], [1, BLUE]], showscale=False,
                               text=[[f'<b>{cm[i, j]}</b><br>{lab[i, j]}' for j in range(2)] for i in range(2)],
                               texttemplate='%{text}', textfont=dict(size=15)))
    fig.update_yaxes(autorange='reversed')
    l.plotly_chart(style_fig(fig, 330), width='stretch')
    with r_:
        k = st.columns(2)
        kpi(k[0], 'Accuracy', pct(m['Accuracy']))
        kpi(k[1], 'Precision', pct(m['Precision']), 'flags that were right')
        k = st.columns(2)
        kpi(k[0], 'Recall', pct(m['Recall']), 'laggards caught')
        kpi(k[1], 'Funds flagged', f"{int(m['TP'] + m['FP'])}/{len(tst)}", f"F1 {m['F1']:.3f}")
    c1, c2 = st.columns(2)
    fpr, tpr = tt['roc']
    fig = go.Figure([go.Scatter(x=fpr, y=tpr, name=f"{E.BEST} (AUC {best['ROC-AUC']:.3f})", fill='tozeroy', line=dict(color=BLUE)),
                     go.Scatter(x=[0, 1], y=[0, 1], name='Random guess', line=dict(dash='dash', color='grey'))])
    fig.update_layout(title='ROC curve (test set)', xaxis_title='False positive rate', yaxis_title='True positive rate')
    c1.plotly_chart(style_fig(fig, 380), width='stretch')
    imp = X['imp'].head(12)[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=[E.NICE.get(i, i) for i in imp.index], orientation='h', marker_color=TEAL))
    fig.update_layout(title='Top 12 factors the model relies on (feature importance)')
    c2.plotly_chart(style_fig(fig, 380), width='stretch')
    q = tst.assign(Q=tst['Obs Date'].dt.to_period('Q').astype(str)).groupby('Q').apply(
        lambda g: pd.Series({'Rows': len(g), 'Underperformed': g[E.TARGET].mean(),
                             'AUC': E.roc_auc_score(g[E.TARGET], g['P']) if g[E.TARGET].nunique() == 2 else np.nan}))
    fig = go.Figure([go.Bar(x=q.index, y=q['AUC'], name='ROC-AUC', marker_color=BLUE, text=[f'{v:.2f}' for v in q['AUC']], textfont=dict(color='white')),
                     go.Scatter(x=q.index, y=q['Underperformed'], name='Share underperformed', mode='lines+markers',
                                line=dict(color='#C0392B'))])
    fig.add_hline(y=.5, line_dash='dot', line_color='#98A2B3')
    fig.update_layout(title='Results by prediction quarter', yaxis_range=[0, 1])
    st.plotly_chart(style_fig(fig, 340), width='stretch')

# ------------------------------------------------------------------ 6. Method
with tabs[5]:
    c1, c2 = st.columns(2)
    c1.markdown("""
<div class='card'><h4>❓ The question</h4>Will the fund's return over the next 12 months be <b>lower than its benchmark
Total Return Index</b> (dividends reinvested)? The model answers with a probability.</div>
<div class='card'><h4>🗄 Data (refreshed daily)</h4>
• NAV history: <a href='https://www.mfapi.in' target='_blank'>mfapi.in</a> (AMFI data)<br>
• NIFTY 100 / MIDCAP 150 / 500 TRI: niftyindices.com (gaps filled with index-fund NAVs)<br>
• Expense ratio and AUM: AMFI disclosures</div>
<div class='card'><h4>🧾 46 features, point-in-time</h4>Trailing and excess returns (3M-3Y), volatility, tracking error, beta,
drawdowns, information ratio, hit-rate, up/down capture, category ranks, fund age, expense ratio, AUM, AUM growth, net flows,
category. Every value uses only data available on that date.</div>""", unsafe_allow_html=True)
    c2.markdown(f"""
<div class='card'><h4>🧪 Train · Test · Predict</h4>
• <b>Train:</b> monthly snapshots Jun-2018 – Dec-2024<br>
• <b>Test:</b> 2025 snapshots, outcomes checked a year later (never seen in training)<br>
• <b>Predict:</b> refit on all labelled months, score {de:%d %b %Y} → {horizon:%d %b %Y}</div>
<div class='card'><h4>🤖 Model</h4>Gradient Boosting, chosen by walk-forward cross-validation on the training years from 8 models
(logistic & linear regression, decision tree, bagging, random forest, SVM, AdaBoost, gradient boosting).</div>
<div class='card'><h4>⚠️ Limitations</h4>30 funds, ~8 years, one test year. Probabilities are not calibrated, so use the ranks and
tiers. Survivorship bias (only funds alive today). Not investment advice.</div>""", unsafe_allow_html=True)
