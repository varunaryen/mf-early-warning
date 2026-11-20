"""Plain-language 'Model & backtest' tab: what drives each fund, and how well the warnings worked."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mfcore import engine as E

# plain-English name and a one-line meaning for every input the model uses
PLAIN = {
    'Fund_Ret_3M': ('3-month return', 'How much the fund made in the last 3 months'),
    'Excess_3M': ('Beat the index? (3 months)', 'Fund return minus index return, last 3 months'),
    'Fund_Ret_6M': ('6-month return', 'How much the fund made in the last 6 months'),
    'Excess_6M': ('Beat the index? (6 months)', 'Fund return minus index return, last 6 months'),
    'Fund_Ret_1Y': ('1-year return', 'How much the fund made in the last year'),
    'Excess_1Y': ('Beat the index? (1 year)', 'Fund return minus index return, last year'),
    'Fund_Ret_2Y': ('2-year return', 'Yearly return over the last 2 years'),
    'Excess_2Y': ('Beat the index? (2 years)', 'Yearly gap vs the index over 2 years'),
    'Fund_Ret_3Y': ('3-year return', 'Yearly return over the last 3 years'),
    'Excess_3Y': ('Beat the index? (3 years)', 'Yearly gap vs the index over 3 years'),
    'Fund_Vol_1Y': ('Ups and downs (1 year)', 'How bumpy the fund\'s returns were (volatility)'),
    'Fund_Vol_3Y': ('Ups and downs (3 years)', 'How bumpy the fund\'s returns were (volatility)'),
    'TE_1Y': ('Strays from index (1 year)', 'How differently the fund moves from its index (tracking error)'),
    'TE_3Y': ('Strays from index (3 years)', 'How differently the fund moves from its index (tracking error)'),
    'Beta_1Y': ('Market sensitivity (1 year)', 'How strongly the fund reacts when the market moves (beta)'),
    'Beta_3Y': ('Market sensitivity (3 years)', 'How strongly the fund reacts when the market moves (beta)'),
    'Fund_MaxDD_1Y': ('Worst fall (1 year)', 'Biggest drop from a peak in the last year'),
    'Fund_MaxDD_3Y': ('Worst fall (3 years)', 'Biggest drop from a peak in 3 years'),
    'IR_1Y': ('Steady extra return (1 year)', 'Extra return earned per unit of straying (information ratio)'),
    'IR_3Y': ('Steady extra return (3 years)', 'Extra return earned per unit of straying (information ratio)'),
    'Ret_per_Vol_1Y': ('Return for the risk taken', 'Return earned per unit of ups and downs'),
    'Hit_Rate_12M': ('Months it beat the index (12)', 'Share of the last 12 months the fund beat its index'),
    'Hit_Rate_36M': ('Months it beat the index (36)', 'Share of the last 36 months the fund beat its index'),
    'Up_Capture_3Y': ('Gains in rising markets', 'How much of the index\'s up-months the fund captured'),
    'Down_Capture_3Y': ('Losses in falling markets', 'How much of the index\'s down-months the fund suffered'),
    'Excess_1Y_Change_6M': ('Getting better or worse?', 'Change in the 1-year gap vs the index over 6 months'),
    'Fund_Age_Yrs': ('Fund age', 'How many years the fund has existed'),
    'CatRank_Excess_1Y': ('Rank among peers (1 year)', 'Where its 1-year result sits among similar funds'),
    'CatRank_Excess_3Y': ('Rank among peers (3 years)', 'Where its 3-year result sits among similar funds'),
    'CatRank_IR_1Y': ('Steadiness rank among peers', 'Where its steady extra return ranks among peers'),
    'TER_Direct': ('Annual fee', 'Expense ratio charged every year'),
    'TER_vs_Cat_Median': ('Fee vs similar funds', 'Expense ratio compared with the category average'),
    'Log_AUM': ('Fund size', 'Total money managed by the fund (AUM)'),
    'AUM_Growth_1Y': ('Growth in fund size', 'How fast the fund\'s money grew in a year'),
    'Net_Flow_1Y': ('New money coming in', 'Fresh investor money in a year, as a share of the fund'),
    'CatRank_AUM': ('Size vs similar funds', 'How big the fund is compared with its peers'),
}
RED, GREEN, GREY, NAVY, BLUE, TEAL = '#C0392B', '#1E8449', '#98A2B3', '#0B2545', '#1F4E79', '#13817A'


def name(c):
    return PLAIN.get(c, (E.NICE.get(c, c), ''))[0]


def top_factors(imp, k=10):
    return [f for f in imp.index if f in PLAIN][:k]


def factor_pushes(sc, imp, direction, feats):
    """For every fund and factor: + pushes towards lagging, - towards beating (vs category peers, weighted by importance)."""
    out = pd.DataFrame(index=sc['Fund'], columns=feats, dtype=float)
    for cat, g in sc.groupby('Category'):
        for c in feats:
            v, med = g[c], g[c].median()
            spread = (g[c].quantile(.75) - g[c].quantile(.25)) or (g[c].std() or 1)
            z = ((v - med) / spread).clip(-2, 2).fillna(0) * direction[c]
            out.loc[g['Fund'], c] = (z * imp[c] / imp[feats].max()).to_numpy()
    return out


def card(html):
    st.markdown(f"<div class='card'>{html}</div>", unsafe_allow_html=True)


def render(tt, res, sc, X, tb, de, horizon, style_fig, kpi, TIER_COLOR, TIERS):
    best = res.loc[E.BEST]
    imp, direction = X['imp'], X['dir']
    ts, te_ = tt['train_span'], tt['test_span']
    feats = top_factors(imp, 10)
    base_rule = res.loc['Baseline: past-1Y laggard']
    tst = tt['test']
    flag = tst['P'] >= 0.5

    # ---------- 1. the idea in one line + how it was built
    st.markdown("### 🧭 How the early-warning works, in plain words")
    card("We showed a computer <b>six and a half years of history</b> for 30 funds: every month, what each fund looked like "
         "(returns, risk, fees, size, new money coming in) and whether it <b>went on to trail its index over the next 12 months</b>. "
         "It learned which combinations usually come before a fund falls behind. Then we tested it on <b>2025</b>, a year it had "
         "never seen, before trusting it with today's funds.")
    s1, s2, s3 = st.columns(3)
    step = lambda col, icon, title, big, sub, bg: col.markdown(
        f"<div class='kpi' style='background:{bg};text-align:center'><div style='font-size:1.8rem'>{icon}</div>"
        f"<div class='l'>{title}</div><div class='v'>{big}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)
    step(s1, '📚', '1 · Learn from the past', f"{tt['n_train']:,} snapshots",
         f"{ts[0]:%b %Y} – {ts[1]:%b %Y} · {tt['base_train']:.0%} of them lagged", '#F5F7FA')
    step(s2, '🧪', '2 · Exam on unseen data', f"{tt['n_test']:,} predictions",
         f"{te_[0]:%b %Y} – {te_[1]:%b %Y} · checked against what really happened", '#FEF5E7')
    step(s3, '🔮', '3 · Warn about the future', f"{len(sc)} funds scored",
         f"{de:%d %b %Y} → {horizon:%d %b %Y}", '#E9F7EF')

    # ---------- 2. how good is it, in plain numbers
    st.markdown("### 🎯 How good were the warnings in the 2025 exam?")
    k = st.columns(4)
    kpi(k[0], 'Right overall', f"{best['Accuracy'] * 10:.1f} in 10", 'funds called correctly (lag or not)')
    kpi(k[1], 'When it warned', f"{best['Precision'] * 10:.1f} in 10", 'warnings came true', RED)
    kpi(k[2], 'Laggards spotted', f"{best['Recall'] * 10:.1f} in 10", 'of funds that really lagged')
    kpi(k[3], 'Ranking skill', f"{best['ROC-AUC']:.2f}", '0.50 = coin toss · 1.00 = perfect', TEAL)
    c1, c2 = st.columns([1.15, 1])
    bars = pd.Series({'Our model': best['ROC-AUC'], 'Tossing a coin': 0.5,
                      '"Last year\'s loser loses again"': base_rule['ROC-AUC']}).sort_values()
    fig = go.Figure(go.Bar(x=bars.values, y=bars.index, orientation='h', text=[f'{v:.2f}' for v in bars.values],
                           textposition='outside', marker_color=[TEAL if i == 'Our model' else GREY for i in bars.index]))
    fig.update_layout(title='Ranking skill: does it put real laggards ahead of real winners?', xaxis_range=[0, 1])
    c1.plotly_chart(style_fig(fig, 300), width='stretch', config={'displayModeBar': False})
    with c2:
        n_flag, n_hit = int(flag.sum()), int((flag & (tst[E.TARGET] == 1)).sum())
        icons = ''.join(f"<span style='color:{RED if i < round(n_hit / max(n_flag, 1) * 10) else GREY};font-size:1.9rem'>●</span>"
                        for i in range(10))
        card(f"<h4>🔔 Out of every 10 warnings</h4>{icons}<div class='small' style='margin-top:6px'><b>{n_hit} of {n_flag}</b> "
             f"funds it warned about in 2025 really did trail their index over the next year. Without the model, picking at random, "
             f"only about <b>{tt['base_test'] * 10:.0f} in 10</b> would have.</div>")
        card("<h4>💡 What this means for you</h4>The model is a <b>smoke alarm, not a crystal ball</b>. It is clearly better than "
             "guessing, and much better than judging a fund by last year's return alone, but it will sometimes be wrong. "
             "Use it to decide <b>which funds to look at more closely</b>.")

    # ---------- 3. did the risk groups come true
    st.markdown("### 🚦 Did the risk groups come true?")
    st.markdown("<div class='small'>Every month of 2025 the funds were sorted into four groups by their score, before anyone knew "
                "the outcome. Then we waited a year and checked.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=tb.index, y=tb['Underperformed'], marker_color=[TIER_COLOR[t] for t in tb.index],
                           text=[f'{v:.0%}' for v in tb['Underperformed']], textposition='outside', textfont=dict(size=16)))
    fig.add_hline(y=tt['base_test'], line_dash='dot', line_color=GREY, annotation_text=f"all funds {tt['base_test']:.0%}")
    fig.update_layout(title='Share of funds that went on to trail their index', yaxis_tickformat='.0%', yaxis_range=[0, 1])
    c1.plotly_chart(style_fig(fig, 340), width='stretch', config={'displayModeBar': False})
    fig = go.Figure(go.Bar(x=tb.index, y=tb['Avg_Excess'], marker_color=[RED if v < 0 else GREEN for v in tb['Avg_Excess']],
                           text=[f'{v:+.1%}' for v in tb['Avg_Excess']], textposition='outside', textfont=dict(size=16)))
    fig.add_hline(y=0, line_color=GREY)
    fig.update_layout(title='Average return vs the index over the next 12 months', yaxis_tickformat='.1%')
    c2.plotly_chart(style_fig(fig, 340), width='stretch', config={'displayModeBar': False})
    hi, lo = tb.loc['High', 'Underperformed'], tb.loc['Low', 'Underperformed']
    st.success(f"High-risk funds fell behind **{hi:.0%}** of the time, against **{lo:.0%}** for Low-risk funds: about "
               f"**{hi / lo:.1f}×** as often. The groups worked in real life.")

    # ---------- 4. what drives lagging / leading
    st.markdown("### 🔍 What makes a fund fall behind or stay ahead?")
    c1, c2 = st.columns([1.1, 1])
    show = imp[feats][::-1]
    cols = [RED if direction[f] > 0 else GREEN for f in show.index]
    fig = go.Figure(go.Bar(x=show.values / show.max() * 100, y=[name(f) for f in show.index], orientation='h', marker_color=cols,
                           hovertext=[PLAIN[f][1] for f in show.index], hoverinfo='text+x'))
    fig.update_layout(title='The 10 signals the model leans on most (longest bar = most important)', xaxis_title='Relative weight',
                      xaxis_range=[0, 110])
    c1.plotly_chart(style_fig(fig, 430), width='stretch', config={'displayModeBar': False})
    with c2:
        rows = ''.join(
            f"<div style='display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #EEF1F5'>"
            f"<span style='color:{RED if direction[f] > 0 else GREEN};font-weight:800;width:18px'>{'▲' if direction[f] > 0 else '▼'}</span>"
            f"<div><b>{name(f)}</b><div class='small'>{PLAIN[f][1]}. Higher than peers → "
            f"{'<span style=color:' + RED + '>more</span>' if direction[f] > 0 else '<span style=color:' + GREEN + '>less</span>'} "
            f"risk of lagging.</div></div></div>" for f in feats)
        card("<h4>📖 What each signal means</h4>" + rows +
             "<div class='small' style='margin-top:8px'>🔴 red = a high value usually came <b>before</b> lagging · "
             "🟢 green = a high value usually came before <b>beating</b> the index (2018–2024 history).</div>"
             "<div class='small' style='margin-top:6px'>🤔 Some links may look surprising (for example, funds that move more "
             "with the market lagged less). They reflect what actually happened in Indian markets in 2018–2024, a mostly rising "
             "market, where funds that kept pace with the index tended not to fall behind.</div>")

    # ---------- 5. each fund, factor by factor
    st.markdown("### 🧩 Every fund, signal by signal")
    st.markdown("<div class='small'>Each fund is compared with similar funds in its own category today. "
                "<b style='color:#C0392B'>Red</b> = this signal pushes the fund towards falling behind; "
                "<b style='color:#1E8449'>green</b> = it helps the fund stay ahead. Darker = stronger push.</div>",
                unsafe_allow_html=True)
    push = factor_pushes(sc, imp, direction, feats)
    order = sc.sort_values('P', ascending=False)['Fund']
    z = push.loc[order].astype(float)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[name(f) for f in feats], y=[f"{f}  ({p:.0%})" for f, p in zip(order, sc.set_index('Fund').loc[order, 'P'])],
        colorscale=[[0, GREEN], [0.5, '#FFFFFF'], [1, RED]], zmid=0, zmin=-2, zmax=2, showscale=False, xgap=2, ygap=2,
        hovertemplate='%{y}<br>%{x}<extra></extra>'))
    fig.update_yaxes(autorange='reversed', tickfont=dict(size=11))
    fig.update_xaxes(side='top', tickangle=-35, tickfont=dict(size=11))
    st.plotly_chart(style_fig(fig, 40 + 26 * len(z) + 120), width='stretch', config={'displayModeBar': False})
    st.caption('Funds are ordered from highest to lowest chance of lagging (shown in brackets).')

    pick = st.selectbox('Zoom into one fund', order.tolist(), index=0)
    r = sc.set_index('Fund').loc[pick]
    p = push.loc[pick].astype(float).sort_values()
    c1, c2 = st.columns([1.3, 1])
    peers = sc[sc['Category'] == r['Category']]
    fig = go.Figure(go.Bar(
        x=p.values, y=[name(f) for f in p.index], orientation='h', marker_color=[RED if v > 0 else GREEN for v in p.values],
        text=[f"{E.fmt_feat(f, r[f])} vs {E.fmt_feat(f, peers[f].median())}" for f in p.index], textposition='auto',
        hovertext=[PLAIN[f][1] for f in p.index]))
    fig.add_vline(x=0, line_color=NAVY)
    fig.update_layout(title=f'{pick}: what is pushing it which way', xaxis=dict(range=[-2.3, 2.3], showticklabels=False,
                      title='◀ helps it stay ahead            pushes towards lagging ▶'))
    c1.plotly_chart(style_fig(fig, 430), width='stretch', config={'displayModeBar': False})
    with c2:
        bad = [f for f in p.index[::-1] if p[f] > 0.15][:3]
        good = [f for f in p.index if p[f] < -0.15][:3]
        tier = r['Risk Tier']
        card(f"<h4>{pick}</h4><div class='small'>{r['Category']} · chance of lagging next 12 months</div>"
             f"<div class='v' style='font-size:2rem;font-weight:800;color:{TIER_COLOR[tier]}'>{r['P']:.0%} · {tier}</div>")
        card("<h4>⚠️ Working against it</h4>" + (''.join(
            f"<div class='sign'><b>{name(f)}</b>: {E.fmt_feat(f, r[f])} vs {E.fmt_feat(f, peers[f].median())} for peers</div>"
            for f in bad) or "<div class='ok'>Nothing stands out</div>"))
        card("<h4>✅ Working for it</h4>" + (''.join(
            f"<div class='ok'><b>{name(f)}</b>: {E.fmt_feat(f, r[f])} vs {E.fmt_feat(f, peers[f].median())} for peers</div>"
            for f in good) or "<div class='sign'>Nothing stands out</div>"))
    st.caption('A guide to the main reasons, comparing the fund with its category; the model combines all 46 inputs, so this is '
               'not an exact breakdown of the score.')

    # ---------- 6. which method, and consistency
    st.markdown("### 🤖 Which method did we pick, and was it steady?")
    c1, c2 = st.columns(2)
    m = res['ROC-AUC'].astype(float).rename(index={'Baseline: past-1Y laggard': '"Last year\'s loser" rule'}).sort_values()
    fig = go.Figure(go.Bar(x=m.values, y=m.index, orientation='h', text=[f'{v:.2f}' for v in m.values], textposition='outside',
                           marker_color=[TEAL if i == E.BEST else GREY for i in m.index]))
    fig.add_vline(x=0.5, line_dash='dot', line_color=RED, annotation_text='coin toss', annotation_position='bottom right')
    fig.update_layout(title='Ranking skill of each method on 2025', xaxis_range=[0, 1])
    c1.plotly_chart(style_fig(fig, 320), width='stretch', config={'displayModeBar': False})
    q = tst.assign(Q=tst['Obs Date'].dt.to_period('Q').astype(str)).groupby('Q').apply(
        lambda g: pd.Series({'AUC': E.roc_auc_score(g[E.TARGET], g['P']) if g[E.TARGET].nunique() == 2 else np.nan,
                             'Lag': g[E.TARGET].mean()}))
    fig = go.Figure([go.Bar(x=q.index, y=q['AUC'], name='Ranking skill', marker_color=TEAL, text=[f'{v:.2f}' for v in q['AUC']],
                            textposition='outside'),
                     go.Scatter(x=q.index, y=q['Lag'], name='Share of funds that lagged', mode='lines+markers', line=dict(color=RED))])
    fig.add_hline(y=.5, line_dash='dot', line_color=GREY)
    fig.update_layout(title='Quarter by quarter in 2025', yaxis_range=[0, 1], yaxis_tickformat='.0%')
    c2.plotly_chart(style_fig(fig, 320), width='stretch', config={'displayModeBar': False})
    card(f"<h4>Why {E.BEST}?</h4>It builds hundreds of small <b>yes/no question trees</b> (\"Is the fund very large? Is new money "
         "pouring in?\"), each one fixing the mistakes of the last. It was chosen on the training years <b>before</b> the 2025 exam, "
         "so the exam stays fair. Its score changes a little with the random start (0.60–0.68 across 20 runs), always well above "
         "a coin toss.")

    with st.expander('📘 Jargon buster: what the technical words mean'):
        st.markdown("""
| Term | In plain words |
|---|---|
| **Benchmark / TRI** | The index a fund is judged against, including dividends (NIFTY 100, Midcap 150 or 500). |
| **Underperform / lag** | The fund earned less than its index over the next 12 months. |
| **Accuracy** | Out of all calls, how many were right. |
| **Precision** | Out of the funds it warned about, how many really lagged. |
| **Recall** | Out of the funds that really lagged, how many it warned about. |
| **ROC-AUC (ranking skill)** | Pick one fund that lagged and one that didn't: the chance the model gave the laggard the higher score. |
| **Backtest** | Replaying the model on past data it never saw, then checking what actually happened. |
| **Tracking error** | How differently a fund moves from its index. |
| **Feature importance** | How much the model relies on a signal when making its call. |
""")
    with st.expander('📊 Full results table (for experts)'):
        cols = ['ROC-AUC', 'Within-month AUC', 'Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1']
        st.dataframe(res[cols].astype(float).style.format('{:.3f}').highlight_max(axis=0, color='#D5F5E3'), width='stretch')
