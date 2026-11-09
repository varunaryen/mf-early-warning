# Mutual Fund Underperformance Early-Warning (web app)

Predicts whether an Indian equity mutual fund (Direct, Growth) will **underperform its benchmark Total Return Index over the next 12 months**.
Built from the PGDM-BIFS (GIM) ML project *"Predicting Mutual Fund Benchmark Underperformance Using Machine Learning"*.

## What the app does
* **Fund check**: pick one of the 30 project funds, or any equity fund by name / scheme code, then click **Run**:
  risk gauge, tier, rank, warning signs, "why this score" factors, growth / excess-return / drawdown charts,
  the model's out-of-sample track record on that fund, and a downloadable fund report.
* **Compare funds**: up to 4 funds side by side with an overlaid growth chart.
* **Portfolio check**: enter holdings and amounts; get a weighted risk score, money by tier and lower-risk peer suggestions.
* **Watchlist**: all 30 funds ranked with filters, a risk-vs-return bubble chart and CSV download.
* **Model & backtest**: tier backtest (how often each tier actually underperformed), model comparison,
  confusion matrix with threshold slider, ROC curve, feature importance, results by quarter.
* **Always uses today's date**: on the first visit each day the app downloads the latest NAVs (mfapi.in) and index values
  (niftyindices.com, or an index-fund proxy for days not yet published), rebuilds the feature panel, retrains
  (train ≤ Dec-2024, test = 2025 snapshots whose 12-month outcome is known) and scores the latest date. Later visits that day reuse the result.

## Files
| Path | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `mfcore/features.py` | 39 point-in-time price features + 1Y target (same code as the project) |
| `mfcore/chars.py` | Expense ratio (TER) and AUM features from AMFI |
| `mfcore/data.py` | Downloads/caches NAV and TRI data |
| `mfcore/engine.py` | Training, testing, confusion matrix, scoring, warning signs |
| `data/` | Fund list, stored NAV/TRI history (used if a download fails), AMFI TER & AUM files |

## Run on your own computer
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud
1. Go to https://share.streamlit.io and **Continue with GitHub**.
2. **Create app → Deploy a public app from GitHub**: repository `varunaryen/mf-early-warning`, branch `main`, main file `app.py`.
3. Optionally pick a custom URL, then **Deploy**. Share the `*.streamlit.app` link.

The free app sleeps after a long period with no visitors; the next visitor wakes it with one click.

*Student project, not investment advice.*
