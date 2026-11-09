"""Data loading and live refresh.
NAV  : api.mfapi.in (free, no key), cached under CACHE_DIR.
TRI  : niftyindices.com; if it is unreachable or not yet published, the TRI is extended
       with the daily returns of a low-cost index fund tracking the same index (proxy).
TER / AUM : AMFI disclosures shipped in data/ (monthly / quarterly; refreshed with the repo).
"""
import json
import os
import tempfile

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(ROOT, 'data')
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'mf_ew_cache')
OFFLINE = os.environ.get('MF_OFFLINE', '0') == '1'      # for local testing without internet

BM = {'Large Cap': 'NIFTY 100 TRI', 'Mid Cap': 'NIFTY MIDCAP 150 TRI', 'Flexi Cap': 'NIFTY 500 TRI'}
# Direct-Growth index funds used to extend a TRI series when niftyindices has not published yet
TRI_PROXY = {'NIFTY 100 TRI': 149868,          # HDFC NIFTY 100 Index Fund
             'NIFTY MIDCAP 150 TRI': 147622,   # Motilal Oswal Nifty Midcap 150 Index Fund
             'NIFTY 500 TRI': 147625}          # Motilal Oswal Nifty 500 Index Fund
UA = {'User-Agent': 'Mozilla/5.0 (mf-early-warning student project)'}


def today():
    return pd.Timestamp.now(tz='Asia/Kolkata').tz_localize(None).normalize()


# ------------------------------------------------------------------ funds
def load_funds():
    f = pd.read_csv(os.path.join(DATA, 'fund_list.csv'))
    return f[['Fund', 'Category', 'AMC', 'Scheme Code', 'Benchmark']]


# ------------------------------------------------------------------ NAV
def nav_series(j):
    s = pd.Series([float(x['nav']) for x in j['data']],
                  index=pd.to_datetime([x['date'] for x in j['data']], format='%d-%m-%Y')).sort_index()
    s = s[s > 0]
    return s[~s.index.duplicated(keep='last')]


def fetch_mfapi(code, timeout=30):
    import requests
    r = requests.get(f'https://api.mfapi.in/mf/{int(code)}', headers=UA, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if not j.get('data'):
        raise ValueError(f'No NAV data for scheme code {code}')
    return j


def get_nav_json(code, online=True, force=False):
    """Latest NAV history for a scheme: live from MFAPI (cached for the day), else the copy in data/nav."""
    os.makedirs(os.path.join(CACHE_DIR, 'nav'), exist_ok=True)
    cpath = os.path.join(CACHE_DIR, 'nav', f'{int(code)}.json')
    seed = os.path.join(DATA, 'nav', f'{int(code)}.json')
    if online and not OFFLINE:
        # reuse today's download (India date) unless a fresh download is forced
        fetched = pd.Timestamp(os.path.getmtime(cpath), unit='s', tz='UTC').tz_convert('Asia/Kolkata').date() if os.path.exists(cpath) else None
        if not force and fetched == today().date():
            return json.load(open(cpath)), 'live (cached today)'
        try:
            j = fetch_mfapi(code)
            json.dump(j, open(cpath, 'w'))
            return j, 'live'
        except Exception:
            pass
    for p in (cpath, seed):
        if os.path.exists(p):
            return json.load(open(p)), 'stored copy'
    raise FileNotFoundError(f'Scheme {code}: could not download NAV and no stored copy exists')


def search_schemes(q):
    """Search MFAPI scheme names (returns list of {schemeCode, schemeName})."""
    import requests
    r = requests.get('https://api.mfapi.in/mf/search', params={'q': q}, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ TRI
def _niftyindices_update(old, end):
    import requests
    start = old['Date'].max() + pd.Timedelta(days=1)
    if start > end:
        return old, 0
    url = 'https://www.niftyindices.com/BackPage/getTotalReturnIndexString'
    hdr = {'Content-Type': 'application/json; charset=utf-8', **UA,
           'Origin': 'https://www.niftyindices.com', 'Referer': 'https://www.niftyindices.com/reports/historical-data'}
    names = {'NIFTY 100 TRI': 'NIFTY 100', 'NIFTY MIDCAP 150 TRI': 'NIFTY MIDCAP 150', 'NIFTY 500 TRI': 'NIFTY 500'}
    new = {}
    for col, name in names.items():
        a = start
        while a <= end:
            b = min(a + pd.DateOffset(months=6), end)
            cinfo = "{'name':'%s','startDate':'%s','endDate':'%s','indexName':'%s'}" % (
                name, f'{a:%d-%b-%Y}', f'{b:%d-%b-%Y}', name)
            r = requests.post(url, json={'cinfo': cinfo}, headers=hdr, timeout=30)
            r.raise_for_status()
            d = r.json()
            d = json.loads(d['d']) if isinstance(d, dict) and 'd' in d else d
            d = json.loads(d) if isinstance(d, str) else d
            for x in d:
                new.setdefault(x['Date'], {})[col] = float(x['TotalReturnsIndex'])
            a = b + pd.Timedelta(days=1)
    add = pd.DataFrame([{'Date': pd.to_datetime(k, format='%d %b %Y'), 'Raw Date': k, **v}
                        for k, v in new.items() if len(v) == 3])
    if not len(add):
        return old, 0
    return pd.concat([old, add]).drop_duplicates('Date', keep='first').sort_values('Date'), len(add)


def get_tri(online=True, force=False):
    """Returns (dict name -> Series, info dict)."""
    old = pd.read_csv(os.path.join(DATA, 'nifty_tri_history.csv'), parse_dates=['Date'])
    cpath = os.path.join(CACHE_DIR, 'tri.csv')
    if os.path.exists(cpath):
        c = pd.read_csv(cpath, parse_dates=['Date'])
        if c['Date'].max() > old['Date'].max():
            old = c
    info = {'official_to': old['Date'].max(), 'source': 'stored niftyindices file', 'proxy_days': 0}
    if online and not OFFLINE:
        try:
            old, n = _niftyindices_update(old, today())
            os.makedirs(CACHE_DIR, exist_ok=True)
            old.to_csv(cpath, index=False, date_format='%Y-%m-%d')
            info.update(official_to=old['Date'].max(), source='niftyindices.com (live' + (f', +{n} new days)' if n else ', checked today)'))
        except Exception as e:
            info['source'] = f'stored niftyindices file (live update failed: {type(e).__name__})'
    t = old.set_index('Date')
    tri = {c: t[c].astype(float) for c in BM.values()}
    # extend each series past the last official date with its index-fund proxy
    for name, code in TRI_PROXY.items():
        try:
            p = nav_series(get_nav_json(code, online, force)[0])
        except Exception:
            continue
        s = tri[name]
        last = s.index[-1]
        ext = p[p.index > last]
        base = p.loc[:last]
        if len(ext) and len(base):
            tri[name] = pd.concat([s, s.iloc[-1] * ext / base.iloc[-1]])
            info['proxy_days'] = max(info['proxy_days'], len(ext))
    info['tri_to'] = min(v.index[-1] for v in tri.values())
    return tri, info


# ------------------------------------------------------------------ AMFI
def load_ter():
    return pd.read_csv(os.path.join(DATA, 'amfi_ter_monthly.csv'), parse_dates=['Date'])[['Scheme Code', 'Date', 'TER_Direct']]


def load_aum():
    return pd.read_csv(os.path.join(DATA, 'amfi_aaum_quarterly.csv'), parse_dates=['Quarter End'])[['Scheme Code', 'Quarter End', 'AUM_Cr']]
