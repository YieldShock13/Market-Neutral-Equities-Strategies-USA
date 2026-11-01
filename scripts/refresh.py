"""Rebuild Meridian using the recovered notebook's estimation stages."""
import json
import re
import runpy
import sys
import shutil
import time
import traceback
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'
sys.path.insert(0,str(ROOT))
from audit_tools import ISSUER_CLASSES
from market_calendar import completed_sessions
from eligibility import POLICY

def issuer_key(info):
    # Issuer identity, not ticker identity. Class suffixes are removed only
    # after common-equity metadata checks; known aliases are a fallback.
    name = info.get('longName') or info.get('shortName') or ''
    name = re.sub(r'\bclass\s+[a-z0-9]+\b.*$', '', name, flags=re.I)
    return re.sub(r'[^a-z0-9]', '', name.lower())

def fetch_security(row):
    ticker, name = row
    symbol = ticker.replace('.', '-')
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        if not info:
            raise ValueError('Empty metadata')
        if not info.get('country') or not info.get('quoteType'):
            raise ValueError('Domicile or security metadata unavailable')
        if info.get('country') != 'United States' or info.get('quoteType') != 'EQUITY':
            return {'ticker':symbol,'status':'EXCLUDED','reason':'Non-US or non-equity metadata'}
        if not info.get('marketCap'):
            raise ValueError('Market capitalisation unavailable')
        if float(info['marketCap']) < 2e9:
            return {'ticker':symbol,'status':'EXCLUDED','reason':'Market capitalisation below $2bn'}
        history = stock.history(period='max', auto_adjust=False, actions=False, repair=True)
        if history.empty or 'Adj Close' not in history:
            raise ValueError('Missing adjusted-close history')
        history.to_csv(RUNTIME / 'vendor' / (symbol+'.csv'))
        (RUNTIME / 'metadata' / (symbol+'.json')).write_text(json.dumps(info, default=str))
        series = history['Adj Close'].copy()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        today = SESSION_END
        # Official exchange cutoff also protects manual intraday refreshes.
        series = series.loc[series.index <= today]
        key = issuer_key(info)
        for group in ISSUER_CLASSES:
            if symbol in group:
                key = 'classes:' + '|'.join(sorted(group))
        return dict(status='ELIGIBLE',reason='Automated screens passed; security classification not independently verified',ticker=symbol, issuer=key, volume=float(info.get('averageVolume') or 0),
                    series=series, name=name)
    except Exception as exc:
        print(f'{symbol}: unavailable ({type(exc).__name__})', flush=True)
        return {'ticker':symbol,'status':'REVIEW','reason':f'{type(exc).__name__}: {str(exc)[:250]}'}

def fetch_with_retry(row):
    result = None
    for attempt in range(3):
        result = fetch_security(row)
        if result['status'] != 'REVIEW':
            break
        time.sleep(2 ** attempt)
    return result

def main():
    import os
    os.chdir(ROOT)
    RUNTIME.mkdir(exist_ok=True)
    (RUNTIME / 'vendor').mkdir(exist_ok=True)
    (RUNTIME / 'metadata').mkdir(exist_ok=True)
    sessions=completed_sessions(years=12)
    global REQUIRED_CALENDAR
    REQUIRED_CALENDAR=sessions
    global SESSION_END
    SESSION_END=sessions.max()
    pd.Series(sessions,name='Date').to_csv(RUNTIME/'calendar.csv',index=False)
    directory = 'https://www.nasdaqtrader.com/dynamic/SymDir/'
    nasdaq = pd.read_csv(directory + 'nasdaqlisted.txt', sep='|')
    other = pd.read_csv(directory + 'otherlisted.txt', sep='|')
    nasdaq = nasdaq[nasdaq['Test Issue'].eq('N') & nasdaq['ETF'].eq('N')]
    other = other[other['Test Issue'].eq('N') & other['ETF'].eq('N') & other['Exchange'].eq('N')]
    listings = pd.concat([nasdaq[['Symbol','Security Name']],
        other[['ACT Symbol','Security Name']].rename(columns={'ACT Symbol':'Symbol'})])
    exclude = r'\bADR\b|\bADS\b|Depositary|Preferred|Preference|Warrant|\bRights?\b|\bUnits?\b|Acquisition Corp|\bNotes?\b|\bDebentures?\b|\bETF\b|\bETN\b|Closed.End Fund'
    listings = listings[~listings['Security Name'].str.contains(exclude, case=False, regex=True, na=True)]
    listings = listings[listings.Symbol.str.match(r'^[A-Z][A-Z.\-]{0,6}$', na=False)]
    rows = list(listings.drop_duplicates('Symbol').itertuples(index=False, name=None))
    print(f'Checking {len(rows)} listings', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        attempts = list(pool.map(fetch_with_retry, rows))
    pd.DataFrame([{k:v for k,v in r.items() if k != 'series'} for r in attempts]).to_csv(RUNTIME / 'universe_audit.csv',index=False)
    failed = sum(r['status']=='REVIEW' for r in attempts)
    if failed / max(1,len(attempts)) > .05:
        raise RuntimeError(f'{failed}/{len(attempts)} metadata/history requests unresolved; refusing incomplete universe')
    records = [r for r in attempts if r['status']=='ELIGIBLE']
    if len(records) < 500:
        raise RuntimeError(f'Only {len(records)} eligible securities; refusing partial publication')
    # Keep the most liquid eligible share class for each issuer deterministically.
    records.sort(key=lambda r: (-r['volume'], r['ticker']))
    selected, seen, audit = [], set(), []
    for record in records:
        key = record['issuer'] or record['ticker']
        keep = key not in seen
        audit.append({k:record[k] for k in ['ticker','issuer','name','volume']} | {'selected':keep})
        if keep:
            selected.append(record)
            seen.add(key)
    prices = RUNTIME / 'prices'
    shutil.rmtree(prices, ignore_errors=True)
    prices.mkdir()
    for record in selected:
        record['series'].to_csv(prices / (record['ticker']+'.csv'))
    pd.DataFrame(audit).to_csv(RUNTIME / 'issuer_audit.csv', index=False)
    runpy.run_path(str(ROOT / 'scripts/data_quality.py'), run_name='__main__')
    for stage in ['samples','discovery','validation','specification','mvou','signals','audit','analytics']:
        print(f'Running {stage}', flush=True)
        runpy.run_path(str(ROOT / 'scripts/stages' / (stage+'.py')), run_name='__main__')
    signals = pd.read_csv(RUNTIME / 'pipeline/signals_v3/basket_signals.csv')
    if signals.empty or not signals.Status.eq('SIGNAL_COMPUTED').any():
        raise RuntimeError('No computed signals; keeping last published results')
    shutil.copy2(RUNTIME/'data_audit.csv',RUNTIME/'pipeline/data_audit.csv')
    shutil.copy2(RUNTIME/'omitted_sessions.csv',RUNTIME/'pipeline/omitted_sessions.csv')
    runpy.run_path(str(ROOT / 'scripts/backend_audit.py'), run_name='__main__')
    destination = ROOT / 'pipeline'
    for folder, files in {
        'signals_v3':['basket_signals.csv'],
        'final_audit':['basket_audit.csv','dollar_positions.csv'],
        'dashboard_analytics':['analytics_summary.csv','eigenmodes.csv','rolling_diagnostics.csv','spread_history.csv','transition_matrix.csv']
    }.items():
        for filename in files:
            target = destination / folder / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNTIME / 'pipeline' / folder / filename, target)
    for filename in ['issuer_audit.csv','universe_audit.csv','data_audit.csv','backend_audit.json','omitted_sessions.csv','sample_audit.csv']:
        shutil.copy2(RUNTIME / filename, destination / filename)
    latest = pd.read_pickle(RUNTIME / 'prices_FULL.pkl').index.max()
    (destination / 'refresh_status.json').write_text(json.dumps({
        'completed_at':datetime.now(timezone.utc).isoformat(), 'price_date':str(latest.date()),
        'eligible_issuers':len(selected), 'status':'success',
        'source_commit':os.environ.get('GITHUB_SHA','local'),
        'unresolved_requests':failed,
        'eligibility_policy':POLICY,
        'max_missing_fraction':.01,
        'window_start':'2017-01-01',
        'window_end':str(REQUIRED_CALENDAR.max().date()),
        'universe_scope':'Current membership and market cap; not historical point-in-time membership'
    }, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception:
        RUNTIME.mkdir(exist_ok=True)
        (RUNTIME/'failure.txt').write_text(traceback.format_exc())
        raise
