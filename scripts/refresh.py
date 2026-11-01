"""Rebuild Meridian using the recovered notebook's estimation stages."""
import json
import re
import runpy
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'runtime'

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
        if info.get('country') != 'United States' or info.get('quoteType') != 'EQUITY':
            return None
        if float(info.get('marketCap') or 0) < 2e9:
            return None
        history = stock.history(period='max', auto_adjust=False, actions=False, repair=True)
        if history.empty or 'Adj Close' not in history:
            return None
        series = history['Adj Close'].dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        today = pd.Timestamp.now(tz='America/New_York').normalize().tz_localize(None)
        # Run after US close; discard future dates, retain completed session.
        series = series.loc[series.index <= today]
        if series.empty or series.index.min() > today - pd.DateOffset(years=10):
            return None
        key = issuer_key(info)
        if symbol in {'Z', 'ZG'}:
            key = 'zillowgroup'
        return dict(ticker=symbol, issuer=key, volume=float(info.get('averageVolume') or 0),
                    series=series, name=name)
    except Exception as exc:
        print(f'{symbol}: unavailable ({type(exc).__name__})', flush=True)
        return None

def main():
    import os
    os.chdir(ROOT)
    RUNTIME.mkdir(exist_ok=True)
    directory = 'https://www.nasdaqtrader.com/dynamic/SymDir/'
    nasdaq = pd.read_csv(directory + 'nasdaqlisted.txt', sep='|')
    other = pd.read_csv(directory + 'otherlisted.txt', sep='|')
    nasdaq = nasdaq[nasdaq['Test Issue'].eq('N') & nasdaq['ETF'].eq('N')]
    other = other[other['Test Issue'].eq('N') & other['ETF'].eq('N') & other['Exchange'].eq('N')]
    listings = pd.concat([nasdaq[['Symbol','Security Name']],
        other[['ACT Symbol','Security Name']].rename(columns={'ACT Symbol':'Symbol'})])
    exclude = r'ADR|ADS|Depositary|Preferred|Preference|Warrant|\bRights?\b|\bUnits?\b|Acquisition Corp|\bNotes?\b|\bDebentures?\b|\bETF\b|\bETN\b|Closed.End Fund'
    listings = listings[~listings['Security Name'].str.contains(exclude, case=False, regex=True, na=True)]
    listings = listings[listings.Symbol.str.match(r'^[A-Z][A-Z.\-]{0,6}$', na=False)]
    rows = list(listings.drop_duplicates('Symbol').itertuples(index=False, name=None))
    print(f'Checking {len(rows)} listings', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [r for r in pool.map(fetch_security, rows) if r]
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
    for stage in ['samples','discovery','validation','specification','mvou','signals','audit','analytics']:
        print(f'Running {stage}', flush=True)
        runpy.run_path(str(ROOT / 'scripts/stages' / (stage+'.py')), run_name='__main__')
    signals = pd.read_csv(RUNTIME / 'pipeline/signals_v3/basket_signals.csv')
    if signals.empty or not signals.Status.eq('SIGNAL_COMPUTED').any():
        raise RuntimeError('No computed signals; keeping last published results')
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
    shutil.copy2(RUNTIME / 'issuer_audit.csv', destination / 'issuer_audit.csv')
    latest = pd.read_pickle(RUNTIME / 'prices_FULL.pkl').index.max()
    (destination / 'refresh_status.json').write_text(json.dumps({
        'completed_at':datetime.now(timezone.utc).isoformat(), 'price_date':str(latest.date()),
        'eligible_issuers':len(selected), 'status':'success'
    }, indent=2))

if __name__ == '__main__':
    main()
