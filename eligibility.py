"""Fixed-start eligibility relative to observed prices across the universe."""
import numpy as np
import pandas as pd

START_DATE=pd.Timestamp('2017-01-01')
MAX_MISSING_FRACTION=.01
POLICY='jan2017_peer_calendar_missing_1pct_v1'

def screen_pack(series_by_ticker):
    cleaned={}; structural={}
    for ticker,series in series_by_ticker.items():
        s=series.copy();s.index=pd.to_datetime(s.index,errors='coerce')
        if s.index.tz is not None:s.index=s.index.tz_localize(None)
        s.index=s.index.normalize()
        structural[ticker]=bool(s.index.isna().any() or s.index.duplicated().any())
        s=s.loc[s.index.notna() & ~s.index.duplicated(keep=False)]
        cleaned[ticker]=pd.to_numeric(s,errors='coerce').sort_index()
    raw=pd.DataFrame(cleaned).sort_index()
    raw=raw.loc[raw.index>=START_DATE]
    # A common absent date is not inserted into the calendar or filled.
    valid=raw.where(np.isfinite(raw)&raw.gt(0))
    common=valid.index[valid.notna().any(axis=1)]
    if len(common)==0:raise ValueError('No observed peer calendar since January 2017')
    matrix=valid.reindex(common)
    cutoff=common.min()
    audit=[]
    for ticker in matrix:
        s=cleaned[ticker];first=s.loc[np.isfinite(s)&s.gt(0)].index.min()
        missing=int(matrix[ticker].isna().sum()); fraction=missing/len(common)
        listed=bool(pd.notna(first) and first<=cutoff and pd.notna(matrix[ticker].iloc[0]))
        eligible=listed and not structural[ticker] and fraction<=MAX_MISSING_FRACTION
        audit.append({'Ticker':ticker,'FirstObservedPrice':str(first),'WindowStart':str(cutoff.date()),'WindowEnd':str(common.max().date()),'ExpectedPeerSessions':len(common),'Missing':missing,'MissingFraction':fraction,'ListedByCutoff':listed,'DateStructureValid':not structural[ticker],'Eligible':eligible})
    proof=pd.DataFrame(audit)
    return matrix.loc[:,proof.loc[proof.Eligible,'Ticker']],proof
