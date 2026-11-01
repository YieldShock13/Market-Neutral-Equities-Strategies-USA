"""Mandatory full-window eligibility shared by every model horizon."""
import numpy as np
import pandas as pd

TRADING_SESSIONS_PER_YEAR=252
REQUIRED_SESSIONS=10*TRADING_SESSIONS_PER_YEAR

def complete_history(series, calendar):
    calendar=pd.DatetimeIndex(calendar)
    if len(calendar)<REQUIRED_SESSIONS:
        raise ValueError('Fewer than 2520 completed exchange sessions in the reference calendar')
    window=calendar[-REQUIRED_SESSIONS:]
    series=series.copy()
    series.index=pd.to_datetime(series.index,errors='coerce')
    if series.index.tz is not None:
        series.index=series.index.tz_localize(None)
    series.index=series.index.normalize()
    invalid_dates=int(series.index.isna().sum())
    within=series.loc[(series.index>=window.min())&(series.index<=window.max())]
    duplicates=int(within.index.duplicated().sum())
    # Only construct a diagnostic alignment after recording duplicates.
    aligned=pd.to_numeric(within.loc[~within.index.duplicated(keep=False)],errors='coerce').reindex(window)
    values=aligned.to_numpy(float)
    counts={'RequiredSessions':REQUIRED_SESSIONS,'WindowStart':str(window.min().date()),
            'WindowEnd':str(window.max().date()),'Missing':int(aligned.isna().sum()),
            'Infinite':int(np.isinf(values).sum()),'Nonpositive':int((values<=0).sum()),
            'DuplicateDates':duplicates,'InvalidDates':invalid_dates}
    passed=all(counts[key]==0 for key in ['Missing','Infinite','Nonpositive','DuplicateDates','InvalidDates'])
    counts['Eligible10TradingYears']=passed
    return aligned,counts
