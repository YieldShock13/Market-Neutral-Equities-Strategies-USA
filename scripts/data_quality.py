"""Inspect raw saved price series before any complete-case filtering."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def inspect_series(frame, ticker, expected=None):
    series=pd.to_numeric(frame.iloc[:,0],errors='coerce')
    dates=pd.to_datetime(frame.index,errors='coerce')
    values=series.to_numpy(float)
    return {'Ticker':ticker,'Rows':len(frame),'Missing':int(series.isna().sum()),
            'Infinite':int(np.isinf(values).sum()),'Nonpositive':int((values<=0).sum()),
            'DuplicateDates':int(dates.duplicated().sum()),'InvalidDates':int(dates.isna().sum()),
            'UnsortedDates':not dates.is_monotonic_increasing,
            'FirstDate':str(dates.min()),'LastDate':str(dates.max()),
            'MissingExpectedSessions':None if expected is None else len(expected.difference(dates)),
            'ZeroReturnFraction':float(series.pct_change(fill_method=None).eq(0).mean())}

def main():
    root=Path('runtime'); records=[]
    for path in sorted((root/'prices').glob('*.csv')):
        record=inspect_series(pd.read_csv(path,index_col=0),path.stem)
        record['SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
    audit=pd.DataFrame(records)
    audit.to_csv(root/'data_audit.csv',index=False)
    bad=['Infinite','Nonpositive','DuplicateDates','InvalidDates','UnsortedDates']
    if audit.empty:
        raise ValueError('No raw prices available')
    # The shared 2520-session gate excludes invalid tickers individually before all tests.

if __name__=='__main__':
    main()
