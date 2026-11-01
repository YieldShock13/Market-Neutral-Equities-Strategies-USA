"""Global rolling ten-trading-year gate, applied before all model horizons."""
from pathlib import Path
import numpy as np
import pandas as pd
from eligibility import complete_history,REQUIRED_SESSIONS,TRADING_SESSIONS_PER_YEAR

ROOT=Path('runtime')
calendar_file=ROOT/'calendar.csv'
if not calendar_file.exists():
    raise ValueError('Missing official completed-session calendar; refusing to infer one from available prices')
calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_file)['Date']))
if len(calendar)!=REQUIRED_SESSIONS or not calendar.is_unique or not calendar.is_monotonic_increasing:
    raise ValueError('Reference calendar must contain exactly 2520 ordered unique completed sessions')
prices={};audit=[]
for path in sorted((ROOT/'prices').glob('*.csv')):
    try:
        frame=pd.read_csv(path,index_col=0)
        aligned,record=complete_history(frame.iloc[:,0],calendar)
        record['Ticker']=path.stem
        if record['Eligible10TradingYears']:
            prices[path.stem]=aligned
    except Exception as exc:
        record={'Ticker':path.stem,'Eligible10TradingYears':False,'Error':str(exc)}
    audit.append(record)
pd.DataFrame(audit).to_csv(ROOT/'data_audit.csv',index=False)
if not prices:
    raise ValueError('No ticker has complete positive finite data for the entire rolling 2520-session window')
matrix=pd.DataFrame(prices,index=calendar)
matrix.index.name='Date'
assert matrix.notna().all().all() and np.isfinite(matrix.to_numpy(float)).all() and matrix.gt(0).all().all()
matrix.to_pickle(ROOT/'adjusted_price_matrix.pkl')
summary=[]
for horizon,years in [('1Y',1),('3Y',3),('5Y',5),('FULL',10)]:
    sample=matrix.iloc[-years*TRADING_SESSIONS_PER_YEAR:].copy()
    # Every horizon shares the exact same globally eligible ticker set.
    assert sample.columns.equals(matrix.columns)
    sample.to_pickle(ROOT/f'prices_{horizon}.pkl')
    summary.append({'Sample':horizon,'Start':str(sample.index.min().date()),'End':str(sample.index.max().date()),'TradingDays':len(sample),'CompleteStocks':sample.shape[1]})
pd.DataFrame(summary).to_csv(ROOT/'sample_audit.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
