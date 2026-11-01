"""Observed common calendar, fixed January 2017 cutoff and <=1% missing."""
from pathlib import Path
import pandas as pd
from eligibility import screen_pack
ROOT=Path('runtime')
series={path.stem:pd.read_csv(path,index_col=0).iloc[:,0] for path in (ROOT/'prices').glob('*.csv')}
matrix,audit=screen_pack(series)
audit.to_csv(ROOT/'data_audit.csv',index=False)
if matrix.empty:raise ValueError('No securities passed January 2017 / 1% missing-data rule')
# Use the same observed dates for ALL securities. Never invent zero changes.
removed=matrix.index[matrix.isna().any(axis=1)]
pd.DataFrame({'Date':removed,'Reason':'At least one eligible ticker missing; jointly omitted'}).to_csv(ROOT/'omitted_sessions.csv',index=False)
matrix=matrix.dropna(how='any')
if len(matrix)<126:raise ValueError('Too few jointly complete observations after date omission')
matrix.index.name='Date'
matrix.to_pickle(ROOT/'adjusted_price_matrix.pkl')
summary=[]
for horizon,years in [('1Y',1),('3Y',3),('5Y',5),('FULL',None)]:
    sample=matrix.copy() if years is None else matrix.loc[matrix.index>=matrix.index.max()-pd.DateOffset(years=years)].copy()
    if len(sample)<100:raise ValueError(f'{horizon}: fewer than 100 jointly observed prices')
    sample.to_pickle(ROOT/f'prices_{horizon}.pkl')
    summary.append({'Sample':horizon,'Start':str(sample.index.min().date()),'End':str(sample.index.max().date()),'ObservedSteps':len(sample),'Stocks':sample.shape[1],'JointlyOmittedDates':len(removed),'Clock':'common-observation steps, not elapsed trading days'})
pd.DataFrame(summary).to_csv(ROOT/'sample_audit.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
