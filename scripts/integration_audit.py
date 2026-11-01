"""Offline integration evidence using explicitly synthetic prices, never production data."""
import contextlib
import io
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    rng=np.random.default_rng(1024)
    dates=pd.bdate_range(end=pd.Timestamp.now().normalize(),periods=2900)
    trend=4+np.cumsum(rng.normal(0,.02,len(dates)))
    prices={}
    for ticker in ['SYN_A','SYN_B','SYN_C']:
        error=np.zeros(len(dates))
        for t in range(1,len(error)):error[t]=.8*error[t-1]+rng.normal(0,.002)
        prices[ticker]=np.exp(trend+error)
    frame=pd.DataFrame(prices,index=dates)
    previous=Path.cwd();results=[]
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        root=Path('runtime');(root/'prices').mkdir(parents=True)
        pd.Series(dates[-1260:],name='Date').to_csv(root/'calendar.csv',index=False)
        for ticker in frame:
            frame[ticker].to_csv(root/'prices'/f'{ticker}.csv')
        damaged=frame.iloc[:,0].copy();damaged.iloc[-1200]=np.nan
        damaged.to_csv(root/'prices/SYN_GAP.csv')
        try:
            for stage in ['samples','discovery','validation','specification','mvou','signals','audit','analytics']:
                capture=io.StringIO()
                with patch('yfinance.download',return_value=pd.DataFrame({'Adj Close':np.exp(trend)},index=dates)),contextlib.redirect_stdout(capture):
                    runpy.run_path(str(ROOT/'scripts/stages'/f'{stage}.py'),run_name='__main__')
                if stage=='samples':
                    for horizon in ['1Y','3Y','5Y','FULL']:
                        sample=pd.read_pickle(root/f'prices_{horizon}.pkl')
                        assert set(sample.columns)=={'SYN_A','SYN_B','SYN_C'},horizon
                results.append({'stage':stage,'status':'PASS','log_tail':capture.getvalue()[-250:]})
            from audit_tools import snapshot_audit
            report=snapshot_audit(root/'pipeline')
            failures=[c for c in report['checks'] if c['Status']=='FAIL']
            if failures:raise AssertionError(failures)
            results.append({'stage':'cross-file reconciliation','status':'PASS'})
        finally:
            os.chdir(previous)
    output={'scope':'Synthetic offline integration; does not verify vendor or real raw data','steps':results}
    (ROOT/'pipeline/integration_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()
