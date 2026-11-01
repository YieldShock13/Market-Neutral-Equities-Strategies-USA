"""Reconcile all runtime stages and retained models before publication."""
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from audit_tools import snapshot_audit

def main():
    root=Path('runtime');report=snapshot_audit(root/'pipeline');checks=[]
    def add(name,ok,detail):
        checks.append({'Check':name,'Status':'PASS' if ok else 'FAIL','Detail':str(detail)})
    for horizon in ['1Y','3Y','5Y','FULL']:
        p=pd.read_pickle(root/f'prices_{horizon}.pkl')
        add(f'{horizon} matrix',not p.empty and np.isfinite(p.to_numpy(float)).all() and p.gt(0).all().all() and p.index.is_unique and p.index.is_monotonic_increasing and p.columns.is_unique, f'{p.shape}; dates {p.index.min()} to {p.index.max()}')
    models=list((root/'pipeline/mvou').glob('model_*.npz'))
    for path in models:
        with np.load(path,allow_pickle=False) as model:
            horizon=path.name.split('_')[1];prices=pd.read_pickle(root/f'prices_{horizon}.pkl')
            B,A,mu,Q,Sigma=[model[k] for k in ['B','A','mu','Q','Sigma']]
            x=np.log(prices[list(model['tickers'])].to_numpy(float))@B
            add(f'{path.stem} latest coordinates',np.allclose(x[-1],model['latest']) and np.allclose(x[-1]-mu,model['deviation']),'Recomputed directly from retained adjusted prices and B')
            add(f'{path.stem} covariance',np.allclose(Sigma,A@Sigma@A.T+Q,atol=1e-9) and np.linalg.eigvalsh(Sigma).min()>0,'Lyapunov residual and positive definiteness')
            add(f'{path.stem} stability',np.max(abs(np.linalg.eigvals(A)))<1,'Discrete spectral radius')
            coord=-np.linalg.solve(Sigma,x[-1]-mu)
            add(f'{path.stem} signal',np.allclose(coord,model['signal']),'Signal sign and covariance solve')
    add('Retained fitted models',len(models)>0,f'{len(models)} models')
    report['runtime_checks']=checks
    report['generated_at']=datetime.now(timezone.utc).isoformat()
    report['input_hashes']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('prices_*.pkl')}
    report['stages']={name:len(pd.read_csv(root/path)) for name,path in {
        'universe':'universe_audit.csv','issuer':'issuer_audit.csv','discovery':'lasso/all_lasso_candidates.csv',
        'validation':'pipeline/validation_results.csv','specification':'pipeline/specification/specification_results.csv',
        'mvou':'pipeline/mvou/mvou_results.csv','signals':'pipeline/signals_v3/basket_signals.csv',
        'audit':'pipeline/final_audit/basket_audit.csv','analytics':'pipeline/dashboard_analytics/analytics_summary.csv'}.items()}
    (root/'backend_audit.json').write_text(json.dumps(report,indent=2,default=str))
    failures=[c for c in checks+report['checks'] if c['Status']=='FAIL']
    if failures:
        raise RuntimeError(f'{len(failures)} backend checks failed; publication blocked')

if __name__=='__main__':
    main()
