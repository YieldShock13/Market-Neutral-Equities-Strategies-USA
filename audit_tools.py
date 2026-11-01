"""Read-only checks of the exact published snapshot. Never fills missing values."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

KEY = ['Horizon', 'BasketID']
ISSUER_CLASSES = [('Z','ZG'),('UA','UAA'),('FWONA','FWONK'),('BATRA','BATRK'),('GOOG','GOOGL'),('BRK-A','BRK-B'),('FOX','FOXA'),('NWS','NWSA')]

def duplicate_issuer_basket(value):
    tickers=set(str(value).split('|'))
    return any(len(tickers.intersection(group))>1 for group in ISSUER_CLASSES)
FILES = {
 'signals': 'signals_v3/basket_signals.csv',
 'audit': 'final_audit/basket_audit.csv',
 'positions': 'final_audit/dollar_positions.csv',
 'history': 'dashboard_analytics/spread_history.csv',
 'rolling': 'dashboard_analytics/rolling_diagnostics.csv',
 'matrix': 'dashboard_analytics/transition_matrix.csv',
 'modes': 'dashboard_analytics/eigenmodes.csv',
 'summary': 'dashboard_analytics/analytics_summary.csv',
}

def snapshot_audit(pipe):
    pipe = Path(pipe)
    checks, missing, frames, fingerprints = [], [], {}, {}
    def add(name, passed, detail, fail='FAIL'):
        checks.append({'Check':name, 'Status':'PASS' if passed else fail, 'Detail':str(detail)})
    for name, relative in FILES.items():
        path = pipe / relative
        if not path.exists():
            add(relative, False, 'File absent; cannot validate')
            continue
        fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            frame = pd.read_csv(path, dtype={'BasketID':str})
        except Exception as exc:
            add(relative, False, f'Cannot parse: {type(exc).__name__}')
            continue
        frames[name] = frame
        add(relative, not frame.empty, f'{len(frame):,} rows')
        for column in frame:
            nulls = int(frame[column].isna().sum())
            numeric = pd.to_numeric(frame[column], errors='coerce')
            infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
            if nulls or infinite:
                missing.append({'File':relative,'Column':column,'Missing':nulls,'Infinite':infinite,'Rows':len(frame)})
    signals, positions = frames.get('signals'), frames.get('positions')
    if signals is not None:
        add('Unique signal keys', not signals.duplicated(KEY).any(), 'One record per horizon / basket')
        active = signals.loc[signals.Status.eq('SIGNAL_COMPUTED')].copy()
        duplicated_issuers=active.Basket.map(duplicate_issuer_basket)
        add('Known duplicate issuer classes',not duplicated_issuers.any(),f'{int(duplicated_issuers.sum())} baskets contain multiple known share classes','FLAG')
        required = ['Basket','JointMispricing','PortfolioBeta','MarketExplainedVariancePct','NetDollarExposurePct']
        add('Required signal values', not active[required].isna().any().any() and np.isfinite(active[required[1:]].to_numpy(dtype=float)).all(), 'Computed signals must have finite mandatory fields')
        add('Non-negative joint distance', active.JointMispricing.ge(0).all(), 'Mahalanobis distance cannot be negative')
        if positions is not None:
            add('Unique position keys', not positions.duplicated(KEY+['Ticker']).any(), 'No repeated ticker inside a basket')
            bad_membership = bad_gross = bad_net = bad_shares = bad_side = 0
            for _, row in active.iterrows():
                group = positions[positions.Horizon.eq(row.Horizon) & positions.BasketID.eq(row.BasketID)]
                bad_membership += set(group.Ticker) != set(str(row.Basket).split('|'))
                w = group.DollarWeight.to_numpy(float)
                p = group.LatestAdjustedPrice.to_numpy(float)
                q = group.SharesPerDollarGross.to_numpy(float)
                bad_gross += not (len(w) and np.isfinite(w).all() and np.isclose(abs(w).sum(),1,atol=1e-8))
                bad_net += not np.isclose(w.sum()*100,row.NetDollarExposurePct,atol=1e-6)
                bad_shares += not (len(p) and np.isfinite(p).all() and (p>0).all() and np.allclose(q*p,w,atol=1e-10))
                expected = np.where(w>0,'LONG',np.where(w<0,'SHORT','FLAT'))
                bad_side += not np.array_equal(group.Side.to_numpy(),expected)
            for name,count in [('Position membership',bad_membership),('Gross weights sum to 1',bad_gross),('Net exposure reconciliation',bad_net),('Adjusted-price quantity arithmetic',bad_shares),('Position side signs',bad_side)]:
                add(name,count==0,f'{count} affected baskets; adjusted-price quantities are research units, not live orders')
        audit = frames.get('audit')
        if audit is not None:
            add('Unique audit keys',not audit.duplicated(KEY).any(),'Audit join must be one-to-one')
            merged = active.merge(audit,on=KEY,how='left',suffixes=('','_audit'),indicator=True)
            add('Audit coverage', merged['_merge'].eq('both').all(), 'Every computed basket must have a matching audit')
            add('Audit completed', merged.Status_audit.eq('AUDITED').all(), 'Failed audits must not masquerade as stability passes')
        summary = frames.get('summary')
        if summary is not None:
            joined=active.merge(summary,on=KEY,how='left',suffixes=('','_summary'))
            add('Analytics coverage',joined.Status_summary.eq('COMPUTED').all(),'Every active basket must have computed analytics')
            add('Signal / analytics distance',np.allclose(joined.JointMispricing,joined.LatestJointMispricing,atol=1e-7,equal_nan=False),'Same fitted snapshot across files')
        neutral = active.PortfolioBeta.abs().le(.01) & active.NetDollarExposurePct.abs().le(.01)
        add('Exact neutrality', bool(neutral.all()),f'{int((~neutral).sum())} baskets exceed display tolerances: |beta| 0.01 or |net dollars| 0.01%; these are diagnostic tolerances, not trading limits', 'FLAG')
    history = frames.get('history')
    if history is not None:
        add('Unique spread dates',not history.duplicated(KEY+['Component','Date']).any(),'One observation per date / component')
        add('Spread standard deviation',history.SpreadSD.gt(0).all() and np.isfinite(history.SpreadSD).all(),'Positive finite fitted dispersion')
        z=(history.Spread-history.Equilibrium)/history.SpreadSD
        add('Z-score identity',np.allclose(z,history.ZScore,atol=1e-7,equal_nan=False),'Recomputed from stored spreads, equilibrium and dispersion')
    status_path=pipe/'refresh_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else None
    add('Refresh provenance',status is not None,'No successful automated refresh manifest' if status is None else json.dumps(status),'UNVERIFIED')
    raw=pipe/'data_audit.csv'
    add('Raw-price audit evidence',raw.exists(),'Raw dates, gaps, excluded columns and price validity must be checked before filtering','UNVERIFIED')
    add('Trading validation',False,'Broker quotes, borrow availability, costs, liquidity and execution have not been validated','UNVERIFIED')
    return {'checks':checks,'missing':missing,'fingerprints':fingerprints,'refresh':status,
            'scope':'Published CSV integrity only. No claim of raw-vendor correctness or trading approval.'}
