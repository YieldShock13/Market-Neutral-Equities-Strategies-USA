"""Run original notebook stages; publish only fresh, complete output files."""
import json
import os
import runpy
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from original_refresh.session import latest_completed_session

OUTPUTS={
    'signals_v3':['basket_signals.csv'],
    'final_audit':['basket_audit.csv','dollar_positions.csv'],
    'dashboard_analytics':['analytics_summary.csv','eigenmodes.csv','rolling_diagnostics.csv','spread_history.csv','transition_matrix.csv']
}

def main():
    os.chdir(ROOT)
    source=ROOT/'data/original_universe.csv'
    if not source.exists():
        raise FileNotFoundError('Original eligible_universe_v2.csv is required at data/original_universe.csv. No replacement universe is inferred.')
    universe=pd.read_csv(source)
    if 'YahooTicker' not in universe or universe.YahooTicker.dropna().empty:
        raise ValueError('Original universe must contain nonempty YahooTicker column')
    target=latest_completed_session()
    print(f'Target completed session: {target.date()}',flush=True)
    for stage in ['download','samples','discovery','validation','specification','mvou','signals','audit','analytics']:
        print(f'Original stage: {stage}',flush=True)
        runpy.run_path(str(ROOT/'original_refresh/stages'/f'{stage}.py'),run_name='__main__')
    runtime=ROOT/'runtime_original'
    price_date=pd.read_pickle(runtime/'prices_FULL.pkl').index.max()
    if pd.Timestamp(price_date).normalize()!=target.normalize():
        raise ValueError(f'Results are dated {price_date}, expected {target}; refusing stale publication')
    prepared=[]
    for folder,names in OUTPUTS.items():
        for name in names:
            path=runtime/'pipeline'/folder/name
            frame=pd.read_csv(path)
            if frame.empty:
                raise ValueError(f'Empty output {path}; retaining previous published snapshot')
            prepared.append((path,ROOT/'pipeline'/folder/name))
    signals=pd.read_csv(runtime/'pipeline/signals_v3/basket_signals.csv')
    if not signals.Status.eq('SIGNAL_COMPUTED').any():
        raise ValueError('No original-model signals computed; publication stopped')
    for path,destination in prepared:
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    status={'price_date':str(target.date()),'completed_at':datetime.now(timezone.utc).isoformat(),
            'source_commit':os.environ.get('GITHUB_SHA','local'),'calculation_source':'original uploaded notebook',
            'universe_source':'original eligible_universe_v2.csv','computed_baskets':int(signals.Status.eq('SIGNAL_COMPUTED').sum())}
    (ROOT/'pipeline/refresh_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)

if __name__=='__main__':
    main()
