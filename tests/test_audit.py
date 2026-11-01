import ast
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import logm,solve_discrete_lyapunov
from audit_tools import snapshot_audit,duplicate_issuer_basket
from scripts.data_quality import inspect_series

ROOT=Path(__file__).resolve().parents[1]

def functions(filename,names,namespace):
    source=ast.parse((ROOT/'scripts/stages'/filename).read_text())
    nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),namespace)
    return namespace

class AuditTests(unittest.TestCase):
    def test_completed_session_cutoff(self):
        from market_calendar import completed_sessions
        self.assertEqual(str(completed_sessions('2026-10-20T19:44Z').max().date()),'2026-10-19')
        self.assertEqual(str(completed_sessions('2026-10-20T21:00Z').max().date()),'2026-10-20')
        self.assertEqual(str(completed_sessions('2026-11-01T12:00Z').max().date()),'2026-10-30')

    def test_duplicate_classes(self):
        for basket in ['Z|ZG','COLM|UA|UAA','FWONA|FWONK','BATRA|BATRK|NXRT']:
            self.assertTrue(duplicate_issuer_basket(basket))
        self.assertFalse(duplicate_issuer_basket('Z|COLM'))

    def test_raw_missing_infinite_duplicate_and_nonpositive(self):
        f=pd.DataFrame({'P':[1,np.nan,np.inf,-1]},index=['2020-01-01','2020-01-01','2020-01-03','bad'])
        r=inspect_series(f,'TEST')
        self.assertEqual((r['Missing'],r['Infinite'],r['Nonpositive'],r['DuplicateDates'],r['InvalidDates']),(1,1,1,1,1))

    def test_sequential_rank_stops_at_first_nonrejection(self):
        from types import SimpleNamespace
        fake=lambda *a,**k:SimpleNamespace(lr1=np.array([10,1,10]),lr2=np.array([10,1,10]),cvt=np.ones((3,3))*5,cvm=np.ones((3,3))*5)
        ns=functions('audit.py',['johansen_rank'],{'np':np,'coint_johansen':fake})
        self.assertEqual(ns['johansen_rank'](np.zeros((20,3)),1,0),1)

    def test_joint_covariance_equilibrium_and_basis_invariance(self):
        ns=functions('mvou.py',['estimate_mvou'],{'np':np,'logm':logm,'solve_discrete_lyapunov':solve_discrete_lyapunov,'MAX_CONDITION_NUMBER':1e10,'IMAGINARY_TOL':1e-7,'STABILITY_TOL':1e-8})
        rng=np.random.default_rng(17);x=np.zeros((6000,2));A=np.array([[.92,.03],[.01,.78]])
        for t in range(1,len(x)):x[t]=A@x[t-1]+rng.normal(0,.01,2)
        r=ns['estimate_mvou'](x,np.eye(2));B=np.array([[2,.2],[.3,.8]]);s=ns['estimate_mvou'](x,B)
        self.assertTrue(np.allclose(r['A'],A,atol=.025))
        self.assertTrue(np.allclose(r['Sigma'],r['A']@r['Sigma']@r['A'].T+r['Q']))
        self.assertAlmostEqual(r['mahalanobis'],s['mahalanobis'],places=8)
        raw=r['B']@r['signal'];trans=s['B']@s['signal']
        self.assertTrue(np.allclose(raw,trans))
        self.assertTrue(np.allclose(r['deviation'],r['latest']-r['mu']))

    def test_signal_sign_and_gross_exposure(self):
        ns=functions('signals.py',['original_mvou_signal'],{'np':np})
        m={'B':np.array([[1.],[-1.]]),'Sigma':np.array([[2.]]),'deviation':np.array([1.])}
        raw,w=ns['original_mvou_signal'](m)
        self.assertTrue(np.allclose(w,[-.5,.5]))
        self.assertAlmostEqual(abs(w).sum(),1)

    def test_constant_series_detection(self):
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler().fit(np.column_stack([np.ones(10),np.arange(10)]))
        self.assertTrue(scaler.scale_[0]>0)
        self.assertFalse(scaler.var_[0]>1e-24)

    def test_missing_snapshot_does_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            report=snapshot_audit(d)
            self.assertTrue(any(c['Status']=='FAIL' for c in report['checks']))
            self.assertTrue(any(c['Status']=='UNVERIFIED' for c in report['checks']))

if __name__=='__main__':
    unittest.main()
