"""Split the baseline's localisation error by axis, for matched and missed cells.

`scripts/diagnose_missed_nodes.py` found that 86.7% of unmatched GT cells have a
detection nearby but outside the 7 um cap, median 9.1 um away. That is a placement
problem rather than a sensitivity one, but "9 um away" is ambiguous: it could be
the same cell localised badly, or a neighbouring cell that the detector did find.
Decomposing the offset by axis separates them, because a detector limitation shows
up on one axis while cell neighbours are roughly isotropic.

It also reports matched cells, which is the control. A systematic bias visible on
cells the detector *did* match is a property of the detector, not of the misses.

Diagnostic only. Nothing here is fit or tuned.

    python scripts/diagnose_localisation.py
"""

import os, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,'.')
from src.config import load_config
from src.data import read_geff, read_scale
from src.pipeline import predict_sample

SAMPLES=("44b6_144b256d","44b6_90724892","44b6_d5e7d891","44b6_db3c847b",
         "6bba_268e1230","6bba_3db54e20","6bba_55c70843","6bba_67ebd073","6bba_6ca87370")
cfg=load_config('conf/baseline.yaml'); d='data/raw/train'
M=[]; U=[]
t0=time.time()
for i,s in enumerate(SAMPLES,1):
    zp=os.path.join(d,s+'.zarr')
    g,_=predict_sample(zp,cfg); gt=read_geff(os.path.join(d,s+'.geff'))
    sc=np.asarray(read_scale(zp),dtype=np.float64)
    gz,gt_t=gt.nodes.zyx(),np.asarray(gt.nodes.t)
    pz,p_t=g.nodes.zyx(),np.asarray(g.nodes.t)
    for t in np.unique(gt_t):
        a=gz[gt_t==t]; b=pz[p_t==t]
        if a.shape[0]==0 or b.shape[0]==0: continue
        dm=np.linalg.norm((a*sc)[:,None,:]-(b*sc)[None,:,:],axis=2)
        cost=np.where(dm<=7.0,dm,1e6); r,c=linear_sum_assignment(cost)
        ok={int(x):int(y) for x,y in zip(r,c) if dm[x,y]<=7.0}
        nn=dm.argmin(axis=1)
        for k in range(a.shape[0]):
            if k in ok: M.append((b[ok[k]]-a[k])*sc)
            else: U.append((b[nn[k]]-a[k])*sc)
    print(f'  [{i}/9] {s} [{(time.time()-t0)/60:.1f} min]',flush=True)

M=np.array(M); U=np.array(U)
print(f'\nsigned offset pred minus gt, microns. matched n={len(M)}, unmatched n={len(U)}')
print(f"{'':>12}{'axis':>6}{'mean':>9}{'median':>9}{'p25':>8}{'p75':>8}{'mean|.|':>10}")
for lab,A in (('matched',M),('unmatched',U)):
    for j,ax in enumerate('ZYX'):
        v=A[:,j]; q=np.percentile(v,[25,50,75])
        print(f'{lab:>12}{ax:>6}{v.mean():>9.2f}{q[1]:>9.2f}{q[0]:>8.2f}{q[2]:>8.2f}{np.abs(v).mean():>10.2f}')
    r=np.linalg.norm(A,axis=1)
    print(f'{"":>12}{"|d|":>6}{r.mean():>9.2f}{np.median(r):>9.2f}')
    # how much of the squared distance each axis carries
    frac=(A**2).sum(axis=0)/ (A**2).sum()
    print(f'{"":>12}{"share of squared distance":>34}  Z {frac[0]:.2f}  Y {frac[1]:.2f}  X {frac[2]:.2f}')
