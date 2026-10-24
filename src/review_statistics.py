"""Paired fixed-model uncertainty; no training or threshold selection.

External resampling unit is the patient (all images retained with multiplicity).
Source resampling is stratified by image label because patient IDs are unavailable.
Intervals are pointwise percentile 95% CIs; no multiplicity-adjusted claims.
"""
from pathlib import Path
import csv, json
import numpy as np

NAMES = ['plain','augmix','aug1','ce3']
METRICS = ['accuracy','auroc','balanced_accuracy','sensitivity','specificity','brier']
CONTRASTS = [('augmix','plain'),('aug1','plain'),('ce3','plain'),('augmix','ce3'),('ce3','aug1')]

class WeightedMetrics:
    def __init__(self, y, score):
        self.y=np.asarray(y,dtype=np.int64); self.s=np.asarray(score,dtype=np.float64)
        assert set(np.unique(self.y))=={0,1}
        assert np.isfinite(self.s).all() and ((self.s>=0)&(self.s<=1)).all()
        self.pred=self.s>=.5
        self.order=np.argsort(self.s,kind='stable')
        ss=self.s[self.order]
        self.starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    def __call__(self,w):
        y=self.y; w=np.asarray(w,dtype=np.float64)
        pos=w*y; neg=w*(1-y); p=pos.sum(); n=neg.sum()
        if p==0 or n==0: return np.full(6,np.nan)
        tp=(pos*self.pred).sum(); tn=(neg*~self.pred).sum()
        gp=np.add.reduceat(pos[self.order],self.starts)
        gn=np.add.reduceat(neg[self.order],self.starts)
        auc=np.sum(gp*(np.cumsum(gn)-.5*gn))/(p*n)
        return np.array([(tp+tn)/(p+n),auc,.5*(tp/p+tn/n),tp/p,tn/n,
                         np.sum(w*(self.s-y)**2)/(p+n)])

def ci_rows(draws, points, names, metrics, contrasts, scope, n, clusters):
    rows=[]
    def add(label, values, point):
        low,high=np.nanquantile(values,[.025,.975],axis=0)
        for j,m in enumerate(metrics):
            rows.append(dict(scope=scope,comparison=label,metric=m,estimate=float(point[j]),
                ci_low=float(low[j]),ci_high=float(high[j]),n_images=n,n_clusters=clusters,
                bootstrap_draws=len(values),valid_draws=int(np.isfinite(values[:,j]).sum())))
    for i,name in enumerate(names): add(name,draws[:,i],points[i])
    for a,b in contrasts:
        ia,ib=names.index(a),names.index(b)
        add(a+' minus '+b,draws[:,ia]-draws[:,ib],points[ia]-points[ib])
    return rows

def prediction_bootstrap(y,scores,patient_ids=None,draws=5000,seed=632919,scope='external_primary',contrasts=None):
    y=np.asarray(y); names=list(scores); funcs=[WeightedMetrics(y,scores[k]) for k in names]
    point=np.array([f(np.ones(len(y))) for f in funcs]); rng=np.random.default_rng(seed)
    if patient_ids is None:
        groups=[np.flatnonzero(y==c) for c in (0,1)]; k=len(y); inv=None
    else:
        unique,inv=np.unique(np.asarray(patient_ids),return_inverse=True); k=len(unique)
    sampled=np.empty((draws,len(names),6))
    for b in range(draws):
        if inv is not None:
            w=np.bincount(rng.integers(0,k,size=k),minlength=k)[inv]
        else:
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            w=np.bincount(ix,minlength=len(y))
        sampled[b]=[f(w) for f in funcs]
        if (b+1)%1000==0: print(scope,'bootstrap',b+1,flush=True)
    contrasts=[(a,b) for a,b in (CONTRASTS if contrasts is None else contrasts) if a in names and b in names]
    rows=ci_rows(sampled,point,names,METRICS,contrasts,scope,len(y),k)
    return rows,sampled

def mean_cluster_bootstrap(values,patient_ids,names,metrics,contrasts,draws=5000,seed=632920,scope='localization'):
    values=np.asarray(values,dtype=float) # image x model x metric
    _,inv=np.unique(np.asarray(patient_ids),return_inverse=True); k=inv.max()+1
    rng=np.random.default_rng(seed); sampled=np.empty((draws,len(names),len(metrics)))
    for b in range(draws):
        w=np.bincount(rng.integers(0,k,size=k),minlength=k)[inv]
        sampled[b]=np.einsum('i,imj->mj',w,values)/w.sum()
    return ci_rows(sampled,values.mean(0),names,metrics,contrasts,scope,len(values),int(k)),sampled

def save_rows(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(out,draws=5000):
    out=Path(out); data=np.load(out/'four_way_predictions.npz'); rows=[]; distributions={}
    names=[n for n in NAMES if n+'_scores' in data]
    for scope,sel in [('external_primary',data['primary_eligible']),('external_expanded',np.ones(len(data['labels']),bool))]:
        r,d=prediction_bootstrap(data['labels'][sel],{n:data[n+'_scores'][sel] for n in names},
            data['patient_ids'][sel],draws=draws,scope=scope)
        rows+=r;distributions[scope]=d
    with (out/'E_final_B64_localization_cases.csv').open(encoding='utf-8-sig') as f: loc=list(csv.DictReader(f))
    lm=['area_corrected_energy','pointing_hit']; ln=names+['center_prior']
    val=np.array([[[float(r[n+'_'+m]) for m in lm] for n in ln] for r in loc])
    contrasts=CONTRASTS+[('plain','center_prior'),('augmix','center_prior')]
    r,d=mean_cluster_bootstrap(val,[r['patient_id'] for r in loc],ln,lm,contrasts,draws,scope='localization')
    rows+=r;distributions['localization']=d
    with (out/'E_final_B64_occlusion_cases.csv').open(encoding='utf-8-sig') as f: occ=list(csv.DictReader(f))
    om=['mean_inside_minus_outside','blur_inside_minus_outside','both_positive_fraction']
    val=np.array([[[float(r[n+'_mean_inside_minus_outside']),float(r[n+'_blur_inside_minus_outside']),
                   float(float(r[n+'_mean_inside_minus_outside'])>0 and float(r[n+'_blur_inside_minus_outside'])>0)]
                  for n in names] for r in occ])
    r,d=mean_cluster_bootstrap(val,[r['patient_id'] for r in occ],names,om,CONTRASTS,draws,seed=632921,scope='occlusion')
    rows+=r;distributions['occlusion']=d
    save_rows(out/'paired_confidence_intervals.csv',rows)
    np.savez_compressed(out/'bootstrap_distributions.npz',**distributions)
    (out/'statistical_protocol.json').write_text(json.dumps(dict(draws=draws,interval='pointwise percentile 95%',
        external_unit='patient',paired=True,training_uncertainty_included=False,multiplicity_adjusted=False,
        limits='Conditional on these frozen models and this retrospectively reused external dataset. No medical causality.',
        prediction_seed=632919,localization_seed=632920,occlusion_seed=632921),indent=2))
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--draws',type=int,default=5000)
    a=p.parse_args();analyze(a.out,a.draws)
