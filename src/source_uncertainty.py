"""Recompute source-test intervals from aligned frozen prediction arrays."""
from pathlib import Path
import numpy as np
from review_statistics import prediction_bootstrap,save_rows

def run(source_archive,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=np.load(source_archive);y=data['labels'].reshape(-1)
    assert len(y)==624 and int(y.sum())==390
    names=[n for n in ('b28','plain','aug1','ce3','augmix') if n+'_scores' in data]
    contrasts=[('plain','b28'),('aug1','plain'),('ce3','plain'),('ce3','aug1')]
    if 'augmix' in names:contrasts += [('augmix','plain'),('augmix','ce3')]
    rows,distributions=prediction_bootstrap(y,{n:data[n+'_scores'] for n in names},draws=5000,seed=632922,
        scope='source_test',contrasts=contrasts)
    save_rows(out/'source_paired_intervals.csv',rows)
    np.savez_compressed(out/'bootstrap_distributions.npz',draws=distributions)
    return rows
