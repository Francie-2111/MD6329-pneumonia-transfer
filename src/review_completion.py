"""Frozen four-way B64 external ablation and explanation artifact completion.

Inputs are the exact historical checkpoints. No optimization, fitting, threshold
selection or candidate selection occurs here. External data were previously seen.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time, zipfile
import numpy as np
import torch
import b64_augmix as model_code
import b64_external_audit as audit
import external_diagnosis as ext
import baseline
from review_explanations import paired_explain
from review_statistics import analyze

HASHES = {
 'plain':'c51030d6e2646dbd4b5b1b4132528590dc85a609f79eb61ed29c4acfd90fdd2f',
 'augmix':'cf3a97804e8614b719367a8cdef4b23d9da5adbb2d48201a4dd6da238a4cf873',
 'aug1':'54c18d60e948b90153c6767046b03ffa4916d8be9603401bb9787fcd2a3b985c',
 'ce3':'f3a7650136ac9d379dc0d3824c6871d038b632f4a82526b9080b00a01f36ef97'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def geometry_audit(files,members,records,out):
    selected=sorted([r for r in records if r['target']==1],
        key=lambda r:hashlib.sha256(f"{audit.EXPLANATION_SEED}|{r['subset_img_id']}".encode()).hexdigest())[:300]
    rows=[]
    with zipfile.ZipFile(files['images']) as z:
        for r in selected:
            image,g=audit.read_dicom64(z,members[r['sop_instance_uid']])
            match=all(int(b['image_height'])==g['height'] and int(b['image_width'])==g['width'] for b in r['boxes'])
            mask=audit.mask_for_boxes64(r['boxes'],g)
            rows.append(dict(subset_img_id=r['subset_img_id'],height=g['height'],width=g['width'],
                annotation_dimensions_match=match,mask_pixels=int(mask.sum()),box_count=len(r['boxes'])))
    audit.save_csv(out/'box_geometry_audit.csv',rows)
    assert all(r['annotation_dimensions_match'] and r['mask_pixels']>0 for r in rows), 'Box geometry failed: stop rather than interpret misaligned CAMs.'
    return dict(n=len(rows),dimension_mismatches=0,empty_masks=0,
                preprocessing='same center-square crop, original x/y box coordinates, floor/ceil rasterization')

def run(root,checkpoints,out,draws=5000):
    root,checkpoints,out=map(Path,(root,checkpoints,out));out.mkdir(parents=True,exist_ok=True)
    weights={n:checkpoints/(n+'.pt') for n in HASHES}
    actual={n:sha(p) for n,p in weights.items()}
    assert actual==HASHES, actual
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline.seed_all(632901)
    plan=dict(weights_sha256=actual,threshold=.5,external_role='retrospective audit, no tuning',
        n_localization=300,n_occlusion_candidates=120,occlusion_outside_positions=20,
        bootstrap_draws=draws,bootstrap_unit='patient',checkpoint_selection='already frozen source validation AUROC',
        device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),started_utc=audit.utcnow())
    plan['evaluation_source_sha256']={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    planfile=out/'evaluation_plan.json'
    if planfile.exists():
        previous=json.loads(planfile.read_text());assert previous['weights_sha256']==actual
    else:audit.save_json(planfile,plan)
    models={}
    for n,p in weights.items():
        m=model_code.ResNet18().to(device)
        m.load_state_dict(torch.load(p,map_location=device,weights_only=False)['model_state']);m.eval();models[n]=m
    # Recover raw source predictions for AC-JSD and verify the archived controls.
    arrays=model_code.load_arrays(model_code.obtain_data(root/'md6329_step15_b64_augmix'),include_test=True)
    loader=model_code.make_loader(arrays['test_images'],arrays['test_labels'],128,False,model_code.SEED)
    cached=np.load(root/'source_statistics/source_predictions.npz')
    source_scores={'b28_scores':cached['b28_scores']}
    for n,m in models.items():
        _,scores,labels=model_code.predict(m,loader,device)
        assert np.array_equal(labels.reshape(-1),cached['labels'].reshape(-1))
        if n+'_scores' in cached:
            np.testing.assert_allclose(scores,cached[n+'_scores'],atol=2e-6,rtol=2e-6)
        source_scores[n+'_scores']=scores
    np.savez_compressed(out/'source_predictions.npz',labels=labels,**source_scores)
    from source_uncertainty import run as source_ci
    source_ci(out/'source_predictions.npz',out/'source_uncertainty')
    audit.save_json(out/'source_metrics.json',{n:model_code.metrics(labels,s) for n,s in source_scores.items()})
    files=ext.obtain_rsna(root/'md6329_step10_resnet_reproduction')
    records,opening=audit.verify_final_records(files);members=ext.zip_members(files['images'])
    audit.save_json(out/'external_opening_audit.json',opening)
    # Keep the complete patient/case/label correspondence needed for independent verification.
    audit.save_csv(out/'case_manifest.csv',[{k:r[k] for k in ('patient_id','subset_img_id','sop_instance_uid','class_name','target','primary_eligible')} for r in records])
    geometry=geometry_audit(files,members,records,out)
    predfile=out/'four_way_predictions.npz'
    if predfile.exists():
        data=np.load(predfile)
        assert np.array_equal(data['subset_img_ids'],np.array([r['subset_img_id'] for r in records]))
        pred={n:{'scores':data[n+'_scores'],'logits':data[n+'_logits']} for n in models}
    else:
        pred=audit.paired_predict(models,files['images'],members,records,device)
        np.savez_compressed(predfile,labels=np.array([r['target'] for r in records]),
            primary_eligible=np.array([r['primary_eligible'] for r in records],bool),
            patient_ids=np.array([r['patient_id'] for r in records]),subset_img_ids=np.array([r['subset_img_id'] for r in records]),
            **{n+'_'+k:v for n,d in pred.items() for k,v in d.items()})
    metrics={n:ext.prediction_metrics(records,pred[n]['scores'],.5) for n in models}
    audit.save_json(out/'four_way_metrics.json',metrics)
    print('FOUR_WAY_PREDICTION_METRICS',json.dumps(metrics),flush=True)
    xfile=out/'E_final_B64_explainability_summary.json'
    xai=json.loads(xfile.read_text()) if xfile.exists() else paired_explain(models,files['images'],members,records,pred,out,device)
    intervals=analyze(out,draws)
    summary=dict(status='complete',completed_utc=audit.utcnow(),weights_sha256=actual,
        prediction_metrics=metrics,explainability=xai,geometry_audit=geometry,interval_rows=len(intervals),
        external_tuning=False,causality_established=False)
    audit.save_json(out/'review_completion_summary.json',summary)
    archive=out.parent/'MD6329_Step19_Results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    print('STEP19_COMPLETE',str(archive),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoints',required=True)
    p.add_argument('--out',required=True);p.add_argument('--draws',type=int,default=5000)
    a=p.parse_args();run(a.root,a.checkpoints,a.out,a.draws)
