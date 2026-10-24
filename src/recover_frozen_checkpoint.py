from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import hashlib, json, platform, sys, time, zipfile, subprocess
import numpy as np
import torch
import b64_augmix as s
import b64_external_audit as audit
from training_recovery import train_recovery, ORIGINAL_SHA256

ROOT=Path('/content/md6329_acjsd_recovery')
DATA=ROOT/'md6329_step15_b64_augmix'
OUT=ROOT/'acjsd'
PLAIN=ROOT/'plain'
OUT.mkdir(exist_ok=True)
assert torch.cuda.is_available()
assert torch.cuda.get_device_name(0)=='Tesla T4'
assert torch.__version__=='2.11.0+cu130'
assert np.__version__=='2.1.3'
assert s.sha256(PLAIN/'best_weights.pt')=='c51030d6e2646dbd4b5b1b4132528590dc85a609f79eb61ed29c4acfd90fdd2f'
assert s.md5(s.obtain_data(DATA))==s.DATA_MD5
env={'started_utc':datetime.now(timezone.utc).isoformat(),'torch':torch.__version__,'numpy':np.__version__,
     'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cudnn':torch.backends.cudnn.version(),
     'source_sha256':{p.name:s.sha256(p) for p in (ROOT/'source').glob('*.py')},
     'original_expected_sha256':ORIGINAL_SHA256,
     'purpose':'Recover exact historical checkpoint if reproducible; otherwise archive a separate new 100-epoch fixed-protocol run. Never select or tune using external results.'}
s.save_json(ROOT/'execution_environment.json',env)
print(json.dumps(env,indent=2),flush=True)
config=asdict(s.TrainingConfig(mode='b64_augmix'))
if not (OUT/'protocol_before_training.json').exists():
    s.save_json(OUT/'protocol_before_training.json',{'config':config,'dataset_md5':s.DATA_MD5,'seed':s.SEED,
        'purpose':env['purpose'],'maximum_replay_epochs':100,
        'early_end_rule':'Only exact SHA256 equality to the already frozen historical weight binary; not metric-based early stopping.'})

# Dataset download is an independent process: no training RNG is touched.
download_log=(ROOT/'download_rsna.log').open('a')
downloader=subprocess.Popen([sys.executable,str(ROOT/'source/download_rsna.py')],stdout=download_log,stderr=subprocess.STDOUT)
summary=train_recovery(DATA,OUT,'b64_augmix')
print('RECONSTRUCTION_TRAINING',json.dumps(summary),flush=True)
with zipfile.ZipFile('/content/MD6329_ACJSD_frozen_model.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in OUT.iterdir():
        if p.is_file():z.write(p,'acjsd/'+p.name)
print('FROZEN_MODEL_ARCHIVED',flush=True)

assert s.sha256(OUT/'best_weights.pt') == ORIGINAL_SHA256, 'Recovered weights differ: do not merge with historical results.'
print('EXACT_CHECKPOINT_READY_FOR_DOWNLOAD',flush=True)
