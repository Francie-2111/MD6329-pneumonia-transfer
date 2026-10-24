from pathlib import Path
import external_diagnosis as e
f=e.obtain_rsna(Path('/content/md6329_acjsd_recovery/md6329_step10_resnet_reproduction'))
print({k:str(v) for k,v in f.items()},flush=True)
