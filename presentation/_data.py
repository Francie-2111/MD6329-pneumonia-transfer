"""Recorded aggregate results used by the public LecTrace presentation."""

INTERNAL_RESULTS = [
    {"model": "B28", "accuracy": 0.8622, "auroc": 0.9444},
    {"model": "B64 plain", "accuracy": 0.9054, "auroc": 0.9751},
]

EXTERNAL_RESULTS = [
    {"dataset": "RSNA", "model": "B64 plain", "accuracy": 0.8057, "auroc": 0.8720},
    {"dataset": "RSNA", "model": "B64 AC-JSD", "accuracy": 0.8093, "auroc": 0.9012},
    {"dataset": "CheXpert", "model": "B64 plain", "accuracy": 0.6224, "auroc": 0.7212},
    {"dataset": "CheXpert", "model": "B64 AC-JSD", "accuracy": 0.8392, "auroc": 0.9310},
]

XAI_RESULTS = [
    {"model": "Plain", "area-corrected energy": -0.0336, "pointing hit": 0.0667, "mean-fill occlusion": 0.0369},
    {"model": "Aug1", "area-corrected energy": 0.0088, "pointing hit": 0.1200, "mean-fill occlusion": 0.0809},
    {"model": "CE3", "area-corrected energy": -0.0257, "pointing hit": 0.0567, "mean-fill occlusion": 0.0215},
    {"model": "AC-JSD", "area-corrected energy": -0.0162, "pointing hit": 0.0867, "mean-fill occlusion": 0.0913},
    {"model": "Center prior", "area-corrected energy": 0.1011, "pointing hit": 0.1300, "mean-fill occlusion": None},
]
