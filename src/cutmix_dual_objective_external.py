"""Post-hoc CutMix comparison against the immutable reproduced-paper model.

This script mirrors Colab cells 152-154 in the canonical MD6329 notebook.
It does not retrain either model. It verifies the frozen weights and the sealed
RSNA E_final manifest, evaluates both models on identical cases, and saves
paired prediction, bootstrap, Grad-CAM, and occlusion evidence.

Required Colab uploads under /content:
  stage10_minimal_restore.zip
  MD6329_Step13_CutMix_Exploration_Results.zip
  stage14_dependencies_20261020.zip
  final_evaluation.py
  external_split_manifest.json
  external_split_summary.json

Install requirements first with: pip install -r requirements_step14.txt
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import zipfile

import numpy as np
import torch


SEED = 632901
BASELINE_WEIGHT_SHA256 = "1e1753581d25a760068ec6215f485df93cbc947eab6f64a6bea2966a7d6cd564"
CUTMIX_P05_WEIGHT_SHA256 = "c72cc23468da8a1bc6140ef165a622b172ff8250a0e402dfa1698dece4ada939"
SEALED_MANIFEST_SHA256 = "4d484f337c8315eb7a767f3eed2c2fbd94a42fb86f34afebd851742283d22020"
EXTERNAL_DIAGNOSIS_SHA256 = "8735240a01fc3eae31dbf8eb044aa38508dfdf366ad5fc846db9bf827ef5b587"
RRR_IMPROVEMENT_SHA256 = "3c341afbacfca34e13aeef204b8b8777313f8a9a44b91de13f703110a2ff16fa"
BASELINE_SOURCE_THRESHOLD = 0.9871550798416138


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_required_inputs(project_root: Path) -> tuple[Path, Path, Path, Path]:
    stage10_root = project_root / "md6329_step10_resnet_reproduction"
    stage12_root = project_root / "md6329_step12_external_final"
    stage13_root = project_root / "md6329_step13_cutmix_exploration"
    stage14_root = project_root / "md6329_step14_cutmix_dual_objective"
    for path in (stage10_root, stage12_root, stage13_root, stage14_root):
        path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile("/content/stage10_minimal_restore.zip") as archive:
        archive.extractall(stage10_root)
    with zipfile.ZipFile("/content/MD6329_Step13_CutMix_Exploration_Results.zip") as archive:
        archive.extractall(stage13_root)
    with zipfile.ZipFile("/content/stage14_dependencies_20261020.zip") as archive:
        (stage10_root / "external_diagnosis.py").write_bytes(archive.read("external_diagnosis.py"))
        rrr_root = project_root / "md6329_step11_rrr_source_improvement"
        rrr_root.mkdir(parents=True, exist_ok=True)
        (rrr_root / "rrr_improvement.py").write_bytes(archive.read("rrr_improvement.py"))

    (stage12_root / "final_evaluation.py").write_bytes(Path("/content/final_evaluation.py").read_bytes())
    latest10 = json.loads((stage10_root / "latest_run.json").read_text(encoding="utf-8"))
    baseline_out = stage10_root / latest10["relative_path"]
    (baseline_out / "external_split_manifest.json").write_bytes(
        Path("/content/external_split_manifest.json").read_bytes()
    )
    (baseline_out / "external_split_summary.json").write_bytes(
        Path("/content/external_split_summary.json").read_bytes()
    )
    return stage10_root, stage12_root, stage13_root, stage14_root


def verify_frozen_inputs(
    project_root: Path, stage10_root: Path, stage13_root: Path
) -> tuple[Path, Path]:
    latest10 = json.loads((stage10_root / "latest_run.json").read_text(encoding="utf-8"))
    baseline_out = stage10_root / latest10["relative_path"]
    latest13 = json.loads((stage13_root / "latest_run.json").read_text(encoding="utf-8"))
    cutmix_out = stage13_root / latest13["relative_path"]
    p05_dir = cutmix_out / "candidate_p_0p5"
    rrr_root = project_root / "md6329_step11_rrr_source_improvement"

    checks = {
        baseline_out / "best_weights.pt": BASELINE_WEIGHT_SHA256,
        p05_dir / "best_weights.pt": CUTMIX_P05_WEIGHT_SHA256,
        baseline_out / "external_split_manifest.json": SEALED_MANIFEST_SHA256,
        stage10_root / "external_diagnosis.py": EXTERNAL_DIAGNOSIS_SHA256,
        rrr_root / "rrr_improvement.py": RRR_IMPROVEMENT_SHA256,
    }
    for path, expected in checks.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen-input hash mismatch: {path}\n{actual}\n!=\n{expected}")
    return baseline_out, p05_dir


def run(project_root: Path = Path("/content/md6329_project")) -> dict:
    stage10_root, stage12_root, stage13_root, stage14_root = extract_required_inputs(project_root)
    baseline_out, p05_dir = verify_frozen_inputs(project_root, stage10_root, stage13_root)

    baseline = import_path("baseline", stage10_root / "baseline.py")
    external_diagnosis = import_path("external_diagnosis", stage10_root / "external_diagnosis.py")
    import_path(
        "rrr_improvement",
        project_root / "md6329_step11_rrr_source_improvement" / "rrr_improvement.py",
    )
    cutmix_experiment = import_path("cutmix_experiment", stage13_root / "cutmix_experiment.py")
    final_evaluation = import_path("final_evaluation", stage12_root / "final_evaluation.py")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = stage14_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p05_training = json.loads((p05_dir / "training_summary.json").read_text(encoding="utf-8"))
    p05_threshold = float(p05_training["source_youden_threshold"])
    p05_sha = sha256(p05_dir / "best_weights.pt")
    freeze_record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_role": "immutable reproduced-paper control",
        "candidate": "ResNet18 B28 trained with CutMix beta=1.0 probability=0.5",
        "seed": SEED,
        "candidate_weights_sha256": p05_sha,
        "candidate_source_youden_threshold": p05_threshold,
        "primary_prediction_endpoint": (
            "RSNA Normal vs Lung Opacity accuracy at paper-comparable threshold 0.5"
        ),
        "secondary_prediction_endpoints": [
            "AUROC",
            "balanced accuracy",
            "Brier",
            "source-Youden metrics",
        ],
        "primary_xai_endpoint": (
            "paired area-corrected Grad-CAM energy inside public lesion boxes"
        ),
        "xai_success_rule": (
            "area-corrected energy and pointing-hit do not worsen; matched mean and blur "
            "occlusion differences do not worsen"
        ),
        "source_evidence": {
            "argmax_accuracy_baseline": 0.8621794871794872,
            "argmax_accuracy_candidate": 0.8701923076923077,
            "auroc_baseline": 0.9444115713346481,
            "auroc_candidate": 0.950597194827964,
            "xai_proxies_improved": 3,
        },
        "source_youden_warning": (
            "candidate own-Youden accuracy was lower; threshold calibration/transport weakness"
        ),
        "study_status": (
            "post-hoc exploratory because E_final was opened earlier; no further tuning after this run"
        ),
    }
    baseline.save_json(run_dir / "freeze_before_cutmix_external_evaluation.json", freeze_record)

    baseline_model = baseline.load_selected_model(baseline_out, device)
    candidate_model = cutmix_experiment.load_candidate(p05_dir, device)
    files = external_diagnosis.obtain_rsna(stage10_root)
    members = external_diagnosis.zip_members(files["images"])
    records, opening_audit = final_evaluation.verify_and_open_final(baseline_out, files)
    labels = np.array([row["target"] for row in records], dtype=np.int64)

    started = time.time()
    prediction = final_evaluation.paired_predict(
        {"baseline": baseline_model, "cutmix_p05": candidate_model},
        files["images"],
        members,
        records,
        device,
    )
    prediction_seconds = time.time() - started
    np.savez_compressed(
        run_dir / "cutmix_external_paired_predictions.npz",
        labels=labels,
        patient_ids=np.array([row["patient_id"] for row in records]),
        subset_img_ids=np.array([row["subset_img_id"] for row in records]),
        baseline_logits=prediction["baseline"]["logits"],
        baseline_scores=prediction["baseline"]["scores"],
        candidate_logits=prediction["cutmix_p05"]["logits"],
        candidate_scores=prediction["cutmix_p05"]["scores"],
    )

    metrics = {
        "baseline": external_diagnosis.prediction_metrics(
            records, prediction["baseline"]["scores"], 0.5
        ),
        "cutmix_p05": external_diagnosis.prediction_metrics(
            records, prediction["cutmix_p05"]["scores"], 0.5
        ),
        "baseline_source_threshold": external_diagnosis.prediction_metrics(
            records, prediction["baseline"]["scores"], BASELINE_SOURCE_THRESHOLD
        ),
        "cutmix_source_threshold": external_diagnosis.prediction_metrics(
            records, prediction["cutmix_p05"]["scores"], p05_threshold
        ),
    }
    base_argmax = metrics["baseline"]["primary_normal_vs_lung_opacity_argmax"]
    cutmix_argmax = metrics["cutmix_p05"]["primary_normal_vs_lung_opacity_argmax"]
    base_source = metrics["baseline_source_threshold"][
        "primary_normal_vs_lung_opacity_source_youden"
    ]
    cutmix_source = metrics["cutmix_source_threshold"][
        "primary_normal_vs_lung_opacity_source_youden"
    ]
    metric_keys = (
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
        "brier",
    )
    argmax_difference = {key: cutmix_argmax[key] - base_argmax[key] for key in metric_keys}
    source_difference = {key: cutmix_source[key] - base_source[key] for key in metric_keys}

    old_base_threshold = final_evaluation.BASELINE_THRESHOLD
    old_candidate_threshold = final_evaluation.CANDIDATE_THRESHOLD
    final_evaluation.BASELINE_THRESHOLD = 0.5
    final_evaluation.CANDIDATE_THRESHOLD = 0.5
    bootstrap_argmax = final_evaluation.patient_bootstrap_predictions(
        records, prediction["baseline"]["scores"], prediction["cutmix_p05"]["scores"]
    )
    final_evaluation.BASELINE_THRESHOLD = BASELINE_SOURCE_THRESHOLD
    final_evaluation.CANDIDATE_THRESHOLD = p05_threshold
    bootstrap_source = final_evaluation.patient_bootstrap_predictions(
        records, prediction["baseline"]["scores"], prediction["cutmix_p05"]["scores"]
    )
    final_evaluation.BASELINE_THRESHOLD = old_base_threshold
    final_evaluation.CANDIDATE_THRESHOLD = old_candidate_threshold

    explainability = final_evaluation.paired_explain(
        {"baseline": baseline_model, "candidate": candidate_model},
        files["images"],
        members,
        records,
        {"baseline": prediction["baseline"], "candidate": prediction["cutmix_p05"]},
        run_dir,
        device,
    )
    localization_difference = explainability["candidate_minus_baseline_localization"]
    occlusion_difference = explainability["candidate_minus_baseline_occlusion"]
    prediction_success = bool(
        cutmix_argmax["accuracy"] > base_argmax["accuracy"]
        and cutmix_argmax["auroc"] >= base_argmax["auroc"]
    )
    explainability_success = bool(
        localization_difference["area_corrected_energy_mean"] >= 0
        and localization_difference["pointing_hit_rate"] >= 0
        and occlusion_difference["mean_inside_minus_outside"] >= 0
        and occlusion_difference["blur_inside_minus_outside"] >= 0
    )
    summary = {
        "status": "cutmix_external_exploratory_complete_no_further_tuning",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "device": str(device),
        "candidate_weights_sha256": p05_sha,
        "opening_audit": opening_audit,
        "prediction_seconds": prediction_seconds,
        "prediction_metrics": metrics,
        "argmax_differences_candidate_minus_baseline": argmax_difference,
        "source_threshold_differences_candidate_minus_baseline": source_difference,
        "argmax_patient_bootstrap": bootstrap_argmax,
        "source_threshold_patient_bootstrap": bootstrap_source,
        "explainability": explainability,
        "external_prediction_success": prediction_success,
        "external_explainability_success": explainability_success,
        "external_dual_objective_success": bool(prediction_success and explainability_success),
        "E_final_status": (
            "reused post-hoc for frozen CutMix comparison; not independent and not used for tuning"
        ),
        "further_tuning_permitted_from_E_final": False,
    }
    baseline.save_json(run_dir / "cutmix_external_summary.json", summary)
    baseline.save_json(
        stage14_root / "latest_run.json",
        {
            "run_id": run_id,
            "relative_path": str(run_dir.relative_to(stage14_root)),
            "status": summary["status"],
        },
    )

    bundle = stage14_root / "MD6329_Step14_CutMix_Dual_Objective_p05.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage14_root))
        archive.write(stage14_root / "latest_run.json", "latest_run.json")
        for name in ("cutmix_dual_objective_external.py", "requirements_step14.txt"):
            path = stage14_root / name
            if path.exists():
                archive.write(path, name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("RESULT_BUNDLE", bundle, bundle.stat().st_size, sha256(bundle))
    return summary


if __name__ == "__main__":
    run()
