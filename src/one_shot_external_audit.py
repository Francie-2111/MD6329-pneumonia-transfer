from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_project_model(checkpoint_path: Path, project_root: Path, device: torch.device) -> torch.nn.Module:
    candidates = [
        project_root / "src",
        project_root / "md6329_step15_b64_augmix",
        project_root / "03_实验代码与数据" / "md6329_step15_b64_augmix",
    ]
    source = next((path for path in candidates if (path / "b64_augmix.py").is_file()), None)
    if source is None:
        raise FileNotFoundError(f"b64_augmix.py was not found under {candidates}")
    sys.path.insert(0, str(source))
    import b64_augmix as step15  # type: ignore

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = step15.ResNet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def patient_from_path(path: str) -> str:
    match = re.search(r"patient(\d+)", path.replace("\\", "/"), flags=re.I)
    if not match:
        raise ValueError(f"Cannot parse patient ID from {path}")
    return f"patient{match.group(1)}"


def prepare_image(path: Path) -> torch.Tensor:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = float(image.min()), float(image.max())
    image = np.zeros_like(image) if hi <= lo else (image - lo) / (hi - lo)
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    image = image[top : top + side, left : left + side]
    resized = Image.fromarray(np.uint8(np.clip(image * 255.0, 0, 255))).resize((64, 64), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array[None, :, :] * 2.0 - 1.0)


def build_cohort(csv_path: Path, image_root: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(csv_path)
    required = {"Path", "Frontal/Lateral", "AP/PA", "No Finding", "Lung Opacity"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CheXpert columns: {sorted(missing)}")
    is_frontal = frame["Frontal/Lateral"].astype(str).str.lower().eq("frontal")
    is_ap_pa = frame["AP/PA"].astype(str).str.upper().isin({"AP", "PA"})
    frontal = frame[is_frontal & is_ap_pa].copy()
    lung_opacity = frontal["Lung Opacity"]
    positive = lung_opacity.eq(1)
    # In CheXpert, a positive No Finding label can coexist with a blank
    # pathology field.  The blank is accepted as negative only when
    # No Finding is explicitly positive; uncertain (-1) entries remain
    # excluded.  This rule was fixed before any CheXpert outcome was read.
    negative = frontal["No Finding"].eq(1) & (lung_opacity.eq(0) | lung_opacity.isna())
    cohort = frontal[positive | negative].copy()
    cohort["label"] = positive.loc[cohort.index].astype(int)
    cohort["patient_id"] = cohort["Path"].map(patient_from_path)
    cohort["image_path"] = cohort["Path"].map(lambda value: str((image_root / str(value)).resolve()))
    absent = [path for path in cohort["image_path"] if not Path(path).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} eligible images are missing; first={absent[0]}")
    conflict = cohort.groupby("patient_id")["label"].nunique()
    conflicting_patients = conflict.index[conflict > 1].tolist()
    conflicting_rows = int(cohort["patient_id"].isin(conflicting_patients).sum())
    if conflicting_patients:
        cohort = cohort[~cohort["patient_id"].isin(conflicting_patients)].copy()
    if cohort["label"].nunique() != 2:
        raise ValueError("Eligible cohort must contain both classes")
    audit = {
        "frontal_ap_pa_rows": int(len(frontal)),
        "eligible_rows_before_patient_conflict_exclusion": int((positive | negative).sum()),
        "excluded_conflicting_patients": int(len(conflicting_patients)),
        "excluded_conflicting_rows": conflicting_rows,
        "negative_rule": "No Finding == 1 and Lung Opacity is 0 or blank; Lung Opacity == -1 is excluded",
    }
    return cohort.reset_index(drop=True), audit


@torch.inference_mode()
def predict(model: torch.nn.Module, cohort: pd.DataFrame, device: torch.device) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(cohort), 64):
        images = torch.stack([prepare_image(Path(path)) for path in cohort.loc[start : start + 63, "image_path"]]).to(device)
        scores.append(torch.softmax(model(images), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores)


def metric_vector(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    negative = labels == 0
    positive = labels == 1
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "sensitivity": float(predictions[positive].mean()),
        "specificity": float((predictions[negative] == 0).mean()),
    }


def patient_groups(patient_ids: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(patient_ids == patient) for patient in np.unique(patient_ids)]


def resampling(
    labels: np.ndarray,
    plain: np.ndarray,
    acjsd: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    groups = patient_groups(patient_ids)
    rng = np.random.default_rng(seed)
    metric_names = list(metric_vector(labels, plain, threshold))
    observed_plain = metric_vector(labels, plain, threshold)
    observed_acjsd = metric_vector(labels, acjsd, threshold)
    observed = {name: observed_acjsd[name] - observed_plain[name] for name in metric_names}
    bootstrap = {name: [] for name in metric_names}
    for _ in range(bootstrap_draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        index = np.concatenate([groups[i] for i in sampled])
        left = metric_vector(labels[index], plain[index], threshold)
        right = metric_vector(labels[index], acjsd[index], threshold)
        for name in metric_names:
            bootstrap[name].append(right[name] - left[name])

    extreme = {name: 0 for name in metric_names}
    for _ in range(permutation_draws):
        swap_groups = rng.random(len(groups)) < 0.5
        p = plain.copy()
        a = acjsd.copy()
        for group_index, swap in enumerate(swap_groups):
            if swap:
                idx = groups[group_index]
                p[idx], a[idx] = acjsd[idx], plain[idx]
        left = metric_vector(labels, p, threshold)
        right = metric_vector(labels, a, threshold)
        for name in metric_names:
            if abs(right[name] - left[name]) >= abs(observed[name]) - 1e-15:
                extreme[name] += 1
    p_values = {name: (extreme[name] + 1) / (permutation_draws + 1) for name in metric_names}
    return bootstrap, p_values


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked one-shot CheXpert confirmation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--chexpert-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-locked-plan", default="")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    checkpoints = {
        name: (Path(record["checkpoint"]) if Path(record["checkpoint"]).is_absolute()
               else (args.config.parent / record["checkpoint"]).resolve())
        for name, record in config["models"].items()
    }
    if not args.execute:
        print(json.dumps({"status": "preflight_only", "config_status": config["status"], "output_exists": args.output_dir.exists()}, indent=2))
        return
    if args.acknowledge_locked_plan != "RUN_ONCE_NO_RETUNING":
        raise RuntimeError("Execution requires --acknowledge-locked-plan RUN_ONCE_NO_RETUNING")
    if config["status"] != "locked_ready" or int(config["external_run_count"]) != 0:
        raise RuntimeError("Configuration is not locked_ready with external_run_count=0")
    for name, path in checkpoints.items():
        if not path.is_file() or file_hash(path) != config["models"][name]["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for {name}")
    if args.output_dir.exists() or args.ledger.exists():
        raise FileExistsError("Outcome directory or ledger already exists; one-shot execution refused")

    lock = args.output_dir.with_suffix(".LOCK")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, datetime.now(timezone.utc).isoformat().encode())
    os.close(descriptor)
    args.output_dir.mkdir(parents=True)

    cohort, cohort_audit = build_cohort(args.chexpert_csv, args.image_root)
    cohort["image_sha256"] = [file_hash(Path(path)) for path in cohort["image_path"]]
    hash_counts = cohort["image_sha256"].value_counts()
    dataset_manifest = {
        "csv_sha256": file_hash(args.chexpert_csv),
        "eligible_images": len(cohort),
        "patients": int(cohort["patient_id"].nunique()),
        "class_counts": {str(k): int(v) for k, v in cohort["label"].value_counts().sort_index().items()},
        "cohort_audit": cohort_audit,
        "duplicate_image_hash_groups": int((hash_counts > 1).sum()),
        "images_in_duplicate_hash_groups": int(hash_counts[hash_counts > 1].sum()),
        "image_manifest_sha256": canonical_hash(cohort[["Path", "patient_id", "label", "image_sha256"]].to_dict("records")),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    for name, checkpoint in checkpoints.items():
        scores[name] = predict(load_project_model(checkpoint, args.project_root, device), cohort, device)
    labels = cohort["label"].to_numpy(dtype=int)
    patients = cohort["patient_id"].to_numpy()
    metrics = {name: metric_vector(labels, values, float(config["threshold"])) for name, values in scores.items()}
    bootstrap, p_values = resampling(
        labels,
        scores["plain"],
        scores["acjsd"],
        patients,
        float(config["threshold"]),
        int(config["bootstrap_draws"]),
        int(config["permutation_draws"]),
        int(config["seed"]) + 21,
    )
    contrasts = {}
    for name, draws in bootstrap.items():
        difference = metrics["acjsd"][name] - metrics["plain"][name]
        contrasts[name] = {
            "difference_acjsd_minus_plain": difference,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "permutation_p_two_sided": p_values[name],
        }
    secondary = {name: p_values[name] for name in config["secondary_endpoints"]}
    adjusted = holm(secondary)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p"] = value

    predictions = cohort[["Path", "patient_id", "label", "image_sha256"]].copy()
    predictions["plain_score"] = scores["plain"]
    predictions["acjsd_score"] = scores["acjsd"]
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    np.savez_compressed(args.output_dir / "resampling_distributions.npz", **{f"delta_{key}": np.asarray(value) for key, value in bootstrap.items()})
    results = {
        "status": "one_shot_complete_locked",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": config["primary_endpoint"],
        "primary_result": contrasts["auroc"],
        "metrics": metrics,
        "contrasts": contrasts,
        "dataset_manifest": dataset_manifest,
        "interpretation_rule": "confirmation requires a positive AUROC point difference and two-sided p<0.05",
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    code_files = [Path(__file__), args.plan, args.config]
    ledger_row = {
        "completed_utc": results["completed_utc"],
        "status": results["status"],
        "analysis_plan_sha256": file_hash(args.plan),
        "config_sha256": file_hash(args.config),
        "code_bundle_sha256": canonical_hash({path.name: file_hash(path) for path in code_files}),
        "dataset_manifest_sha256": canonical_hash(dataset_manifest),
        "checkpoint_sha256": {name: file_hash(path) for name, path in checkpoints.items()},
        "git_commit": None,
        "repository_state": "repository_not_initialized",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
    }
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    with args.ledger.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(ledger_row, sort_keys=True) + "\n")
    (args.output_dir / "ledger_row.json").write_text(json.dumps(ledger_row, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
