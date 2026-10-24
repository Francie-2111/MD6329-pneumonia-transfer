from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import b64_augmix as step15


MODE = "b64_aug_ce3_jsd0"
SEED = step15.SEED


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refusing to save an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path) -> Path:
    root = Path(root)
    dataset = step15.obtain_data(root)
    step15.load_arrays(dataset, include_test=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "runs" / run_id / MODE
    out.mkdir(parents=True)
    config = step15.TrainingConfig(mode=MODE, jsd_weight=0.0)
    protocol = {
        "status": "protocol_frozen_before_training",
        "created_utc": utcnow(),
        "purpose": "separate the contribution of JSD from the otherwise identical three-view acquisition augmentation",
        "dataset": "PneumoniaMNIST v2 64x64",
        "dataset_md5": step15.md5(dataset),
        "dataset_sha256": step15.sha256(dataset),
        "splits_loaded_before_training": {split: step15.EXPECTED_SPLITS[split] for split in ("train", "val")},
        "test_loaded_before_training": False,
        "config": asdict(config),
        "objective": "mean cross-entropy over clean image and two independently augmented views; JSD is logged but multiplied by zero",
        "augmentation_identity": "same acquisition_view implementation, ranges, generator seed, and number of views as completed B64 AC-JSD",
        "checkpoint_selection": "best clean source-validation AUROC",
        "internal_test_policy": "one evaluation after the checkpoint is frozen; this is a retrospective component ablation because prior test results are already known",
        "external_policy": "no external data are used for fitting or checkpoint selection; this first run stops after source validation and internal test",
    }
    save_json(out / "protocol_before_training.json", protocol)
    return out


def train(root: Path, out: Path) -> dict:
    root, out = Path(root), Path(out)
    config = step15.TrainingConfig(mode=MODE, jsd_weight=0.0)
    step15.seed_all(config.seed)
    arrays = step15.load_arrays(step15.obtain_data(root), include_test=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = step15.ResNet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, list(config.milestones), gamma=config.gamma
    )
    train_loader = step15.make_loader(
        arrays["train_images"], arrays["train_labels"], config.batch_size, True, config.seed
    )
    val_loader = step15.make_loader(
        arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed
    )
    augmentation_generator = torch.Generator().manual_seed(config.seed + 1)
    best_auc, best_epoch, rows = -np.inf, None, []
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = total_ce = total_jsd = 0.0
        count = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            view1 = step15.acquisition_view(x, augmentation_generator)
            view2 = step15.acquisition_view(x, augmentation_generator)
            outputs = [model(x), model(view1), model(view2)]
            ce = sum(F.cross_entropy(output, y) for output in outputs) / 3.0
            diagnostic_jsd = step15.jsd_loss(outputs)
            loss = ce
            loss.backward()
            optimizer.step()
            batch = len(y)
            total_loss += float(loss.detach()) * batch
            total_ce += float(ce.detach()) * batch
            total_jsd += float(diagnostic_jsd.detach()) * batch
            count += batch
            correct += int((outputs[0].argmax(1) == y).sum())
        val_logits, val_scores, val_labels = step15.predict(model, val_loader, device)
        val = step15.metrics(val_labels, val_scores)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / count,
            "train_ce": total_ce / count,
            "train_jsd_diagnostic_not_optimized": total_jsd / count,
            "train_accuracy_clean_view": correct / count,
            "val_accuracy": val["accuracy"],
            "val_auroc": val["auroc"],
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if val["auroc"] > best_auc:
            best_auc, best_epoch = val["auroc"], epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_auroc": best_auc,
                    "config": asdict(config),
                    "objective": "three_view_mean_cross_entropy_jsd_weight_zero",
                },
                out / "best_weights.pt",
            )
            np.savez_compressed(
                out / "best_validation_predictions.npz",
                logits=val_logits,
                scores=val_scores,
                labels=val_labels,
            )
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"{MODE} epoch={epoch:03d} val_auc={val['auroc']:.6f} "
                f"val_acc={val['accuracy']:.6f}",
                flush=True,
            )
    save_csv(out / "learning_curve.csv", rows)
    summary = {
        "status": "training_complete_validation_checkpoint_frozen",
        "mode": MODE,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "best_epoch": int(best_epoch),
        "best_validation_auroc": float(best_auc),
        "weights_sha256": step15.sha256(out / "best_weights.pt"),
        "training_seconds": time.time() - started,
        "jsd_weight": 0.0,
    }
    save_json(out / "training_summary.json", summary)
    return summary


def load_model(out: Path, device: torch.device) -> torch.nn.Module:
    return step15.load_model(Path(out), device)


def validate(root: Path, out: Path) -> dict:
    result = step15.validate(Path(root), Path(out))
    result["status"] = "validation_complete_checkpoint_frozen"
    result["mode"] = MODE
    result["objective"] = "three-view mean cross-entropy; JSD weight zero"
    save_json(Path(out) / "validation_robustness_and_xai.json", result)
    return result


def evaluate_internal(root: Path, out: Path, label: str) -> dict:
    root, out = Path(root), Path(out)
    arrays = step15.load_arrays(step15.obtain_data(root), include_test=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(out, device)
    loader = step15.make_loader(
        arrays["test_images"], arrays["test_labels"], 128, False, SEED
    )
    logits, scores, labels = step15.predict(model, loader, device)
    result = {
        "status": "retrospective_internal_test_ablation_complete",
        "model": label,
        "weights_sha256": step15.sha256(out / "best_weights.pt"),
        "metrics": step15.metrics(labels, scores),
        "explainability": step15.explanation_proxies(
            model, arrays["test_images"], arrays["test_labels"], device
        ),
        "interpretation": "the source test was already known before this ablation; use for attribution, not confirmatory model selection",
    }
    np.savez_compressed(
        out / "internal_test_predictions_step17.npz",
        logits=logits,
        scores=scores,
        labels=labels,
    )
    save_json(out / "internal_test_and_xai_step17.json", result)
    return result


def mean_cam_stability(xai: dict) -> float:
    return float(np.mean([item["mean_cam_cosine"] for item in xai["cam_stability"].values()]))


def compact_row(label: str, validation: dict, internal: dict) -> dict:
    va, vx = validation["aggregate"], validation["explainability"]
    im, ix = internal["metrics"], internal["explainability"]
    return {
        "model": label,
        "val_clean_accuracy": va["clean_accuracy"],
        "val_clean_auroc": va["clean_auroc"],
        "val_mean_perturbed_auroc": va["mean_perturbed_auroc"],
        "val_worst_perturbed_accuracy": va["worst_perturbed_accuracy"],
        "val_mean_cam_stability": mean_cam_stability(vx),
        "val_center_minus_border_drop": vx["center_minus_border_drop"],
        "test_accuracy": im["accuracy"],
        "test_balanced_accuracy": im["balanced_accuracy"],
        "test_auroc": im["auroc"],
        "test_brier": im["brier"],
        "test_sensitivity": im["sensitivity"],
        "test_specificity": im["specificity"],
        "test_mean_cam_stability": mean_cam_stability(ix),
        "test_center_minus_border_drop": ix["center_minus_border_drop"],
    }


def compare(
    plain_validation: dict,
    plain_internal: dict,
    ce3_validation: dict,
    ce3_internal: dict,
    full_validation: dict,
    full_internal: dict,
) -> dict:
    rows = [
        compact_row("B64 plain", plain_validation, plain_internal),
        compact_row("B64 CE3 / JSD=0", ce3_validation, ce3_internal),
        compact_row("B64 CE3 + JSD=12 (AC-JSD)", full_validation, full_internal),
    ]
    by_name = {row["model"]: row for row in rows}
    plain = by_name["B64 plain"]
    ce3 = by_name["B64 CE3 / JSD=0"]
    full = by_name["B64 CE3 + JSD=12 (AC-JSD)"]
    keys = [key for key in plain if key != "model"]
    result = {
        "status": "source_component_ablation_complete",
        "created_utc": utcnow(),
        "seed": SEED,
        "models": rows,
        "deltas": {
            "ce3_minus_plain": {key: ce3[key] - plain[key] for key in keys},
            "full_minus_ce3_estimated_jsd_increment": {key: full[key] - ce3[key] for key in keys},
            "full_minus_plain_total": {key: full[key] - plain[key] for key in keys},
        },
        "attribution_boundary": [
            "CE3 minus plain estimates the combined effect of acquisition augmentation plus three-view supervised cross-entropy.",
            "Full minus CE3 estimates the incremental effect of adding JSD weight 12 with the same view construction.",
            "A separate one-view augmentation control would be required to isolate multi-view supervision from augmentation itself.",
            "All comparisons use one seed and previously known source test data; they do not estimate retraining uncertainty or confirmatory significance.",
        ],
    }
    return result


def smoke_test() -> dict:
    step15.seed_all(SEED)
    x = torch.linspace(-1, 1, 2 * 64 * 64).reshape(2, 1, 64, 64)
    generator = torch.Generator().manual_seed(SEED + 1)
    a = step15.acquisition_view(x, generator)
    generator = torch.Generator().manual_seed(SEED + 1)
    b = step15.acquisition_view(x, generator)
    if not torch.equal(a, b) or a.shape != x.shape:
        raise AssertionError("augmentation reproducibility check failed")
    return {
        "mode": MODE,
        "seed": SEED,
        "augmentation_reproducible": True,
        "shape": list(a.shape),
        "jsd_weight": 0.0,
    }


if __name__ == "__main__":
    print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))
