from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CHECKPOINTS = (
    "best_validation_brier",
    "best_validation_auroc",
    "epoch_40",
    "last_epoch",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_module(project_root: Path) -> Path:
    candidates = [
        project_root / "md6329_step15_b64_augmix",
        project_root / "03_实验代码与数据" / "md6329_step15_b64_augmix",
    ]
    found = next((path for path in candidates if (path / "b64_augmix.py").is_file()), None)
    if found is None:
        raise FileNotFoundError(f"b64_augmix.py was not found under {candidates}")
    return found


def expanded_arrays(dataset: Path, indices: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset)
    split = np.load(indices)
    moved = split["moved_train_indices"]
    return {
        "validation_images": np.concatenate([data["val_images"], data["train_images"][moved]], axis=0),
        "validation_labels": np.concatenate([
            data["val_labels"].reshape(-1), data["train_labels"][moved].reshape(-1)
        ]),
        "test_images": data["test_images"],
        "test_labels": data["test_labels"].reshape(-1),
    }


def choose_lambda(run_root: Path, candidates: tuple[int, ...]) -> dict:
    rows = []
    for weight in candidates:
        summary_path = run_root / f"acjsd_lambda_{weight}" / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "lambda": weight,
            "best_validation_auroc": float(summary["best_validation_auroc"]),
            "best_validation_auroc_epoch": int(summary["best_validation_auroc_epoch"]),
            "best_validation_brier": float(summary["best_validation_brier"]),
            "best_validation_brier_epoch": int(summary["best_validation_brier_epoch"]),
            "training_seconds": float(summary["training_seconds"]),
        })
    best_auc = max(row["best_validation_auroc"] for row in rows)
    eligible = [row for row in rows if row["best_validation_auroc"] >= best_auc - 0.005]
    selected = min(eligible, key=lambda row: (row["best_validation_brier"], row["lambda"]))
    return {
        "selection_rule": "lowest expanded-validation Brier among candidates within 0.005 AUROC of the best; exact ties choose smaller lambda",
        "best_candidate_auroc": best_auc,
        "eligible_lambdas": [row["lambda"] for row in eligible],
        "selected_lambda": selected["lambda"],
        "candidates": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(step15, path: Path, loaders: dict, device: torch.device) -> tuple[dict, dict, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = step15.ResNet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metrics = {}
    for split_name, loader in loaders.items():
        _, scores, labels = step15.predict(model, loader, device)
        metrics[split_name] = step15.metrics(labels, scores)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics["validation"], metrics["test"], int(checkpoint["epoch"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_module(args.project_root)))
    import b64_augmix as step15  # type: ignore

    lambda_selection = choose_lambda(args.run_root, (1, 3, 12, 36))
    selected_lambda = int(lambda_selection["selected_lambda"])
    (args.output_dir / "lambda_selection.json").write_text(
        json.dumps(lambda_selection, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "lambda_sweep.csv", lambda_selection["candidates"])

    arrays = expanded_arrays(args.dataset, args.indices)
    loaders = {
        "validation": step15.make_loader(arrays["validation_images"], arrays["validation_labels"], 128, False, 632901),
        "test": step15.make_loader(arrays["test_images"], arrays["test_labels"], 128, False, 632901),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = {
        "plain": args.run_root / "plain",
        "aug1": args.run_root / "aug1",
        "ce3": args.run_root / "ce3",
        "acjsd": args.run_root / f"acjsd_lambda_{selected_lambda}",
    }
    rows = []
    for model_name, run_dir in run_dirs.items():
        for checkpoint_name in CHECKPOINTS:
            checkpoint_path = run_dir / f"{checkpoint_name}.pt"
            validation, test, epoch = evaluate_checkpoint(step15, checkpoint_path, loaders, device)
            row = {
                "model": model_name,
                "lambda": selected_lambda if model_name == "acjsd" else "",
                "checkpoint_rule": checkpoint_name,
                "epoch": epoch,
                "checkpoint_sha256": sha256(checkpoint_path),
            }
            for split_name, values in (("validation", validation), ("test", test)):
                for metric in ("accuracy", "balanced_accuracy", "auroc", "brier", "ece10", "sensitivity", "specificity"):
                    row[f"{split_name}_{metric}"] = values[metric]
            rows.append(row)
            print(model_name, checkpoint_name, epoch, test["auroc"], test["accuracy"], flush=True)

    write_csv(args.output_dir / "checkpoint_sensitivity.csv", rows)
    result = {
        "status": "source_checkpoint_sensitivity_complete_external_unopened",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "selected_lambda": selected_lambda,
        "device": str(device),
        "rows": rows,
        "interpretation": "All rules are evaluated after training. Source test results are used for sensitivity reporting only and do not alter lambda or checkpoint selection.",
    }
    (args.output_dir / "checkpoint_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    colors = {"plain": "#666666", "aug1": "#b17b28", "ce3": "#925a80", "acjsd": "#24738b"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    x = np.arange(len(CHECKPOINTS))
    for model_name in run_dirs:
        chosen = [next(row for row in rows if row["model"] == model_name and row["checkpoint_rule"] == rule) for rule in CHECKPOINTS]
        axes[0].plot(x, [row["test_auroc"] for row in chosen], marker="o", label=model_name, color=colors[model_name])
        axes[1].plot(x, [row["test_brier"] for row in chosen], marker="o", label=model_name, color=colors[model_name])
    for axis, ylabel in zip(axes, ("Source-test AUROC", "Source-test Brier")):
        axis.set_xticks(x, ("best Brier", "best AUROC", "epoch 40", "last"), rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(args.output_dir / "checkpoint_sensitivity.pdf")
    fig.savefig(args.output_dir / "checkpoint_sensitivity.png", dpi=180)
    plt.close(fig)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["status"] = "locked_ready"
    config["selected_lambda"] = selected_lambda
    config["models"] = {
        "plain": {
            "checkpoint": str((run_dirs["plain"] / "best_validation_brier.pt").resolve()),
            "sha256": sha256(run_dirs["plain"] / "best_validation_brier.pt"),
        },
        "acjsd": {
            "checkpoint": str((run_dirs["acjsd"] / "best_validation_brier.pt").resolve()),
            "sha256": sha256(run_dirs["acjsd"] / "best_validation_brier.pt"),
        },
    }
    config["source_lock_completed_utc"] = result["completed_utc"]
    config["source_sensitivity_sha256"] = sha256(args.output_dir / "checkpoint_sensitivity.json")
    args.config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"selected_lambda": selected_lambda, "config_status": config["status"]}, indent=2))


if __name__ == "__main__":
    main()
