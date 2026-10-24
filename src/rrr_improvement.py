from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import baseline


@dataclass(frozen=True)
class RRRConfig:
    seed: int = baseline.SEED
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    milestones: tuple[int, int] = (50, 75)
    gamma: float = 0.1
    border_fraction: float = 0.03
    lambda_rrr: float = 0.1
    selection_checkpoint: str = "maximum validation AUROC"


CANDIDATE_LAMBDAS = (0.1, 1.0)
AUROC_TOLERANCE = 0.01
BOOTSTRAP_REPLICATES = 5000


def border_mask(height: int = 28, width: int = 28, fraction: float = 0.03, device=None) -> torch.Tensor:
    pixels = max(1, int(math.ceil(min(height, width) * fraction)))
    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    mask[:, :, :pixels, :] = 1
    mask[:, :, -pixels:, :] = 1
    mask[:, :, :, :pixels] = 1
    mask[:, :, :, -pixels:] = 1
    return mask


def random_same_area_mask(height: int = 28, width: int = 28, fraction: float = 0.03, seed: int = baseline.EXPLANATION_SEED, device=None) -> torch.Tensor:
    edge = border_mask(height, width, fraction).cpu().numpy()[0, 0].astype(bool)
    count = int(edge.sum())
    available = np.flatnonzero(~edge.reshape(-1))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(available, count, replace=False)
    random_mask = np.zeros(height * width, dtype=np.float32)
    random_mask[chosen] = 1
    return torch.from_numpy(random_mask.reshape(1, 1, height, width)).to(device=device)


def rrr_penalty(logits: torch.Tensor, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    margin = logits[:, 1] - logits[:, 0]
    gradient = torch.autograd.grad(margin.sum(), inputs, create_graph=True, retain_graph=True)[0]
    denominator = inputs.shape[0] * mask.sum().clamp_min(1.0)
    return (gradient.square() * mask).sum() / denominator


def _prediction_metrics_from_npz(path: Path, threshold: float) -> dict:
    values = np.load(path)
    return baseline.metrics(values["labels"], values["scores"], threshold)


def load_baseline_context(baseline_out: Path, arrays: dict[str, np.ndarray], device: torch.device) -> dict:
    training = json.loads((baseline_out / "training_summary.json").read_text(encoding="utf-8"))
    threshold = float(training["source_youden_threshold"])
    val_metrics = _prediction_metrics_from_npz(baseline_out / "best_validation_predictions.npz", threshold)
    model = baseline.load_selected_model(baseline_out, device)
    mechanism = gradient_mechanism(model, arrays, "val", device)
    explain_path = baseline_out / "source_val_explainability.json"
    if explain_path.exists():
        explainability = json.loads(explain_path.read_text(encoding="utf-8"))
    else:
        explainability = baseline.source_explainability(model, arrays, "val", baseline_out, device)
    return {
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "validation_mechanism": mechanism,
        "validation_explainability": explainability,
        "weights_sha256": baseline.sha256(baseline_out / "best_weights.pt"),
        "run_dir": str(baseline_out),
    }


def prepare_protocol(stage_root: Path, baseline_out: Path, source_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    run_id = baseline.utcnow().replace(":", "").replace("-", "").replace("+0000", "Z").replace(".", "_")
    out = stage_root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "source_improvement_protocol_frozen_before_training",
        "created_utc": baseline.utcnow(),
        "method": "RRR-inspired input-gradient penalty on the outer 3% image band",
        "rationale": "external E_diag diagnosis found weak lesion localization and edge/corner attention",
        "candidate_lambdas": list(CANDIDATE_LAMBDAS),
        "shared_seed": baseline.SEED,
        "shared_initialization_order_budget": True,
        "dataset": "PneumoniaMNIST 28 source domain only",
        "source_sha256": baseline.sha256(source_path),
        "splits": {split: int(len(arrays[f"{split}_labels"])) for split in ("train", "val", "test")},
        "selection_rule": {
            "checkpoint_within_candidate": "maximum validation AUROC",
            "candidate_acceptance": [
                "validation balanced accuracy strictly exceeds baseline",
                "validation AUROC is no more than 0.01 below baseline",
                "validation border-occlusion probability change is lower than baseline",
            ],
            "tie_break": "higher validation balanced accuracy, then smaller lambda",
            "threshold": "Youden threshold estimated separately on each model's source validation predictions",
        },
        "mechanism_control": "fixed equal-area random interior mask; gradient statistics are supportive only",
        "test_policy": "source test is evaluated once only after a candidate is selected and frozen",
        "external_policy": "E_diag and E_final are not loaded or evaluated in this stage",
        "baseline_run_dir": str(baseline_out),
        "baseline_weights_sha256": baseline.sha256(baseline_out / "best_weights.pt"),
        "implementation": {
            "target": "pneumonia logit minus normal logit",
            "penalty": "mean squared input gradient inside the fixed outer band",
            "border_pixels_at_28": int(border_mask().sum().item()),
            "training": asdict(RRRConfig()),
        },
    }
    baseline.save_json(out / "plan_before_training.json", protocol)
    baseline.save_json(out / "environment.json", baseline.environment_record())
    return out


def train_candidate(arrays: dict[str, np.ndarray], out: Path, lambda_rrr: float) -> dict:
    config = RRRConfig(lambda_rrr=lambda_rrr)
    baseline.seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = baseline.resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones), gamma=config.gamma)
    criterion = nn.CrossEntropyLoss()
    train_loader = baseline.loader(arrays["train_images"], arrays["train_labels"], config.batch_size, True, config.seed)
    val_loader = baseline.loader(arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed)
    mask = border_mask(28, 28, config.border_fraction, device)
    rows: list[dict] = []
    best_auc = -np.inf
    best_epoch = None
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        sums = {"total": 0.0, "ce": 0.0, "rrr": 0.0, "correct": 0, "count": 0}
        for x, y in train_loader:
            x = x.to(device).requires_grad_(True)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = criterion(logits, y)
            penalty = rrr_penalty(logits, x, mask)
            loss = ce + config.lambda_rrr * penalty
            loss.backward()
            optimizer.step()
            n = len(y)
            sums["total"] += float(loss.detach()) * n
            sums["ce"] += float(ce.detach()) * n
            sums["rrr"] += float(penalty.detach()) * n
            sums["correct"] += int((logits.argmax(1) == y).sum())
            sums["count"] += n
        val_logits, val_scores, val_labels = baseline.predict(model, val_loader, device)
        val_metrics = baseline.metrics(val_labels, val_scores, 0.5)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": sums["total"] / sums["count"],
            "train_ce": sums["ce"] / sums["count"],
            "train_rrr_penalty": sums["rrr"] / sums["count"],
            "train_accuracy": sums["correct"] / sums["count"],
            "val_auroc": val_metrics["auroc"],
            "val_accuracy_0_5": val_metrics["accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_auroc": best_auc, "config": asdict(config)}, out / "best_weights.pt")
            np.savez_compressed(out / "best_validation_predictions.npz", logits=val_logits, scores=val_scores, labels=val_labels)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"lambda={lambda_rrr:g} epoch={epoch:03d} loss={row['train_loss']:.4f} rrr={row['train_rrr_penalty']:.6f} val_auc={row['val_auroc']:.4f}", flush=True)
    baseline.save_csv(out / "learning_curve.csv", rows)
    validation = np.load(out / "best_validation_predictions.npz")
    threshold = baseline.threshold_youden(validation["labels"], validation["scores"])
    summary = {
        "lambda_rrr": lambda_rrr,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_auroc": float(best_auc),
        "source_youden_threshold": float(threshold),
        "training_seconds": time.time() - started,
        "best_weights_sha256": baseline.sha256(out / "best_weights.pt"),
        "config": asdict(config),
    }
    baseline.save_json(out / "training_summary.json", summary)
    return summary


def load_candidate(path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path / "best_weights.pt", map_location=device, weights_only=False)
    model = baseline.resnet18().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def gradient_mechanism(model: nn.Module, arrays: dict[str, np.ndarray], split: str, device: torch.device, maximum: int = 200) -> dict:
    labels = arrays[f"{split}_labels"]
    indices = baseline.stratified_indices(labels, maximum, baseline.EXPLANATION_SEED + (0 if split == "val" else 1))
    x_all = baseline.normalized_tensor(arrays[f"{split}_images"][indices])
    edge = border_mask(device=device)
    random_mask = random_same_area_mask(device=device)
    edge_sum = float(edge.sum())
    random_sum = float(random_mask.sum())
    total_edge_energy = total_random_energy = total_all_energy = 0.0
    count = 0
    for start in range(0, len(x_all), 64):
        x = x_all[start:start + 64].to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        margin = logits[:, 1] - logits[:, 0]
        gradient = torch.autograd.grad(margin.sum(), x, create_graph=False)[0].square()
        total_edge_energy += float((gradient * edge).sum())
        total_random_energy += float((gradient * random_mask).sum())
        total_all_energy += float(gradient.sum())
        count += len(x)
    edge_mean = total_edge_energy / (count * edge_sum)
    random_mean = total_random_energy / (count * random_sum)
    return {
        "split": split,
        "n": int(count),
        "border_fraction": 0.03,
        "border_pixels": int(edge_sum),
        "random_control_pixels": int(random_sum),
        "border_gradient_squared_mean": edge_mean,
        "random_control_gradient_squared_mean": random_mean,
        "border_to_random_ratio": float(edge_mean / random_mean) if random_mean > 0 else None,
        "border_energy_fraction_of_total": float(total_edge_energy / total_all_energy) if total_all_energy > 0 else None,
        "interpretation": "supportive mechanism measure only; candidate selection uses independent border occlusion",
    }


def evaluate_candidate_validation(candidate_dir: Path, arrays: dict[str, np.ndarray], device: torch.device) -> dict:
    training = json.loads((candidate_dir / "training_summary.json").read_text(encoding="utf-8"))
    threshold = float(training["source_youden_threshold"])
    model = load_candidate(candidate_dir, device)
    validation_metrics = _prediction_metrics_from_npz(candidate_dir / "best_validation_predictions.npz", threshold)
    explainability = baseline.source_explainability(model, arrays, "val", candidate_dir, device)
    mechanism = gradient_mechanism(model, arrays, "val", device)
    result = {
        "lambda_rrr": float(training["lambda_rrr"]),
        "threshold": threshold,
        "validation_metrics": validation_metrics,
        "validation_explainability": explainability,
        "validation_mechanism": mechanism,
    }
    baseline.save_json(candidate_dir / "validation_evaluation.json", result)
    return result


def select_candidate(baseline_context: dict, candidates: list[dict]) -> dict:
    b_metrics = baseline_context["validation_metrics"]
    b_border = baseline_context["validation_explainability"]["border_occlusion_probability_abs_change_mean"]
    rows = []
    for candidate in candidates:
        c_metrics = candidate["validation_metrics"]
        c_border = candidate["validation_explainability"]["border_occlusion_probability_abs_change_mean"]
        checks = {
            "balanced_accuracy_strictly_higher": c_metrics["balanced_accuracy"] > b_metrics["balanced_accuracy"],
            "auroc_within_tolerance": c_metrics["auroc"] >= b_metrics["auroc"] - AUROC_TOLERANCE,
            "border_occlusion_dependency_lower": c_border < b_border,
        }
        rows.append({**candidate, "acceptance_checks": checks, "accepted": all(checks.values())})
    accepted = [r for r in rows if r["accepted"]]
    selected = None
    if accepted:
        accepted.sort(key=lambda r: (-r["validation_metrics"]["balanced_accuracy"], r["lambda_rrr"]))
        selected = accepted[0]
    return {
        "status": "candidate_selected_and_frozen" if selected else "no_candidate_met_pre_registered_rule",
        "baseline": baseline_context,
        "candidates": rows,
        "selected_lambda": None if selected is None else selected["lambda_rrr"],
        "selected_candidate_directory": None if selected is None else selected["candidate_directory"],
        "selection_used_external_data": False,
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    return 0.5 * (float(predictions[positive].mean()) + float((~predictions[negative]).mean()))


def paired_bootstrap(labels: np.ndarray, baseline_predictions: np.ndarray, candidate_predictions: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    rng = np.random.default_rng(baseline.SEED + 99)
    n = len(labels)
    ba_diff = np.empty(replicates, dtype=np.float64)
    accuracy_diff = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if len(np.unique(y)) < 2:
            ba_diff[i] = np.nan
        else:
            ba_diff[i] = _balanced_accuracy(y, candidate_predictions[idx]) - _balanced_accuracy(y, baseline_predictions[idx])
        accuracy_diff[i] = (candidate_predictions[idx] == y).mean() - (baseline_predictions[idx] == y).mean()
    return {
        "replicates": replicates,
        "sampling_unit": "image; patient IDs are unavailable in PneumoniaMNIST",
        "balanced_accuracy_difference": {
            "point": _balanced_accuracy(labels, candidate_predictions) - _balanced_accuracy(labels, baseline_predictions),
            "ci95": [float(x) for x in np.nanpercentile(ba_diff, [2.5, 97.5])],
        },
        "accuracy_difference": {
            "point": float((candidate_predictions == labels).mean() - (baseline_predictions == labels).mean()),
            "ci95": [float(x) for x in np.percentile(accuracy_diff, [2.5, 97.5])],
        },
    }


def evaluate_frozen_internal(selected_dir: Path, baseline_out: Path, arrays: dict[str, np.ndarray], run_out: Path, device: torch.device) -> dict:
    candidate_training = json.loads((selected_dir / "training_summary.json").read_text(encoding="utf-8"))
    candidate_threshold = float(candidate_training["source_youden_threshold"])
    candidate_model = load_candidate(selected_dir, device)
    test_loader = baseline.loader(arrays["test_images"], arrays["test_labels"], 256, False, baseline.SEED)
    candidate_logits, candidate_scores, labels = baseline.predict(candidate_model, test_loader, device)
    candidate_metrics = baseline.metrics(labels, candidate_scores, candidate_threshold)
    np.savez_compressed(run_out / "selected_internal_test_predictions.npz", logits=candidate_logits, scores=candidate_scores, labels=labels)
    candidate_xai = baseline.source_explainability(candidate_model, arrays, "test", selected_dir, device)
    candidate_mechanism = gradient_mechanism(candidate_model, arrays, "test", device)

    base_training = json.loads((baseline_out / "training_summary.json").read_text(encoding="utf-8"))
    base_threshold = float(base_training["source_youden_threshold"])
    base_npz = np.load(baseline_out / "internal_test_predictions.npz")
    if not np.array_equal(labels, base_npz["labels"]):
        raise AssertionError("Baseline and candidate test labels are not aligned")
    base_scores = base_npz["scores"]
    base_metrics = baseline.metrics(labels, base_scores, base_threshold)
    bootstrap = paired_bootstrap(labels, base_scores >= base_threshold, candidate_scores >= candidate_threshold)
    result = {
        "status": "selected_candidate_internal_test_completed",
        "selected_lambda": float(candidate_training["lambda_rrr"]),
        "baseline_threshold": base_threshold,
        "candidate_threshold": candidate_threshold,
        "baseline_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_differences_candidate_minus_baseline": {key: candidate_metrics[key] - base_metrics[key] for key in ("accuracy", "balanced_accuracy", "auroc", "average_precision", "sensitivity", "specificity", "brier")},
        "paired_bootstrap": bootstrap,
        "candidate_test_explainability": candidate_xai,
        "candidate_test_mechanism": candidate_mechanism,
        "external_data_evaluated": False,
    }
    baseline.save_json(run_out / "internal_comparison.json", result)
    return result


def plot_candidates(run_out: Path, selection: dict) -> None:
    labels = ["Baseline"] + [f"RRR {x['lambda_rrr']:g}" for x in selection["candidates"]]
    ba = [selection["baseline"]["validation_metrics"]["balanced_accuracy"]] + [x["validation_metrics"]["balanced_accuracy"] for x in selection["candidates"]]
    auc = [selection["baseline"]["validation_metrics"]["auroc"]] + [x["validation_metrics"]["auroc"] for x in selection["candidates"]]
    border = [selection["baseline"]["validation_explainability"]["border_occlusion_probability_abs_change_mean"]] + [x["validation_explainability"]["border_occlusion_probability_abs_change_mean"] for x in selection["candidates"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, values, title in zip(axes, (ba, auc, border), ("Validation balanced accuracy", "Validation AUROC", "Border occlusion |Δp|")):
        ax.bar(labels, values, color=["#777777", "#4C78A8", "#F58518"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_out / "validation_candidate_comparison.png", dpi=180)
    plt.close(fig)


def write_report(run_out: Path, selection: dict, internal: dict | None) -> None:
    lines = [
        "# RRR 源域改进阶段报告",
        "",
        "## 预注册边界",
        "",
        "本阶段只使用 PneumoniaMNIST train/validation 训练和选择候选。E_diag 与 E_final 均未读取、未评价。候选固定为 λ=0.1 与 λ=1.0，种子均为 632901。",
        "",
        "## Validation 选择结果",
        "",
        "| 模型 | BA | AUROC | 边缘遮挡 | 接受 |",
        "|---|---:|---:|---:|:---:|",
    ]
    b = selection["baseline"]
    lines.append(f"| Baseline | {b['validation_metrics']['balanced_accuracy']:.6f} | {b['validation_metrics']['auroc']:.6f} | {b['validation_explainability']['border_occlusion_probability_abs_change_mean']:.6f} | 参考 |")
    for c in selection["candidates"]:
        lines.append(f"| RRR λ={c['lambda_rrr']:g} | {c['validation_metrics']['balanced_accuracy']:.6f} | {c['validation_metrics']['auroc']:.6f} | {c['validation_explainability']['border_occlusion_probability_abs_change_mean']:.6f} | {'是' if c['accepted'] else '否'} |")
    lines.extend(["", f"选择状态：`{selection['status']}`。", ""])
    if internal is None:
        lines.extend(["没有候选满足预注册规则，因此没有打开源域 test，也没有继续外部验证。该负结果必须保留在最终报告中。", ""])
    else:
        lines.extend([
            "## 冻结后的内部 test",
            "",
            f"选中 λ={internal['selected_lambda']:g}。Baseline BA={internal['baseline_metrics']['balanced_accuracy']:.6f}，候选 BA={internal['candidate_metrics']['balanced_accuracy']:.6f}，差值={internal['metric_differences_candidate_minus_baseline']['balanced_accuracy']:+.6f}。",
            f"配对 bootstrap 95% 区间：{internal['paired_bootstrap']['balanced_accuracy_difference']['ci95']}。内部 test 不用于继续调参。",
            "",
            "下一步只有在方法、权重、阈值和主要终点全部冻结后，才运行一次 E_final 配对评价。",
            "",
        ])
    (run_out / "RRR源域改进阶段报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(stage_root: Path, baseline_out: Path) -> dict:
    stage_root.mkdir(parents=True, exist_ok=True)
    source_path = baseline.obtain_source_data(stage_root)
    arrays = baseline.load_arrays(source_path)
    run_out = prepare_protocol(stage_root, baseline_out, source_path, arrays)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_context = load_baseline_context(baseline_out, arrays, device)
    baseline.save_json(run_out / "baseline_validation_context.json", baseline_context)

    candidate_results = []
    for value in CANDIDATE_LAMBDAS:
        candidate_dir = run_out / f"candidate_lambda_{value:g}".replace(".", "p")
        candidate_dir.mkdir()
        train_candidate(arrays, candidate_dir, value)
        evaluated = evaluate_candidate_validation(candidate_dir, arrays, device)
        evaluated["candidate_directory"] = str(candidate_dir)
        candidate_results.append(evaluated)

    selection = select_candidate(baseline_context, candidate_results)
    baseline.save_json(run_out / "candidate_selection.json", selection)
    plot_candidates(run_out, selection)
    internal = None
    if selection["selected_candidate_directory"] is not None:
        selected_dir = Path(selection["selected_candidate_directory"])
        freeze = {
            "status": "candidate_frozen_before_source_test",
            "frozen_utc": baseline.utcnow(),
            "lambda_rrr": selection["selected_lambda"],
            "weights_sha256": baseline.sha256(selected_dir / "best_weights.pt"),
            "threshold": json.loads((selected_dir / "training_summary.json").read_text(encoding="utf-8"))["source_youden_threshold"],
            "selection_file_sha256": baseline.sha256(run_out / "candidate_selection.json"),
            "external_final_opened": False,
        }
        baseline.save_json(run_out / "freeze_before_test.json", freeze)
        internal = evaluate_frozen_internal(selected_dir, baseline_out, arrays, run_out, device)
    final = {
        "status": selection["status"],
        "run_directory": str(run_out),
        "selection": selection,
        "internal_comparison": internal,
        "E_diag_evaluated": False,
        "E_final_evaluated": False,
        "next_step": "paired E_final evaluation once" if internal is not None else "report negative source-validation result and stop improvement search",
    }
    baseline.save_json(run_out / "stage_summary.json", final)
    write_report(run_out, selection, internal)
    baseline.save_json(stage_root / "latest_run.json", {"run_id": run_out.name, "relative_path": str(run_out.relative_to(stage_root)), "status": final["status"], "stage_summary_sha256": baseline.sha256(run_out / "stage_summary.json")})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def smoke_test() -> dict:
    edge = border_mask()
    control = random_same_area_mask()
    assert edge.shape == (1, 1, 28, 28)
    assert int(edge.sum()) == 108
    assert int(control.sum()) == 108
    assert float((edge * control).sum()) == 0.0
    model = baseline.resnet18()
    x = torch.zeros(2, 1, 28, 28, requires_grad=True)
    logits = model(x)
    penalty = rrr_penalty(logits, x, edge)
    assert penalty.ndim == 0 and torch.isfinite(penalty)
    return {"border_pixels": 108, "random_control_pixels": 108, "penalty_finite": True}


if __name__ == "__main__":
    raise SystemExit("Import this module from the stage notebook so the baseline run directory is explicit.")
