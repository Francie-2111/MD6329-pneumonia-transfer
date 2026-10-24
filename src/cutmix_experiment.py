from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import baseline


@dataclass(frozen=True)
class CutMixConfig:
    seed: int = baseline.SEED
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    milestones: tuple[int, int] = (50, 75)
    gamma: float = 0.1
    beta: float = 1.0
    cutmix_probability: float = 0.5
    checkpoint_selection: str = "maximum validation AUROC"


# The official repository uses beta=1.0 and demonstrates probabilities 0.5
# (CIFAR) and 1.0 (ImageNet).  These two settings are fixed before training.
CANDIDATE_PROBABILITIES = (0.5, 1.0)
AUROC_TOLERANCE = 0.01
BOOTSTRAP_REPLICATES = 5000


def rand_bbox(size: torch.Size, lam: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Official CutMix rectangle rule, adapted to NCHW tensors."""
    height, width = int(size[-2]), int(size[-1])
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y = int(rng.integers(0, height))
    center_x = int(rng.integers(0, width))
    y1 = int(np.clip(center_y - cut_h // 2, 0, height))
    y2 = int(np.clip(center_y + cut_h // 2, 0, height))
    x1 = int(np.clip(center_x - cut_w // 2, 0, width))
    x2 = int(np.clip(center_x + cut_w // 2, 0, width))
    return x1, y1, x2, y2


def apply_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    probability: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """Return mixed images, paired labels and the area-corrected lambda."""
    if beta <= 0 or float(rng.random()) >= probability:
        return images, labels, labels, 1.0, False
    lam = float(rng.beta(beta, beta))
    permutation = torch.randperm(images.size(0), device=images.device)
    target_a, target_b = labels, labels[permutation]
    x1, y1, x2, y2 = rand_bbox(images.size(), lam, rng)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    replaced_fraction = ((x2 - x1) * (y2 - y1)) / float(images.size(-1) * images.size(-2))
    corrected_lam = 1.0 - replaced_fraction
    return mixed, target_a, target_b, float(corrected_lam), True


def mixed_loss(criterion: nn.Module, logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(logits, a) + (1.0 - lam) * criterion(logits, b)


def _metrics_from_npz(path: Path, threshold: float) -> dict:
    values = np.load(path)
    return baseline.metrics(values["labels"], values["scores"], threshold)


def load_baseline_context(baseline_out: Path, arrays: dict[str, np.ndarray], device: torch.device) -> dict:
    training = json.loads((baseline_out / "training_summary.json").read_text(encoding="utf-8"))
    threshold = float(training["source_youden_threshold"])
    val_path = baseline_out / "best_validation_predictions.npz"
    model = baseline.load_selected_model(baseline_out, device)
    explain_path = baseline_out / "source_val_explainability.json"
    if explain_path.exists():
        explanations = json.loads(explain_path.read_text(encoding="utf-8"))
    else:
        explanations = baseline.source_explainability(model, arrays, "val", baseline_out, device)
    return {
        "threshold": threshold,
        "validation_metrics_source_threshold": _metrics_from_npz(val_path, threshold),
        "validation_metrics_argmax": _metrics_from_npz(val_path, 0.5),
        "validation_explainability": explanations,
        "weights_sha256": baseline.sha256(baseline_out / "best_weights.pt"),
        "run_dir": str(baseline_out),
    }


def prepare_protocol(stage_root: Path, baseline_out: Path, source_path: Path, arrays: dict[str, np.ndarray]) -> Path:
    run_id = baseline.utcnow().replace(":", "").replace("-", "").replace("+0000", "Z").replace(".", "_")
    out = stage_root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "status": "post_final_cutmix_protocol_frozen_before_training",
        "created_utc": baseline.utcnow(),
        "classification": "post-E_final exploratory follow-up; cannot create a new independent external final test",
        "method": "CutMix following clovaai/CutMix-PyTorch: beta sampled patch replacement and area-weighted labels",
        "reference": {
            "repository": "https://github.com/clovaai/CutMix-PyTorch",
            "paper": "https://arxiv.org/abs/1905.04899",
        },
        "candidate_probabilities": list(CANDIDATE_PROBABILITIES),
        "fixed_beta": 1.0,
        "shared_seed": baseline.SEED,
        "dataset": "PneumoniaMNIST 28 source domain",
        "source_sha256": baseline.sha256(source_path),
        "splits": {s: int(len(arrays[f"{s}_labels"])) for s in ("train", "val", "test")},
        "selection_rule": {
            "checkpoint_within_candidate": "maximum validation AUROC",
            "hard_accuracy_gate": [
                "candidate validation accuracy at its own source-validation Youden threshold >= baseline",
                "candidate validation argmax accuracy >= baseline",
            ],
            "safety_gate": "candidate validation AUROC >= baseline AUROC - 0.01",
            "ranking_among_eligible": "higher validation source-threshold accuracy, then balanced accuracy, AUROC, then lower probability",
            "explainability": "reported separately; dual-objective success requires improvement in a majority of five source proxies",
        },
        "test_policy": "source test opened once only for the selected, frozen validation-eligible candidate",
        "external_policy": "E_final is not reopened and no target-domain result is used for candidate selection",
        "baseline_run_dir": str(baseline_out),
        "baseline_weights_sha256": baseline.sha256(baseline_out / "best_weights.pt"),
        "training": asdict(CutMixConfig()),
    }
    baseline.save_json(out / "plan_before_training.json", protocol)
    baseline.save_json(out / "environment.json", baseline.environment_record())
    return out


def train_candidate(arrays: dict[str, np.ndarray], out: Path, probability: float) -> dict:
    config = CutMixConfig(cutmix_probability=probability)
    baseline.seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = baseline.resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.milestones), gamma=config.gamma)
    criterion = nn.CrossEntropyLoss()
    train_loader = baseline.loader(arrays["train_images"], arrays["train_labels"], config.batch_size, True, config.seed)
    val_loader = baseline.loader(arrays["val_images"], arrays["val_labels"], config.batch_size, False, config.seed)
    rows: list[dict] = []
    best_auc, best_epoch = -np.inf, None
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = total_correct = total_count = mixed_batches = 0
        lambda_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x_mix, a, b, lam, was_mixed = apply_cutmix(x, y, config.beta, config.cutmix_probability, rng)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_mix)
            loss = mixed_loss(criterion, logits, a, b, lam)
            loss.backward()
            optimizer.step()
            n = len(y)
            total_loss += float(loss.detach()) * n
            # Weighted label agreement is a diagnostic only, not ordinary accuracy.
            pred = logits.argmax(1)
            total_correct += float((lam * (pred == a).float() + (1.0 - lam) * (pred == b).float()).sum())
            total_count += n
            mixed_batches += int(was_mixed)
            lambda_sum += lam
        val_logits, val_scores, val_labels = baseline.predict(model, val_loader, device)
        val_metrics = baseline.metrics(val_labels, val_scores, 0.5)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_cutmix_loss": total_loss / total_count,
            "train_weighted_label_agreement": total_correct / total_count,
            "mixed_batches": mixed_batches,
            "mean_batch_lambda": lambda_sum / len(train_loader),
            "val_auroc": val_metrics["auroc"],
            "val_accuracy_argmax": val_metrics["accuracy"],
            "elapsed_seconds": time.time() - started,
        }
        rows.append(row)
        if val_metrics["auroc"] > best_auc:
            best_auc, best_epoch = val_metrics["auroc"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_auroc": best_auc, "config": asdict(config)}, out / "best_weights.pt")
            np.savez_compressed(out / "best_validation_predictions.npz", logits=val_logits, scores=val_scores, labels=val_labels)
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"p={probability:g} epoch={epoch:03d} loss={row['train_cutmix_loss']:.4f} val_auc={row['val_auroc']:.4f} val_acc={row['val_accuracy_argmax']:.4f}", flush=True)
    baseline.save_csv(out / "learning_curve.csv", rows)
    values = np.load(out / "best_validation_predictions.npz")
    threshold = baseline.threshold_youden(values["labels"], values["scores"])
    summary = {
        "cutmix_probability": probability,
        "beta": config.beta,
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


def proxy_improvements(candidate: dict, reference: dict) -> dict:
    c, b = candidate, reference
    checks = {
        "zero_cam_rate_lower": c["zero_cam_rate"] < b["zero_cam_rate"],
        "brightness_cam_cosine_higher": c["brightness_1.10_cam_cosine_mean"] > b["brightness_1.10_cam_cosine_mean"],
        "contrast_cam_cosine_higher": c["contrast_1.10_cam_cosine_mean"] > b["contrast_1.10_cam_cosine_mean"],
        "brightness_probability_change_lower": c["brightness_probability_abs_change_mean"] < b["brightness_probability_abs_change_mean"],
        "contrast_probability_change_lower": c["contrast_probability_abs_change_mean"] < b["contrast_probability_abs_change_mean"],
    }
    return {"checks": checks, "improved_count": int(sum(checks.values())), "total": len(checks), "majority_improved": sum(checks.values()) >= 3}


def evaluate_candidate_validation(candidate_dir: Path, arrays: dict[str, np.ndarray], device: torch.device, baseline_context: dict) -> dict:
    training = json.loads((candidate_dir / "training_summary.json").read_text(encoding="utf-8"))
    threshold = float(training["source_youden_threshold"])
    model = load_candidate(candidate_dir, device)
    xai = baseline.source_explainability(model, arrays, "val", candidate_dir, device)
    result = {
        "cutmix_probability": float(training["cutmix_probability"]),
        "threshold": threshold,
        "validation_metrics_source_threshold": _metrics_from_npz(candidate_dir / "best_validation_predictions.npz", threshold),
        "validation_metrics_argmax": _metrics_from_npz(candidate_dir / "best_validation_predictions.npz", 0.5),
        "validation_explainability": xai,
        "explainability_proxy_comparison": proxy_improvements(xai, baseline_context["validation_explainability"]),
    }
    baseline.save_json(candidate_dir / "validation_evaluation.json", result)
    return result


def select_candidate(baseline_context: dict, candidates: list[dict]) -> dict:
    b_y = baseline_context["validation_metrics_source_threshold"]
    b_a = baseline_context["validation_metrics_argmax"]
    rows = []
    for candidate in candidates:
        c_y = candidate["validation_metrics_source_threshold"]
        c_a = candidate["validation_metrics_argmax"]
        checks = {
            "source_threshold_accuracy_not_lower": c_y["accuracy"] >= b_y["accuracy"],
            "argmax_accuracy_not_lower": c_a["accuracy"] >= b_a["accuracy"],
            "auroc_within_tolerance": c_y["auroc"] >= b_y["auroc"] - AUROC_TOLERANCE,
        }
        rows.append({**candidate, "acceptance_checks": checks, "validation_eligible": all(checks.values())})
    eligible = [r for r in rows if r["validation_eligible"]]
    selected = None
    if eligible:
        eligible.sort(key=lambda r: (
            -r["validation_metrics_source_threshold"]["accuracy"],
            -r["validation_metrics_source_threshold"]["balanced_accuracy"],
            -r["validation_metrics_source_threshold"]["auroc"],
            r["cutmix_probability"],
        ))
        selected = eligible[0]
    return {
        "status": "candidate_selected_and_frozen" if selected else "no_cutmix_candidate_met_accuracy_gate",
        "baseline": baseline_context,
        "candidates": rows,
        "selected_probability": None if selected is None else selected["cutmix_probability"],
        "selected_candidate_directory": None if selected is None else selected["candidate_directory"],
        "selection_used_test_or_external_data": False,
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    return 0.5 * (float(predictions[labels == 1].mean()) + float((~predictions[labels == 0]).mean()))


def paired_bootstrap(labels: np.ndarray, baseline_predictions: np.ndarray, candidate_predictions: np.ndarray) -> dict:
    rng = np.random.default_rng(baseline.SEED + 130)
    n = len(labels)
    accuracy_delta = np.empty(BOOTSTRAP_REPLICATES)
    ba_delta = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        accuracy_delta[i] = (candidate_predictions[idx] == y).mean() - (baseline_predictions[idx] == y).mean()
        ba_delta[i] = np.nan if len(np.unique(y)) < 2 else _balanced_accuracy(y, candidate_predictions[idx]) - _balanced_accuracy(y, baseline_predictions[idx])
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "sampling_unit": "image; patient identifiers unavailable in source data",
        "accuracy_difference_ci95": [float(v) for v in np.percentile(accuracy_delta, [2.5, 97.5])],
        "balanced_accuracy_difference_ci95": [float(v) for v in np.nanpercentile(ba_delta, [2.5, 97.5])],
        "fixed_model_sampling_uncertainty_only": True,
    }


def evaluate_internal(selected_dir: Path, baseline_out: Path, arrays: dict[str, np.ndarray], run_out: Path, device: torch.device) -> dict:
    candidate_training = json.loads((selected_dir / "training_summary.json").read_text(encoding="utf-8"))
    candidate_threshold = float(candidate_training["source_youden_threshold"])
    model = load_candidate(selected_dir, device)
    test_loader = baseline.loader(arrays["test_images"], arrays["test_labels"], 256, False, baseline.SEED)
    logits, scores, labels = baseline.predict(model, test_loader, device)
    np.savez_compressed(run_out / "selected_internal_test_predictions.npz", logits=logits, scores=scores, labels=labels)
    candidate_youden = baseline.metrics(labels, scores, candidate_threshold)
    candidate_argmax = baseline.metrics(labels, scores, 0.5)
    candidate_xai = baseline.source_explainability(model, arrays, "test", selected_dir, device)

    base_train = json.loads((baseline_out / "training_summary.json").read_text(encoding="utf-8"))
    base_threshold = float(base_train["source_youden_threshold"])
    base_npz = np.load(baseline_out / "internal_test_predictions.npz")
    if not np.array_equal(labels, base_npz["labels"]):
        raise AssertionError("Baseline and CutMix source-test labels differ")
    base_scores = base_npz["scores"]
    base_youden = baseline.metrics(labels, base_scores, base_threshold)
    base_argmax = baseline.metrics(labels, base_scores, 0.5)
    base_xai_path = baseline_out / "source_test_explainability.json"
    base_xai = json.loads(base_xai_path.read_text(encoding="utf-8"))
    test_proxy = proxy_improvements(candidate_xai, base_xai)
    accuracy_guardrail = {
        "source_threshold_accuracy_not_lower": candidate_youden["accuracy"] >= base_youden["accuracy"],
        "argmax_accuracy_not_lower": candidate_argmax["accuracy"] >= base_argmax["accuracy"],
    }
    final_adoption = all(accuracy_guardrail.values())
    result = {
        "status": "cutmix_adoptable_on_internal_accuracy_guardrail" if final_adoption else "cutmix_rejected_internal_accuracy_below_baseline",
        "selected_probability": float(candidate_training["cutmix_probability"]),
        "baseline_source_threshold_metrics": base_youden,
        "candidate_source_threshold_metrics": candidate_youden,
        "baseline_argmax_metrics": base_argmax,
        "candidate_argmax_metrics": candidate_argmax,
        "source_threshold_differences_candidate_minus_baseline": {k: candidate_youden[k] - base_youden[k] for k in ("accuracy", "balanced_accuracy", "auroc", "average_precision", "sensitivity", "specificity", "brier")},
        "argmax_accuracy_difference": candidate_argmax["accuracy"] - base_argmax["accuracy"],
        "accuracy_guardrail": accuracy_guardrail,
        "candidate_test_explainability": candidate_xai,
        "test_explainability_proxy_comparison": test_proxy,
        "dual_objective_success_on_source_test": bool(final_adoption and test_proxy["majority_improved"]),
        "paired_bootstrap": paired_bootstrap(labels, base_scores >= base_threshold, scores >= candidate_threshold),
        "external_final_reopened": False,
    }
    baseline.save_json(run_out / "internal_comparison.json", result)
    return result


def plot_results(run_out: Path, selection: dict, internal: dict | None) -> None:
    labels = ["Baseline"] + [f"CutMix p={x['cutmix_probability']:g}" for x in selection["candidates"]]
    val_acc = [selection["baseline"]["validation_metrics_source_threshold"]["accuracy"]] + [x["validation_metrics_source_threshold"]["accuracy"] for x in selection["candidates"]]
    val_auc = [selection["baseline"]["validation_metrics_source_threshold"]["auroc"]] + [x["validation_metrics_source_threshold"]["auroc"] for x in selection["candidates"]]
    proxy_count = [0] + [x["explainability_proxy_comparison"]["improved_count"] for x in selection["candidates"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, values, title, fmt in zip(axes, (val_acc, val_auc, proxy_count), ("Validation accuracy", "Validation AUROC", "Improved XAI proxies (of 5)"), (".4f", ".4f", ".0f")):
        ax.bar(labels, values, color=["#777777", "#4C78A8", "#F58518"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, format(value, fmt), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_out / "cutmix_validation_comparison.png", dpi=180)
    plt.close(fig)
    if internal is not None:
        measures = ["Accuracy", "Balanced accuracy", "AUROC"]
        b = internal["baseline_source_threshold_metrics"]
        c = internal["candidate_source_threshold_metrics"]
        x = np.arange(len(measures))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.18, [b["accuracy"], b["balanced_accuracy"], b["auroc"]], width=0.36, label="Baseline")
        ax.bar(x + 0.18, [c["accuracy"], c["balanced_accuracy"], c["auroc"]], width=0.36, label="CutMix")
        ax.set_xticks(x, measures)
        ax.set_ylim(0.75, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(run_out / "cutmix_internal_test_comparison.png", dpi=180)
        plt.close(fig)


def write_report(run_out: Path, selection: dict, internal: dict | None) -> None:
    b = selection["baseline"]
    lines = [
        "# CutMix 探索性实验记录",
        "",
        "本实验在 E_final 已经打开之后提出，因此只能作为 post-hoc 探索。候选训练与选择只使用源域 train/validation；E_final 不会重新打开，也不能把本实验称为新的独立外部验证。",
        "",
        "## Validation 选择",
        "",
        "| 模型 | 源阈值 accuracy | argmax accuracy | AUROC | 改善的解释代理/5 | 通过 accuracy gate |",
        "|---|---:|---:|---:|---:|:---:|",
        f"| Baseline | {b['validation_metrics_source_threshold']['accuracy']:.6f} | {b['validation_metrics_argmax']['accuracy']:.6f} | {b['validation_metrics_source_threshold']['auroc']:.6f} | — | 参考 |",
    ]
    for c in selection["candidates"]:
        lines.append(f"| CutMix p={c['cutmix_probability']:g} | {c['validation_metrics_source_threshold']['accuracy']:.6f} | {c['validation_metrics_argmax']['accuracy']:.6f} | {c['validation_metrics_source_threshold']['auroc']:.6f} | {c['explainability_proxy_comparison']['improved_count']} | {'是' if c['validation_eligible'] else '否'} |")
    lines.extend(["", f"选择状态：`{selection['status']}`。", ""])
    if internal is None:
        lines.append("没有候选达到预注册的 validation accuracy gate，源域 test 未打开，CutMix 不进入采用名单。")
    else:
        bm, cm = internal["baseline_source_threshold_metrics"], internal["candidate_source_threshold_metrics"]
        lines.extend([
            "## 冻结候选的内部 test",
            "",
            f"Baseline/CutMix 源阈值 accuracy：{bm['accuracy']:.6f}/{cm['accuracy']:.6f}；balanced accuracy：{bm['balanced_accuracy']:.6f}/{cm['balanced_accuracy']:.6f}；AUROC：{bm['auroc']:.6f}/{cm['auroc']:.6f}。",
            f"argmax accuracy 差值：{internal['argmax_accuracy_difference']:+.6f}。解释性代理改善 {internal['test_explainability_proxy_comparison']['improved_count']}/5。",
            f"最终判定：`{internal['status']}`；源域双目标是否同时满足：`{internal['dual_objective_success_on_source_test']}`。",
            "",
            "解释性结论仅涉及 Grad-CAM 稳定性与预测对轻微扰动的响应。PneumoniaMNIST 无病灶框，因此不能由本阶段证明模型更关注病灶，更不能声称医学因果。",
        ])
    (run_out / "CutMix探索性实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(stage_root: Path, baseline_out: Path) -> dict:
    stage_root.mkdir(parents=True, exist_ok=True)
    source_path = baseline.obtain_source_data(stage_root)
    arrays = baseline.load_arrays(source_path)
    run_out = prepare_protocol(stage_root, baseline_out, source_path, arrays)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_context = load_baseline_context(baseline_out, arrays, device)
    baseline.save_json(run_out / "baseline_validation_context.json", baseline_context)
    candidates = []
    for probability in CANDIDATE_PROBABILITIES:
        candidate_dir = run_out / f"candidate_p_{probability:g}".replace(".", "p")
        candidate_dir.mkdir()
        train_candidate(arrays, candidate_dir, probability)
        evaluated = evaluate_candidate_validation(candidate_dir, arrays, device, baseline_context)
        evaluated["candidate_directory"] = str(candidate_dir)
        candidates.append(evaluated)
    selection = select_candidate(baseline_context, candidates)
    baseline.save_json(run_out / "candidate_selection.json", selection)
    internal = None
    if selection["selected_candidate_directory"] is not None:
        selected_dir = Path(selection["selected_candidate_directory"])
        freeze = {
            "status": "cutmix_candidate_frozen_before_source_test",
            "frozen_utc": baseline.utcnow(),
            "probability": selection["selected_probability"],
            "weights_sha256": baseline.sha256(selected_dir / "best_weights.pt"),
            "threshold": json.loads((selected_dir / "training_summary.json").read_text(encoding="utf-8"))["source_youden_threshold"],
            "E_final_reopen_permitted": False,
        }
        baseline.save_json(run_out / "freeze_before_test.json", freeze)
        internal = evaluate_internal(selected_dir, baseline_out, arrays, run_out, device)
    plot_results(run_out, selection, internal)
    final = {
        "status": selection["status"] if internal is None else internal["status"],
        "run_directory": str(run_out),
        "selection": selection,
        "internal_comparison": internal,
        "E_diag_evaluated": False,
        "E_final_reopened": False,
        "study_classification": "post-final exploratory",
    }
    baseline.save_json(run_out / "stage_summary.json", final)
    write_report(run_out, selection, internal)
    baseline.save_json(stage_root / "latest_run.json", {"run_id": run_out.name, "relative_path": str(run_out.relative_to(stage_root)), "status": final["status"], "stage_summary_sha256": baseline.sha256(run_out / "stage_summary.json")})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return final


def smoke_test() -> dict:
    rng = np.random.default_rng(7)
    x = torch.stack([torch.zeros(1, 28, 28), torch.ones(1, 28, 28)])
    y = torch.tensor([0, 1])
    mixed, a, b, lam, used = apply_cutmix(x, y, 1.0, 1.0, rng)
    assert mixed.shape == x.shape and a.shape == b.shape == y.shape
    assert used and 0.0 <= lam <= 1.0
    replaced = float((mixed[0] != x[0]).float().mean())
    assert abs((1.0 - lam) - replaced) < 1e-8
    return {"shape": list(mixed.shape), "lambda": lam, "replaced_fraction": replaced, "area_correction_exact": True}


if __name__ == "__main__":
    raise SystemExit("Import this module from the stage notebook so the frozen baseline directory is explicit.")
