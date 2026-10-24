from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import baseline
import external_diagnosis as ext
import rrr_improvement


FINAL_EXPLANATION_SEED = 632912
BOOTSTRAP_SEED = 632913
BOOTSTRAP_REPLICATES = 5000
EXPECTED_SEALED_MANIFEST_SHA256 = "4d484f337c8315eb7a767f3eed2c2fbd94a42fb86f34afebd851742283d22020"
EXPECTED_BASELINE_WEIGHTS_SHA256 = "1e1753581d25a760068ec6215f485df93cbc947eab6f64a6bea2966a7d6cd564"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "3273ac246accc9460a0b889223efac2e44547ad86b885885e7636333412ecc30"
BASELINE_THRESHOLD = 0.9871550798416138
CANDIDATE_THRESHOLD = 0.46137702465057373


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash_rows(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_models(project_root: Path, device: torch.device):
    stage10 = project_root / "md6329_step10_resnet_reproduction"
    stage11 = project_root / "md6329_step11_rrr_source_improvement"
    baseline_latest = json.loads((stage10 / "latest_run.json").read_text(encoding="utf-8"))
    candidate_latest = json.loads((stage11 / "latest_run.json").read_text(encoding="utf-8"))
    baseline_out = stage10 / baseline_latest["relative_path"]
    candidate_run = stage11 / candidate_latest["relative_path"]
    candidate_out = candidate_run / "candidate_lambda_1"
    baseline_weights = baseline_out / "best_weights.pt"
    candidate_weights = candidate_out / "best_weights.pt"
    if baseline.sha256(baseline_weights) != EXPECTED_BASELINE_WEIGHTS_SHA256:
        raise RuntimeError("Frozen baseline weight hash mismatch")
    if baseline.sha256(candidate_weights) != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise RuntimeError("Frozen candidate weight hash mismatch")
    freeze = json.loads((candidate_run / "freeze_before_test.json").read_text(encoding="utf-8"))
    if freeze["weights_sha256"] != EXPECTED_CANDIDATE_WEIGHTS_SHA256 or float(freeze["threshold"]) != CANDIDATE_THRESHOLD:
        raise RuntimeError("Candidate freeze record mismatch")
    baseline_model = baseline.load_selected_model(baseline_out, device)
    candidate_model = rrr_improvement.load_candidate(candidate_out, device)
    return baseline_model, candidate_model, baseline_out, candidate_run


def verify_and_open_final(baseline_out: Path, files: dict[str, Path]):
    manifest_path = baseline_out / "external_split_manifest.json"
    summary_path = baseline_out / "external_split_summary.json"
    if baseline.sha256(manifest_path) != EXPECTED_SEALED_MANIFEST_SHA256:
        raise RuntimeError("Sealed external manifest hash mismatch")
    sealed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = ext.build_records(files["annotations"], files["mapping"])
    diag, final, recomputed = ext.patient_split(records)
    keys = ("patient_id", "subset_img_id", "study_uid", "sop_instance_uid", "class_name")
    final_rows = [{k: r[k] for k in keys} for r in final]
    sealed_rows = sealed_manifest["E_final_sealed"]
    if final_rows != sealed_rows:
        raise AssertionError("Recomputed E_final does not exactly match the sealed manifest")
    if {r["patient_id"] for r in diag} & {r["patient_id"] for r in final}:
        raise AssertionError("Patient leakage between E_diag and E_final")
    if len(final) != 26774 or len({r["patient_id"] for r in final}) != 11009:
        raise AssertionError((len(final), len({r["patient_id"] for r in final})))
    return final, {
        "sealed_manifest_sha256": EXPECTED_SEALED_MANIFEST_SHA256,
        "sealed_rows_sha256": stable_hash_rows(sealed_rows),
        "recomputed_rows_sha256": stable_hash_rows(final_rows),
        "manifest_exact_match": True,
        "patient_overlap_with_E_diag": 0,
        "patients_final": len({r["patient_id"] for r in final}),
        "images_final": len(final),
        "class_counts": sealed_summary["final_class_counts_only_for_split_audit"],
        "opened_utc": utcnow(),
    }


def paired_predict(models: dict[str, torch.nn.Module], zip_path: Path, members: dict[str, str], records: list[dict], device: torch.device, batch_size: int = 256):
    logits = {name: [] for name in models}
    with zipfile.ZipFile(zip_path) as archive, torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            x, _masks, _geo = ext.load_batch(archive, members, batch)
            x = x.to(device)
            for name, model in models.items():
                logits[name].append(model(x).cpu())
            if start == 0 or (start // batch_size + 1) % 10 == 0:
                print(f"paired E_final predictions: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    result = {}
    for name, parts in logits.items():
        logit = torch.cat(parts).numpy()
        score = torch.softmax(torch.from_numpy(logit), 1).numpy()[:, 1]
        result[name] = {"logits": logit, "scores": score}
    return result


def prepare_weighted_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    return order, y, starts


def weighted_auc_prepared(prepared: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray) -> float:
    order, y, starts = prepared
    w = weights[order]
    pos = np.add.reduceat(w * (y == 1), starts)
    neg = np.add.reduceat(w * (y == 0), starts)
    total_pos, total_neg = pos.sum(), neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    before_neg = np.cumsum(neg) - neg
    return float(np.sum(pos * (before_neg + 0.5 * neg)) / (total_pos * total_neg))


def weighted_binary(labels: np.ndarray, scores: np.ndarray, threshold: float, weights: np.ndarray) -> dict:
    pred = scores >= threshold
    pos, neg = labels == 1, labels == 0
    tp = weights[pos & pred].sum(); fn = weights[pos & ~pred].sum()
    tn = weights[neg & ~pred].sum(); fp = weights[neg & pred].sum()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float((tp + tn) / weights.sum()),
        "brier": float(np.average((scores - labels) ** 2, weights=weights)),
    }


def patient_bootstrap_predictions(records: list[dict], base_scores: np.ndarray, cand_scores: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    primary = np.array([r["primary_eligible"] for r in records], dtype=bool)
    labels = np.array([r["target"] for r in records], dtype=np.int64)[primary]
    bscore = base_scores[primary]
    cscore = cand_scores[primary]
    patient_ids = np.array([r["patient_id"] for r in records], dtype=object)[primary]
    patients, inv = np.unique(patient_ids, return_inverse=True)
    n_patients = len(patients)
    base_auc_prepared = prepare_weighted_auc(labels, bscore)
    cand_auc_prepared = prepare_weighted_auc(labels, cscore)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = {k: np.empty(replicates, dtype=np.float64) for k in ("auroc", "balanced_accuracy", "accuracy", "brier")}
    for i in range(replicates):
        counts = rng.multinomial(n_patients, np.full(n_patients, 1 / n_patients))
        weights = counts[inv].astype(np.float64)
        differences["auroc"][i] = weighted_auc_prepared(cand_auc_prepared, weights) - weighted_auc_prepared(base_auc_prepared, weights)
        bm = weighted_binary(labels, bscore, BASELINE_THRESHOLD, weights)
        cm = weighted_binary(labels, cscore, CANDIDATE_THRESHOLD, weights)
        for key in ("balanced_accuracy", "accuracy", "brier"):
            differences[key][i] = cm[key] - bm[key]
    output = {"replicates": replicates, "sampling_unit": "patient", "patients": n_patients}
    for key, values in differences.items():
        output[key + "_difference"] = {"ci95": [float(x) for x in np.quantile(values[np.isfinite(values)], [0.025, 0.975])]}
    return output


def mean_bootstrap(rows: list[dict], keys: list[str], seed: int, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    patients, inv = np.unique(np.array([r["patient_id"] for r in rows], dtype=object), return_inverse=True)
    matrix = np.array([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sims = np.empty((replicates, len(keys)), dtype=np.float64)
    for i in range(replicates):
        counts = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))
        weights = counts[inv].astype(np.float64)
        sims[i] = np.average(matrix, axis=0, weights=weights)
    return {key: {"ci95": [float(x) for x in np.quantile(sims[:, j], [0.025, 0.975])]} for j, key in enumerate(keys)}


def model_occlusion(model: torch.nn.Module, x: torch.Tensor, mask: np.ndarray, outside: list[np.ndarray], device: torch.device) -> dict:
    with torch.no_grad():
        clean = float(torch.softmax(model(x.unsqueeze(0).to(device)), 1)[0, 1])
        result = {"outside_locations": len(outside)}
        for mode in ("mean", "blur"):
            variants = [ext.occluded(x, mask, mode)] + [ext.occluded(x, m, mode) for m in outside]
            scores = torch.softmax(model(torch.stack(variants).to(device)), 1)[:, 1].cpu().numpy()
            inside = clean - float(scores[0])
            outside_drop = clean - float(scores[1:].mean())
            result[f"{mode}_inside_drop"] = inside
            result[f"{mode}_outside_drop"] = outside_drop
            result[f"{mode}_inside_minus_outside"] = inside - outside_drop
    return result


def summarize_localization(rows: list[dict], prefix: str) -> dict:
    return {
        "n": len(rows),
        "area_corrected_energy_mean": float(np.mean([r[f"{prefix}_area_corrected_energy"] for r in rows])),
        "area_corrected_energy_median": float(np.median([r[f"{prefix}_area_corrected_energy"] for r in rows])),
        "pointing_hit_rate": float(np.mean([r[f"{prefix}_pointing_hit"] for r in rows])),
        "zero_cam_rate": float(np.mean([r[f"{prefix}_zero_cam"] for r in rows])),
    }


def paired_explain(models: dict[str, torch.nn.Module], zip_path: Path, members: dict[str, str], records: list[dict], prediction: dict, out: Path, device: torch.device) -> dict:
    positives = [i for i, r in enumerate(records) if r["class_name"] == "Lung Opacity"]
    selected = sorted(positives, key=lambda i: hashlib.sha256(f"{FINAL_EXPLANATION_SEED}|{records[i]['subset_img_id']}".encode()).hexdigest())[:300]
    selected_records = [records[i] for i in selected]
    cams_parts = {name: [] for name in models}
    images_parts, masks_all = [], []
    with zipfile.ZipFile(zip_path) as archive:
        for start in range(0, len(selected_records), 32):
            batch = selected_records[start:start + 32]
            x, masks, _ = ext.load_batch(archive, members, batch)
            for name, model in models.items():
                cams, _ = baseline.gradcam(model, x, device)
                cams_parts[name].append(cams)
            images_parts.append((x[:, 0].numpy() * 0.5 + 0.5).clip(0, 1))
            masks_all.extend(masks)
    cams = {name: np.concatenate(parts) for name, parts in cams_parts.items()}
    images = np.concatenate(images_parts)
    center = ext.center_prior()
    localization_rows = []
    for local_i, global_i in enumerate(selected):
        row = {"subset_img_id": records[global_i]["subset_img_id"], "patient_id": records[global_i]["patient_id"], "box_area_fraction": float(masks_all[local_i].mean())}
        for name in models:
            corrected, pointing, zero = ext.energy_and_pointing(cams[name][local_i], masks_all[local_i])
            row.update({f"{name}_score": float(prediction[name]["scores"][global_i]), f"{name}_area_corrected_energy": corrected, f"{name}_pointing_hit": pointing, f"{name}_zero_cam": zero})
        center_corrected, center_pointing, _ = ext.energy_and_pointing(center, masks_all[local_i])
        row.update({"center_prior_area_corrected_energy": center_corrected, "center_prior_pointing_hit": center_pointing})
        row.update({
            "difference_area_corrected_energy": row["candidate_area_corrected_energy"] - row["baseline_area_corrected_energy"],
            "difference_pointing_hit": row["candidate_pointing_hit"] - row["baseline_pointing_hit"],
            "difference_zero_cam": row["candidate_zero_cam"] - row["baseline_zero_cam"],
        })
        localization_rows.append(row)
    baseline.save_csv(out / "final_localization_cases.csv", localization_rows)

    rng = np.random.default_rng(FINAL_EXPLANATION_SEED)
    occlusion_rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for local_i, global_i in enumerate(selected[:120]):
            x, masks, _ = ext.load_batch(archive, members, [records[global_i]])
            outside = ext.translated_masks(masks[0], 20, rng)
            if not outside:
                continue
            row = {"subset_img_id": records[global_i]["subset_img_id"], "patient_id": records[global_i]["patient_id"], "outside_locations": len(outside)}
            for name, model in models.items():
                values = model_occlusion(model, x[0], masks[0], outside, device)
                for key, value in values.items():
                    if key != "outside_locations":
                        row[f"{name}_{key}"] = value
            for mode in ("mean", "blur"):
                key = f"{mode}_inside_minus_outside"
                row[f"difference_{key}"] = row[f"candidate_{key}"] - row[f"baseline_{key}"]
            occlusion_rows.append(row)
            if (local_i + 1) % 20 == 0:
                print(f"paired E_final occlusion: {local_i + 1}/120", flush=True)
    baseline.save_csv(out / "final_occlusion_cases.csv", occlusion_rows)

    with zipfile.ZipFile(zip_path) as archive:
        x50, _, _ = ext.load_batch(archive, members, selected_records[:50])
    sanity = {}
    for name, model in models.items():
        baseline.seed_all(FINAL_EXPLANATION_SEED + 1)
        randomized = copy.deepcopy(model).cpu()
        randomized.apply(ext.reset_parameters)
        randomized.to(device).eval()
        random_cams, _ = baseline.gradcam(randomized, x50, device)
        similarity = baseline.cosine_rows(cams[name][:50], random_cams)
        sanity[name] = {"n": 50, "trained_vs_fully_randomized_cam_cosine_mean": float(similarity.mean()), "trained_vs_fully_randomized_cam_cosine_median": float(np.median(similarity))}

    loc_base = summarize_localization(localization_rows, "baseline")
    loc_cand = summarize_localization(localization_rows, "candidate")
    loc_diff = {key: loc_cand[key] - loc_base[key] for key in ("area_corrected_energy_mean", "area_corrected_energy_median", "pointing_hit_rate", "zero_cam_rate")}
    center_summary = {"area_corrected_energy_mean": float(np.mean([r["center_prior_area_corrected_energy"] for r in localization_rows])), "pointing_hit_rate": float(np.mean([r["center_prior_pointing_hit"] for r in localization_rows]))}
    occ_summary = {}
    for name in models:
        occ_summary[name] = {"n": len(occlusion_rows)}
        for mode in ("mean", "blur"):
            key = f"{name}_{mode}_inside_minus_outside"
            occ_summary[name][f"{mode}_inside_minus_outside"] = float(np.mean([r[key] for r in occlusion_rows]))
        occ_summary[name]["both_positive_fraction"] = float(np.mean([(r[f"{name}_mean_inside_minus_outside"] > 0 and r[f"{name}_blur_inside_minus_outside"] > 0) for r in occlusion_rows]))
    occ_diff = {key: occ_summary["candidate"][key] - occ_summary["baseline"][key] for key in ("mean_inside_minus_outside", "blur_inside_minus_outside", "both_positive_fraction")}
    loc_boot = mean_bootstrap(localization_rows, ["difference_area_corrected_energy", "difference_pointing_hit", "difference_zero_cam"], BOOTSTRAP_SEED + 1)
    occ_boot = mean_bootstrap(occlusion_rows, ["difference_mean_inside_minus_outside", "difference_blur_inside_minus_outside"], BOOTSTRAP_SEED + 2)

    np.savez_compressed(out / "final_explanations.npz", selected_indices=np.array(selected), images=images, masks=np.stack(masks_all), baseline_cams=cams["baseline"], candidate_cams=cams["candidate"])
    plot_explanation_examples(out, selected, records, images, masks_all, cams, prediction)
    summary = {
        "selection": {"seed": FINAL_EXPLANATION_SEED, "n_localization": len(localization_rows), "n_occlusion": len(occlusion_rows), "n_randomization": 50, "same_cases_for_both_models": True},
        "baseline_localization": loc_base,
        "candidate_localization": loc_cand,
        "candidate_minus_baseline_localization": loc_diff,
        "center_prior": center_summary,
        "localization_patient_bootstrap": loc_boot,
        "matched_occlusion": occ_summary,
        "candidate_minus_baseline_occlusion": occ_diff,
        "occlusion_patient_bootstrap": occ_boot,
        "randomization_sanity": sanity,
        "limits": ["bounding boxes are coarse rather than pixel masks", "28x28 Grad-CAM is spatially coarse", "occlusion can introduce artifacts", "paired interventions strengthen mechanism evidence but do not establish medical causality"],
    }
    baseline.save_json(out / "final_explainability_summary.json", summary)
    return summary


def plot_explanation_examples(out: Path, selected: list[int], records: list[dict], images: np.ndarray, masks: list[np.ndarray], cams: dict, prediction: dict):
    improvement = []
    for local_i, global_i in enumerate(selected):
        b, _, _ = ext.energy_and_pointing(cams["baseline"][local_i], masks[local_i])
        c, _, _ = ext.energy_and_pointing(cams["candidate"][local_i], masks[local_i])
        improvement.append(c - b)
    order = np.argsort(improvement)
    chosen = np.unique(np.linspace(0, len(order) - 1, 8, dtype=int))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 2.2 * len(chosen)))
    for row_i, position in enumerate(chosen):
        local_i = order[position]
        global_i = selected[local_i]
        axes[row_i, 0].imshow(images[local_i], cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].contour(masks[local_i].astype(float), levels=[0.5], colors=["lime"], linewidths=1)
        axes[row_i, 0].set_title(f"Image / box | delta={improvement[local_i]:+.3f}", fontsize=8)
        for col, name in ((1, "baseline"), (2, "candidate")):
            axes[row_i, col].imshow(images[local_i], cmap="gray", vmin=0, vmax=1)
            axes[row_i, col].imshow(cams[name][local_i], cmap="jet", alpha=.42, vmin=0, vmax=1)
            axes[row_i, col].contour(masks[local_i].astype(float), levels=[0.5], colors=["lime"], linewidths=1)
            axes[row_i, col].set_title(f"{name} p={prediction[name]['scores'][global_i]:.2f}", fontsize=8)
        for ax in axes[row_i]:
            ax.axis("off")
    fig.suptitle("E_final paired Grad-CAM: public boxes in green")
    fig.tight_layout()
    fig.savefig(out / "final_paired_gradcam_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(out: Path, metrics: dict, xai: dict):
    b = metrics["baseline"]["primary_normal_vs_lung_opacity_source_youden"]
    c = metrics["candidate"]["primary_normal_vs_lung_opacity_source_youden"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    entries = [
        ("E_final AUROC", [b["auroc"], c["auroc"]]),
        ("E_final balanced accuracy", [b["balanced_accuracy"], c["balanced_accuracy"]]),
        ("Area-corrected CAM energy", [xai["baseline_localization"]["area_corrected_energy_mean"], xai["candidate_localization"]["area_corrected_energy_mean"]]),
    ]
    for ax, (title, values) in zip(axes, entries):
        bars = ax.bar(["Baseline", "RRR 1"], values, color=["#7F7F7F", "#F58518"])
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=.8)
        ax.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value, f"{value:.4f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "final_summary_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(out: Path, summary: dict):
    pm = summary["prediction_metrics"]
    b = pm["baseline"]["primary_normal_vs_lung_opacity_source_youden"]
    c = pm["candidate"]["primary_normal_vs_lung_opacity_source_youden"]
    pd = summary["prediction_differences_candidate_minus_baseline"]
    x = summary["explainability"]
    locd = x["candidate_minus_baseline_localization"]
    occd = x["candidate_minus_baseline_occlusion"]
    lines = [
        "# E_final 配对外部最终评价",
        "",
        f"运行时间：{summary['completed_utc']}。封存清单 SHA256：`{summary['opening_audit']['sealed_manifest_sha256']}`。",
        "",
        "## 预测迁移",
        "",
        "| 模型 | AUROC | Balanced accuracy | Accuracy | Brier |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline | {b['auroc']:.6f} | {b['balanced_accuracy']:.6f} | {b['accuracy']:.6f} | {b['brier']:.6f} |",
        f"| RRR λ=1 | {c['auroc']:.6f} | {c['balanced_accuracy']:.6f} | {c['accuracy']:.6f} | {c['brier']:.6f} |",
        f"| 差值 | {pd['auroc']:+.6f} | {pd['balanced_accuracy']:+.6f} | {pd['accuracy']:+.6f} | {pd['brier']:+.6f} |",
        "",
        "## 病灶区域一致性与干预",
        "",
        f"- 面积校正框内热量差值：{locd['area_corrected_energy_mean']:+.6f}。",
        f"- pointing hit rate 差值：{locd['pointing_hit_rate']:+.6f}。",
        f"- 匹配框内减框外遮挡差值（均值替换）：{occd['mean_inside_minus_outside']:+.6f}。",
        f"- 匹配框内减框外遮挡差值（模糊替换）：{occd['blur_inside_minus_outside']:+.6f}。",
        "",
        "## 解释边界",
        "",
        "同一 E_final 病例、同一公开框和同一遮挡位置用于两个模型，提供配对机制证据。公开框较粗、28×28 Grad-CAM 分辨率低、遮挡会引入伪影，因此不能据此宣称医学因果。E_final 已经打开，本项目不得再根据结果调参。",
    ]
    (out / "E_final最终评价报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(project_root: Path) -> dict:
    project_root = Path(project_root)
    stage12 = project_root / "md6329_step12_external_final"
    stage12.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = stage12 / "runs" / run_id
    out.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline.seed_all(baseline.SEED)
    baseline_model, candidate_model, baseline_out, candidate_run = load_frozen_models(project_root, device)
    plan = {
        "status": "frozen_before_E_final_opening",
        "created_utc": utcnow(),
        "baseline_weights_sha256": EXPECTED_BASELINE_WEIGHTS_SHA256,
        "candidate_weights_sha256": EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        "baseline_threshold": BASELINE_THRESHOLD,
        "candidate_threshold": CANDIDATE_THRESHOLD,
        "sealed_manifest_sha256": EXPECTED_SEALED_MANIFEST_SHA256,
        "primary_prediction_endpoint": "Normal vs Lung Opacity AUROC difference, candidate minus baseline",
        "fixed_threshold_endpoint": "Normal vs Lung Opacity balanced accuracy at each frozen source-validation threshold",
        "primary_explanation_endpoint": "paired difference in area-corrected Grad-CAM energy within public boxes",
        "intervention_endpoint": "paired difference in matched inside-minus-outside occlusion effect",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "sampling_unit": "patient"},
        "no_further_tuning_after_opening": True,
    }
    baseline.save_json(out / "final_evaluation_plan.json", plan)
    files = ext.obtain_rsna(project_root / "md6329_step10_resnet_reproduction")
    members = ext.zip_members(files["images"])
    final_records, opening = verify_and_open_final(baseline_out, files)
    missing = [r["sop_instance_uid"] for r in final_records if r["sop_instance_uid"] not in members]
    if missing:
        raise AssertionError(f"Missing {len(missing)} E_final DICOMs")
    baseline.save_json(out / "E_final_opening_audit.json", opening)
    models = {"baseline": baseline_model, "candidate": candidate_model}
    started = time.time()
    prediction = paired_predict(models, files["images"], members, final_records, device)
    prediction_seconds = time.time() - started
    labels = np.array([r["target"] for r in final_records], dtype=np.int64)
    np.savez_compressed(out / "E_final_paired_predictions.npz", labels=labels, patient_ids=np.array([r["patient_id"] for r in final_records]), subset_img_ids=np.array([r["subset_img_id"] for r in final_records]), baseline_logits=prediction["baseline"]["logits"], baseline_scores=prediction["baseline"]["scores"], candidate_logits=prediction["candidate"]["logits"], candidate_scores=prediction["candidate"]["scores"])
    rows = []
    for i, r in enumerate(final_records):
        rows.append({"patient_id": r["patient_id"], "subset_img_id": r["subset_img_id"], "class_name": r["class_name"], "target": r["target"], "primary_eligible": r["primary_eligible"], "baseline_probability": float(prediction["baseline"]["scores"][i]), "candidate_probability": float(prediction["candidate"]["scores"][i]), "baseline_prediction": int(prediction["baseline"]["scores"][i] >= BASELINE_THRESHOLD), "candidate_prediction": int(prediction["candidate"]["scores"][i] >= CANDIDATE_THRESHOLD)})
    baseline.save_csv(out / "E_final_paired_predictions.csv", rows)
    metrics = {
        "baseline": ext.prediction_metrics(final_records, prediction["baseline"]["scores"], BASELINE_THRESHOLD),
        "candidate": ext.prediction_metrics(final_records, prediction["candidate"]["scores"], CANDIDATE_THRESHOLD),
    }
    bprimary = metrics["baseline"]["primary_normal_vs_lung_opacity_source_youden"]
    cprimary = metrics["candidate"]["primary_normal_vs_lung_opacity_source_youden"]
    differences = {key: cprimary[key] - bprimary[key] for key in ("accuracy", "balanced_accuracy", "auroc", "average_precision", "sensitivity", "specificity", "brier")}
    prediction_bootstrap = patient_bootstrap_predictions(final_records, prediction["baseline"]["scores"], prediction["candidate"]["scores"])
    baseline.save_json(out / "final_prediction_metrics.json", {"metrics": metrics, "candidate_minus_baseline_primary": differences, "patient_bootstrap": prediction_bootstrap})
    xai = paired_explain(models, files["images"], members, final_records, prediction, out, device)
    plot_summary(out, metrics, xai)
    summary = {
        "status": "E_final_paired_evaluation_completed_no_further_tuning",
        "completed_utc": utcnow(),
        "run_id": run_id,
        "run_directory": str(out),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "opening_audit": opening,
        "prediction_seconds": prediction_seconds,
        "prediction_metrics": metrics,
        "prediction_differences_candidate_minus_baseline": differences,
        "prediction_patient_bootstrap": prediction_bootstrap,
        "explainability": xai,
        "baseline_weights_sha256": EXPECTED_BASELINE_WEIGHTS_SHA256,
        "candidate_weights_sha256": EXPECTED_CANDIDATE_WEIGHTS_SHA256,
        "E_diag_reused_for_final_selection": False,
        "E_final_evaluated_once": True,
        "further_tuning_permitted": False,
    }
    baseline.save_json(out / "final_summary.json", summary)
    write_markdown(out, summary)
    latest = {"run_id": run_id, "relative_path": str(out.relative_to(stage12)).replace("\\", "/"), "status": summary["status"], "final_summary_sha256": baseline.sha256(out / "final_summary.json")}
    baseline.save_json(stage12 / "latest_run.json", latest)
    return summary


def smoke_test() -> dict:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    weights = np.ones(4)
    auc = weighted_auc_prepared(prepare_weighted_auc(labels, scores), weights)
    if abs(auc - 1.0) > 1e-12:
        raise AssertionError(auc)
    return {"weighted_auc_perfect": auc, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "sealed_manifest_sha256": EXPECTED_SEALED_MANIFEST_SHA256}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print(json.dumps(run(root), ensure_ascii=False, indent=2))
