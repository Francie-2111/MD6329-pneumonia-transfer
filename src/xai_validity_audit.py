"""Matched explanation validity checks for the four frozen B64 models.

This module performs inference only. It does not train, select, or tune a model.
It adds (1) a shared-reference/cascading parameter-randomization check,
(2) margin-target Grad-CAM sensitivity, and (3) logit-margin occlusion to detect
softmax saturation. Existing probability and Grad-CAM results remain unchanged.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import b64_augmix as model_code
import b64_external_audit as audit
import external_diagnosis as ext

SEED = 632930
N_CAM = 50
N_DRAWS = 5000
NAMES = ("plain", "aug1", "ce3", "augmix")
LABELS = {"plain": "Plain", "aug1": "Aug1", "ce3": "CE3", "augmix": "AC-JSD"}
HASHES = {
    "plain": "c51030d6e2646dbd4b5b1b4132528590dc85a609f79eb61ed29c4acfd90fdd2f",
    "aug1": "54c18d60e948b90153c6767046b03ffa4916d8be9603401bb9787fcd2a3b985c",
    "ce3": "f3a7650136ac9d379dc0d3824c6871d038b632f4a82526b9080b00a01f36ef97",
    "augmix": "cf3a97804e8614b719367a8cdef4b23d9da5adbb2d48201a4dd6da238a4cf873",
}
STAGES = {
    "trained": (),
    "linear": ("linear",),
    "layer4_linear": ("layer4", "linear"),
    "layer3_to_linear": ("layer3", "layer4", "linear"),
    "layer2_to_linear": ("layer2", "layer3", "layer4", "linear"),
    "layer1_to_linear": ("layer1", "layer2", "layer3", "layer4", "linear"),
    "all": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "linear"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(model: torch.nn.Module, roots: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        if any(key == root or key.startswith(root + ".") for root in roots):
            value = state[key].detach().cpu().contiguous().numpy()
            h.update(key.encode())
            h.update(value.tobytes())
    return h.hexdigest()


def reset_roots(model: torch.nn.Module, roots: tuple[str, ...], seed: int) -> None:
    torch.manual_seed(seed)
    for root in roots:
        getattr(model, root).apply(lambda module: module.reset_parameters()
                                  if hasattr(module, "reset_parameters") else None)


def gradcam(model: torch.nn.Module, images: torch.Tensor, target: str,
            device: torch.device, batch_size: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM for pneumonia logit z1 or binary margin z1-z0."""
    cams, logits_all = [], []
    model.eval()
    for start in range(0, len(images), batch_size):
        saved = {}

        def hook(_module, _inputs, output):
            output.retain_grad()
            saved["activation"] = output

        handle = model.layer4.register_forward_hook(hook)
        try:
            x = images[start:start + batch_size].to(device).requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x)
            score = logits[:, 1] if target == "pneumonia_logit" else logits[:, 1] - logits[:, 0]
            score.sum().backward()
            activation = saved["activation"]
            weights = activation.grad.mean((2, 3), keepdim=True)
            cam = F.relu((weights * activation).sum(1, keepdim=True))
            cam = F.interpolate(cam, images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            maximum = cam.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
            cams.append((cam / maximum).detach().cpu().numpy())
            logits_all.append(logits.detach().cpu().numpy())
        finally:
            handle.remove()
    return np.concatenate(cams), np.concatenate(logits_all)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def energy_pointing(cam: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    total = float(cam.sum())
    if total <= 1e-12:
        return -float(mask.mean()), 0
    return float(cam[mask].sum() / total - mask.mean()), int(mask.flat[int(cam.argmax())])


def patient_bootstrap(values: np.ndarray, patient_ids: np.ndarray, seed: int,
                      draws: int = N_DRAWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point, lower, upper for case x endpoint values."""
    _, inverse = np.unique(patient_ids, return_inverse=True)
    n_patients = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    sampled = np.empty((draws, values.shape[1]), float)
    for draw in range(draws):
        weights = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)[inverse]
        sampled[draw] = (values * weights[:, None]).sum(0) / weights.sum()
    lo, hi = np.quantile(sampled, [.025, .975], axis=0)
    return values.mean(0), lo, hi


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_models(checkpoints: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    models = {}
    for name in NAMES:
        path = checkpoints / f"{name}.pt"
        assert sha256(path) == HASHES[name]
        model = model_code.ResNet18().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state"])
        model.eval(); models[name] = model
    return models


def run_randomization(models, images, masks, patient_ids, out, device):
    original = {}
    fresh_logits = {}
    for name, model in models.items():
        original[name], fresh_logits[name] = gradcam(model, images, "pneumonia_logit", device)

    # One identical fully randomized network is the reference for every trained model.
    torch.manual_seed(SEED)
    shared_random = model_code.ResNet18().to(device).eval()
    shared_cam, _ = gradcam(shared_random, images, "pneumonia_logit", device)

    case_rows, stage_hashes = [], {}
    progressive = {name: {} for name in NAMES}
    for name in NAMES:
        progressive[name]["trained"] = np.ones(len(images))
        progressive[name]["shared_fully_random"] = cosine_rows(original[name], shared_cam)

    for stage_index, (stage, roots) in enumerate(STAGES.items()):
        if stage == "trained":
            continue
        hashes = []
        for name in NAMES:
            candidate = copy.deepcopy(models[name]).cpu()
            reset_roots(candidate, roots, SEED + 100 + stage_index)
            hashes.append(state_hash(candidate, roots))
            candidate.to(device).eval()
            randomized_cam, _ = gradcam(candidate, images, "pneumonia_logit", device)
            progressive[name][stage] = cosine_rows(original[name], randomized_cam)
            del candidate
        assert len(set(hashes)) == 1, f"Randomized parameters differ across models at {stage}"
        stage_hashes[stage] = hashes[0]

    for i in range(len(images)):
        row = {"case_index": i, "patient_id": str(patient_ids[i])}
        for name in NAMES:
            for stage, values in progressive[name].items():
                row[f"{name}_{stage}_cosine"] = float(values[i])
        case_rows.append(row)
    save_csv(out / "matched_parameter_randomization_cases.csv", case_rows)

    interval_rows = []
    for name in NAMES:
        stages = list(progressive[name])
        values = np.column_stack([progressive[name][stage] for stage in stages])
        point, lo, hi = patient_bootstrap(values, patient_ids, SEED + 1)
        for j, stage in enumerate(stages):
            interval_rows.append({"model": name, "stage": stage, "mean_cosine": point[j],
                                  "ci_low": lo[j], "ci_high": hi[j], "n_images": len(images),
                                  "n_patients": len(np.unique(patient_ids)), "draws": N_DRAWS})
    save_csv(out / "matched_parameter_randomization_intervals.csv", interval_rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    display_stages = [s for s in STAGES if s != "trained"]
    for name in NAMES:
        mean = [np.mean(progressive[name][s]) for s in display_stages]
        ax.plot(range(len(display_stages)), mean, marker="o", label=LABELS[name])
    ax.set_xticks(range(len(display_stages)), ["Head", "L4+", "L3+", "L2+", "L1+", "All"])
    ax.set_ylabel("CAM cosine vs trained model"); ax.set_xlabel("Cumulative randomized modules")
    ax.set_ylim(-.05, 1.02); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "matched_parameter_randomization.pdf"); plt.close(fig)
    return original, fresh_logits, stage_hashes, interval_rows


def verify_stage_hashes(models) -> dict[str, str]:
    """Cheaply reproduce the matched-random-parameter identity assertion."""
    stage_hashes = {}
    for stage_index, (stage, roots) in enumerate(STAGES.items()):
        if stage == "trained":
            continue
        hashes = []
        for name in NAMES:
            candidate = copy.deepcopy(models[name]).cpu()
            reset_roots(candidate, roots, SEED + 100 + stage_index)
            hashes.append(state_hash(candidate, roots))
        assert len(set(hashes)) == 1, f"Randomized parameters differ across models at {stage}"
        stage_hashes[stage] = hashes[0]
    return stage_hashes


def run_target_sensitivity(models, images, masks, patient_ids, original, fresh_logits, out, device):
    rows = []
    for name in NAMES:
        margin_cam, margin_logits = gradcam(models[name], images, "logit_margin", device)
        np.testing.assert_allclose(margin_logits, fresh_logits[name], atol=2e-6, rtol=2e-6)
        cos = cosine_rows(original[name], margin_cam)
        for i in range(len(images)):
            z_energy, z_hit = energy_pointing(original[name][i], masks[i])
            m_energy, m_hit = energy_pointing(margin_cam[i], masks[i])
            rows.append({"case_index": i, "patient_id": str(patient_ids[i]), "model": name,
                         "target_cam_cosine": float(cos[i]), "pneumonia_logit_energy": z_energy,
                         "margin_energy": m_energy, "margin_minus_logit_energy": m_energy - z_energy,
                         "pneumonia_logit_pointing": z_hit, "margin_pointing": m_hit,
                         "margin_minus_logit_pointing": m_hit - z_hit})
    save_csv(out / "gradcam_target_sensitivity_cases.csv", rows)
    interval_rows = []
    for name in NAMES:
        model_rows = [r for r in rows if r["model"] == name]
        endpoints = ["target_cam_cosine", "margin_minus_logit_energy", "margin_minus_logit_pointing"]
        values = np.array([[r[k] for k in endpoints] for r in model_rows], float)
        point, lo, hi = patient_bootstrap(values, patient_ids, SEED + 2)
        for j, endpoint in enumerate(endpoints):
            interval_rows.append({"model": name, "endpoint": endpoint, "estimate": point[j],
                                  "ci_low": lo[j], "ci_high": hi[j], "n_images": len(images),
                                  "n_patients": len(np.unique(patient_ids)), "draws": N_DRAWS})
    save_csv(out / "gradcam_target_sensitivity_intervals.csv", interval_rows)
    return rows, interval_rows


def run_margin_occlusion(models, images, masks, patient_ids, selected_indices,
                         predictions, old_csv: Path, out: Path, device: torch.device):
    with old_csv.open(encoding="utf-8-sig") as f:
        old_rows = {r["subset_img_id"]: r for r in csv.DictReader(f)}
    rng = np.random.default_rng(audit.EXPLANATION_SEED)
    outside_by_case = {}
    for local_i in range(120):
        outside = audit.translated_masks64(masks[local_i], 20, rng)
        if outside:
            outside_by_case[local_i] = outside
    assert len(outside_by_case) == len(old_rows) == 89

    rows = []
    for local_i, outside in outside_by_case.items():
        uid = str(predictions["subset_img_ids"][selected_indices[local_i]])
        row = {"case_index": local_i, "subset_img_id": uid, "patient_id": str(patient_ids[local_i]),
               "box_area_fraction": float(masks[local_i].mean()), "outside_locations": len(outside)}
        x = images[local_i]
        for name in NAMES:
            model = models[name]
            with torch.no_grad():
                clean_logits = model(x[None].to(device)).cpu().numpy()[0]
            global_i = int(selected_indices[local_i])
            np.testing.assert_allclose(clean_logits, predictions[f"{name}_logits"][global_i], atol=2e-5, rtol=2e-5)
            clean_probability = float(torch.softmax(torch.from_numpy(clean_logits), 0)[1])
            clean_margin = float(clean_logits[1] - clean_logits[0])
            row[f"{name}_clean_probability"] = clean_probability
            row[f"{name}_clean_margin"] = clean_margin
            for mode in ("mean", "blur"):
                variants = [ext.occluded(x, masks[local_i], mode)] + [ext.occluded(x, m, mode) for m in outside]
                with torch.no_grad():
                    logits = model(torch.stack(variants).to(device)).cpu()
                probability = torch.softmax(logits, 1)[:, 1].numpy()
                margin = (logits[:, 1] - logits[:, 0]).numpy()
                for kind, values, clean in (("probability", probability, clean_probability),
                                             ("margin", margin, clean_margin)):
                    inside = clean - float(values[0]); outside_drop = clean - float(values[1:].mean())
                    row[f"{name}_{mode}_{kind}_inside_drop"] = inside
                    row[f"{name}_{mode}_{kind}_outside_drop"] = outside_drop
                    row[f"{name}_{mode}_{kind}_inside_minus_outside"] = inside - outside_drop
                old = old_rows[uid]
                # The archived values were produced on CUDA; CPU re-inference differs only at
                # single-precision rounding scale. This tolerance is far below any reported digit.
                np.testing.assert_allclose(row[f"{name}_{mode}_probability_inside_drop"],
                                           float(old[f"{name}_{mode}_inside_drop"]), atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(row[f"{name}_{mode}_probability_outside_drop"],
                                           float(old[f"{name}_{mode}_outside_drop"]), atol=1e-5, rtol=1e-5)
        rows.append(row)
    save_csv(out / "margin_occlusion_cases.csv", rows)

    endpoints = [f"{mode}_{kind}_inside_minus_outside" for mode in ("mean", "blur")
                 for kind in ("probability", "margin")]
    interval_rows = []
    ids = np.array([r["patient_id"] for r in rows])
    for name in NAMES:
        values = np.array([[r[f"{name}_{endpoint}"] for endpoint in endpoints] for r in rows])
        point, lo, hi = patient_bootstrap(values, ids, SEED + 3)
        for j, endpoint in enumerate(endpoints):
            interval_rows.append({"comparison": name, "endpoint": endpoint, "estimate": point[j],
                                  "ci_low": lo[j], "ci_high": hi[j], "n_images": len(rows),
                                  "n_patients": len(np.unique(ids)), "draws": N_DRAWS})
    for a, b in (("augmix", "plain"), ("aug1", "plain"), ("ce3", "plain"),
                 ("augmix", "ce3"), ("ce3", "aug1")):
        values = np.array([[r[f"{a}_{endpoint}"] - r[f"{b}_{endpoint}"] for endpoint in endpoints]
                           for r in rows])
        point, lo, hi = patient_bootstrap(values, ids, SEED + 3)
        for j, endpoint in enumerate(endpoints):
            interval_rows.append({"comparison": f"{a} minus {b}", "endpoint": endpoint,
                                  "estimate": point[j], "ci_low": lo[j], "ci_high": hi[j],
                                  "n_images": len(rows), "n_patients": len(np.unique(ids)), "draws": N_DRAWS})
    save_csv(out / "margin_occlusion_intervals.csv", interval_rows)
    return rows, interval_rows


def run(root: Path, out: Path) -> dict:
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(root / "checkpoints", device)
    saved = np.load(root / "results/E_final_B64_explanations.npz")
    predictions = np.load(root / "results/four_way_predictions.npz")
    images_all = torch.from_numpy(saved["images"]).float().unsqueeze(1).mul(2).sub(1)
    masks_all = saved["masks"].astype(bool)
    selected_indices = saved["selected_indices"].astype(int)
    selected_patients = predictions["patient_ids"][selected_indices].astype(str)
    images, masks, patient_ids = images_all[:N_CAM], masks_all[:N_CAM], selected_patients[:N_CAM]

    random_file = out / "matched_parameter_randomization_intervals.csv"
    target_file = out / "gradcam_target_sensitivity_intervals.csv"
    if random_file.exists() and target_file.exists():
        print("Reusing completed matched randomization and target-sensitivity artifacts.", flush=True)
        stage_hashes = verify_stage_hashes(models)
    else:
        original, logits, stage_hashes, random_intervals = run_randomization(
            models, images, masks, patient_ids, out, device)
        target_rows, target_intervals = run_target_sensitivity(
            models, images, masks, patient_ids, original, logits, out, device)
    occlusion_rows, occlusion_intervals = run_margin_occlusion(
        models, images_all[:120], masks_all[:120], selected_patients[:120], selected_indices,
        predictions, root / "results/E_final_B64_occlusion_cases.csv", out, device)

    summary = {
        "status": "complete", "training_performed": False, "models_selected": False,
        "device": str(device), "seed": SEED, "cam_cases": N_CAM, "occlusion_cases": len(occlusion_rows),
        "shared_random_reference": True, "identical_randomized_parameter_hashes_by_stage": stage_hashes,
        "probability_occlusion_reproduced": True,
        "limits": [
            "Randomization is a necessary sensitivity check, not proof of clinical correctness.",
            "Margin-target sensitivity uses 50 prespecified cases, not all 300 localization cases.",
            "Occlusion is limited to 89 cases permitting matched translated masks and favors smaller boxes.",
            "Intervals are conditional on frozen single-seed models and do not include training uncertainty.",
        ],
        "files": ["matched_parameter_randomization_cases.csv", "matched_parameter_randomization_intervals.csv",
                  "matched_parameter_randomization.pdf", "gradcam_target_sensitivity_cases.csv",
                  "gradcam_target_sensitivity_intervals.csv", "margin_occlusion_cases.csv",
                  "margin_occlusion_intervals.csv"],
    }
    (out / "xai_validity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print("XAI_VALIDITY_COMPLETE", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); run(Path(args.root), Path(args.out))
