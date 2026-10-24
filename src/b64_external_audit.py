from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import baseline
import b64_augmix as step15
import external_diagnosis as ext


IMAGE_SIZE = 64
EXPLANATION_SEED = 632916
EXPECTED_EFINAL_ROWS_SHA256 = "9995be2754f605e45bc908a83101445aa1539ddadbd7ddbf2048da775146d086"
HISTORICAL_EFINAL_MANIFEST_SHA256 = "4d484f337c8315eb7a767f3eed2c2fbd94a42fb86f34afebd851742283d22020"
HISTORICAL_B28_ARGMAX = {
    "accuracy": 0.7188836964391496,
    "balanced_accuracy": 0.7401017114753163,
    "auroc": 0.8379146603138572,
    "average_precision": 0.7754291522238987,
    "sensitivity": 0.8750390502967823,
    "specificity": 0.6051643726538505,
    "brier": 0.2558032959401595,
}


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


def stable_hash_rows(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_dicom64(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, dict]:
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(archive.read(member)))
    image = ds.pixel_array.astype(np.float32)
    image = image * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        image = image.max() + image.min() - image
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        raise ValueError(f"constant DICOM image: {member}")
    image = (image - lo) / (hi - lo)
    height, width = image.shape
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    crop = image[top:top + side, left:left + side]
    resized = np.asarray(
        Image.fromarray(np.uint8(np.round(crop * 255))).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        ), dtype=np.float32
    ) / 255.0
    return resized, {
        "height": height, "width": width, "side": side,
        "top": top, "left": left, "photometric": photometric,
    }


def mask_for_boxes64(boxes: list[dict], geometry: dict) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for box in boxes:
        x0 = (float(box["x"]) - geometry["left"]) / geometry["side"] * IMAGE_SIZE
        y0 = (float(box["y"]) - geometry["top"]) / geometry["side"] * IMAGE_SIZE
        x1 = (float(box["x"]) + float(box["width"]) - geometry["left"]) / geometry["side"] * IMAGE_SIZE
        y1 = (float(box["y"]) + float(box["height"]) - geometry["top"]) / geometry["side"] * IMAGE_SIZE
        xa, xb = max(0, int(np.floor(x0))), min(IMAGE_SIZE, int(np.ceil(x1)))
        ya, yb = max(0, int(np.floor(y0))), min(IMAGE_SIZE, int(np.ceil(y1)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def load_batch64(archive: zipfile.ZipFile, members: dict[str, str], records: list[dict]):
    images, masks, geometries = [], [], []
    for record in records:
        image, geometry = read_dicom64(archive, members[record["sop_instance_uid"]])
        images.append(image)
        masks.append(mask_for_boxes64(record["boxes"], geometry) if record["boxes"] else np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool))
        geometries.append(geometry)
    x = (torch.from_numpy(np.stack(images)).float().unsqueeze(1) - 0.5) / 0.5
    return x, masks, geometries


def center_prior64() -> np.ndarray:
    center = (IMAGE_SIZE - 1) / 2
    sigma = IMAGE_SIZE * 6.0 / 28.0
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    prior = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return prior / prior.sum()


def translated_masks64(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    ys, xs = np.where(mask)
    if not len(xs):
        return []
    output, seen = [], set()
    for _ in range(5000):
        dy = int(rng.integers(-int(ys.min()), IMAGE_SIZE - int(ys.max())))
        dx = int(rng.integers(-int(xs.min()), IMAGE_SIZE - int(xs.max())))
        if (dy, dx) == (0, 0) or (dy, dx) in seen:
            continue
        shifted = np.zeros_like(mask)
        shifted[ys + dy, xs + dx] = True
        if (shifted & mask).any():
            continue
        seen.add((dy, dx))
        output.append(shifted)
        if len(output) == count:
            break
    return output


def verify_final_records(files: dict[str, Path]) -> tuple[list[dict], dict]:
    records = ext.build_records(files["annotations"], files["mapping"])
    diag, final, summary = ext.patient_split(records)
    keys = ("patient_id", "subset_img_id", "study_uid", "sop_instance_uid", "class_name")
    final_rows = [{key: row[key] for key in keys} for row in final]
    rows_hash = stable_hash_rows(final_rows)
    if rows_hash != EXPECTED_EFINAL_ROWS_SHA256:
        raise RuntimeError(f"Historical E_final rows changed: {rows_hash}")
    overlap = {r["patient_id"] for r in diag} & {r["patient_id"] for r in final}
    if overlap or len(final) != 26774 or len({r["patient_id"] for r in final}) != 11009:
        raise AssertionError((len(overlap), len(final), len({r["patient_id"] for r in final})))
    return final, {
        "status": "historical_E_final_reopened_for_post_hoc_transfer_audit",
        "historical_manifest_sha256": HISTORICAL_EFINAL_MANIFEST_SHA256,
        "recomputed_rows_sha256": rows_hash,
        "patient_overlap_with_E_diag": 0,
        "patients": 11009,
        "images": 26774,
        "class_counts": summary["final_class_counts_only_for_split_audit"],
        "selection_or_tuning_permitted": False,
        "opened_utc": utcnow(),
    }


def paired_predict(models: dict[str, torch.nn.Module], zip_path: Path, members: dict[str, str], records: list[dict], device: torch.device, batch_size: int = 128):
    logits = {name: [] for name in models}
    with zipfile.ZipFile(zip_path) as archive, torch.no_grad():
        for start in range(0, len(records), batch_size):
            x, _, _ = load_batch64(archive, members, records[start:start + batch_size])
            x = x.to(device)
            for name, model in models.items():
                logits[name].append(model(x).cpu())
            if start == 0 or (start // batch_size + 1) % 20 == 0:
                print(f"B64 E_final predictions: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    output = {}
    for name, parts in logits.items():
        logit = torch.cat(parts).numpy()
        output[name] = {"logits": logit, "scores": torch.softmax(torch.from_numpy(logit), 1).numpy()[:, 1]}
    return output


def model_occlusion(model: torch.nn.Module, x: torch.Tensor, mask: np.ndarray, outside: list[np.ndarray], device: torch.device) -> dict:
    with torch.no_grad():
        clean = float(torch.softmax(model(x.unsqueeze(0).to(device)), 1)[0, 1])
        result = {"outside_locations": len(outside)}
        for mode in ("mean", "blur"):
            variants = [ext.occluded(x, mask, mode)] + [ext.occluded(x, item, mode) for item in outside]
            scores = torch.softmax(model(torch.stack(variants).to(device)), 1)[:, 1].cpu().numpy()
            inside = clean - float(scores[0])
            outside_drop = clean - float(scores[1:].mean())
            result[f"{mode}_inside_drop"] = inside
            result[f"{mode}_outside_drop"] = outside_drop
            result[f"{mode}_inside_minus_outside"] = inside - outside_drop
    return result


def summarize_localization(rows: list[dict], name: str) -> dict:
    return {
        "n": len(rows),
        "area_corrected_energy_mean": float(np.mean([r[f"{name}_area_corrected_energy"] for r in rows])),
        "area_corrected_energy_median": float(np.median([r[f"{name}_area_corrected_energy"] for r in rows])),
        "pointing_hit_rate": float(np.mean([r[f"{name}_pointing_hit"] for r in rows])),
        "zero_cam_rate": float(np.mean([r[f"{name}_zero_cam"] for r in rows])),
    }


def paired_explain(models: dict[str, torch.nn.Module], zip_path: Path, members: dict[str, str], records: list[dict], prediction: dict, out: Path, device: torch.device) -> dict:
    positives = [i for i, row in enumerate(records) if row["class_name"] == "Lung Opacity"]
    selected = sorted(positives, key=lambda i: hashlib.sha256(f"{EXPLANATION_SEED}|{records[i]['subset_img_id']}".encode()).hexdigest())[:300]
    selected_records = [records[i] for i in selected]
    cams_parts = {name: [] for name in models}
    images_parts, masks_all = [], []
    with zipfile.ZipFile(zip_path) as archive:
        for start in range(0, len(selected_records), 24):
            x, masks, _ = load_batch64(archive, members, selected_records[start:start + 24])
            for name, model in models.items():
                cams, _ = baseline.gradcam(model, x, device)
                cams_parts[name].append(cams)
            images_parts.append((x[:, 0].numpy() * 0.5 + 0.5).clip(0, 1))
            masks_all.extend(masks)
    cams = {name: np.concatenate(parts) for name, parts in cams_parts.items()}
    images = np.concatenate(images_parts)
    center = center_prior64()
    localization_rows = []
    for local_i, global_i in enumerate(selected):
        row = {
            "subset_img_id": records[global_i]["subset_img_id"],
            "patient_id": records[global_i]["patient_id"],
            "box_area_fraction": float(masks_all[local_i].mean()),
        }
        for name in models:
            corrected, pointing, zero = ext.energy_and_pointing(cams[name][local_i], masks_all[local_i])
            row.update({
                f"{name}_score": float(prediction[name]["scores"][global_i]),
                f"{name}_area_corrected_energy": corrected,
                f"{name}_pointing_hit": pointing,
                f"{name}_zero_cam": zero,
            })
        center_corrected, center_pointing, _ = ext.energy_and_pointing(center, masks_all[local_i])
        row["center_prior_area_corrected_energy"] = center_corrected
        row["center_prior_pointing_hit"] = center_pointing
        row["difference_area_corrected_energy"] = row["augmix_area_corrected_energy"] - row["plain_area_corrected_energy"]
        row["difference_pointing_hit"] = row["augmix_pointing_hit"] - row["plain_pointing_hit"]
        localization_rows.append(row)
    save_csv(out / "E_final_B64_localization_cases.csv", localization_rows)

    rng = np.random.default_rng(EXPLANATION_SEED)
    occlusion_rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for local_i, global_i in enumerate(selected[:120]):
            x, masks, _ = load_batch64(archive, members, [records[global_i]])
            outside = translated_masks64(masks[0], 20, rng)
            if not outside:
                continue
            row = {"subset_img_id": records[global_i]["subset_img_id"], "patient_id": records[global_i]["patient_id"]}
            for name, model in models.items():
                values = model_occlusion(model, x[0], masks[0], outside, device)
                for key, value in values.items():
                    row[f"{name}_{key}"] = value
            for mode in ("mean", "blur"):
                key = f"{mode}_inside_minus_outside"
                row[f"difference_{key}"] = row[f"augmix_{key}"] - row[f"plain_{key}"]
            occlusion_rows.append(row)
            if (local_i + 1) % 20 == 0:
                print(f"B64 E_final occlusion: {local_i + 1}/120", flush=True)
    save_csv(out / "E_final_B64_occlusion_cases.csv", occlusion_rows)

    localization = {name: summarize_localization(localization_rows, name) for name in models}
    localization["augmix_minus_plain"] = {
        key: localization["augmix"][key] - localization["plain"][key]
        for key in ("area_corrected_energy_mean", "area_corrected_energy_median", "pointing_hit_rate", "zero_cam_rate")
    }
    occlusion = {}
    for name in models:
        occlusion[name] = {
            "n": len(occlusion_rows),
            "mean_inside_minus_outside": float(np.mean([r[f"{name}_mean_inside_minus_outside"] for r in occlusion_rows])),
            "blur_inside_minus_outside": float(np.mean([r[f"{name}_blur_inside_minus_outside"] for r in occlusion_rows])),
            "both_positive_fraction": float(np.mean([
                r[f"{name}_mean_inside_minus_outside"] > 0 and r[f"{name}_blur_inside_minus_outside"] > 0
                for r in occlusion_rows
            ])),
        }
    occlusion["augmix_minus_plain"] = {
        key: occlusion["augmix"][key] - occlusion["plain"][key]
        for key in ("mean_inside_minus_outside", "blur_inside_minus_outside", "both_positive_fraction")
    }
    sanity = {}
    with zipfile.ZipFile(zip_path) as archive:
        x50, _, _ = load_batch64(archive, members, selected_records[:50])
    for offset, (name, model) in enumerate(models.items()):
        baseline.seed_all(EXPLANATION_SEED + 10 + offset)
        randomized = copy.deepcopy(model).cpu()
        randomized.apply(ext.reset_parameters)
        randomized.to(device).eval()
        random_cams, _ = baseline.gradcam(randomized, x50, device)
        similarity = baseline.cosine_rows(cams[name][:50], random_cams)
        sanity[name] = {
            "n": 50,
            "trained_vs_fully_randomized_cam_cosine_mean": float(similarity.mean()),
            "trained_vs_fully_randomized_cam_cosine_median": float(np.median(similarity)),
        }
    np.savez_compressed(
        out / "E_final_B64_explanations.npz",
        selected_indices=np.array(selected), images=images, masks=np.stack(masks_all),
        plain_cams=cams["plain"], augmix_cams=cams["augmix"],
    )
    plot_examples(out, selected, records, images, masks_all, cams, prediction)
    summary = {
        "selection": {"seed": EXPLANATION_SEED, "n_localization": 300, "n_occlusion": len(occlusion_rows), "same_cases_and_settings": True},
        "localization": localization,
        "matched_occlusion": occlusion,
        "center_prior": {
            "area_corrected_energy_mean": float(np.mean([r["center_prior_area_corrected_energy"] for r in localization_rows])),
            "pointing_hit_rate": float(np.mean([r["center_prior_pointing_hit"] for r in localization_rows])),
        },
        "randomization_sanity": sanity,
        "limits": [
            "public bounding boxes are coarse rather than pixel masks",
            "Grad-CAM and occlusion provide region-dependence evidence, not medical causality",
            "the historical E_final was previously opened and cannot support candidate selection",
        ],
    }
    save_json(out / "E_final_B64_explainability_summary.json", summary)
    return summary


def plot_examples(out: Path, selected: list[int], records: list[dict], images: np.ndarray, masks: list[np.ndarray], cams: dict, prediction: dict) -> None:
    difference = []
    for local_i in range(len(selected)):
        plain, _, _ = ext.energy_and_pointing(cams["plain"][local_i], masks[local_i])
        augmix, _, _ = ext.energy_and_pointing(cams["augmix"][local_i], masks[local_i])
        difference.append(augmix - plain)
    order = np.argsort(difference)
    chosen = np.unique(np.linspace(0, len(order) - 1, 8, dtype=int))
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 2.3 * len(chosen)))
    for row_i, position in enumerate(chosen):
        local_i = order[position]
        global_i = selected[local_i]
        axes[row_i, 0].imshow(images[local_i], cmap="gray", vmin=0, vmax=1)
        axes[row_i, 0].contour(masks[local_i].astype(float), levels=[0.5], colors=["lime"], linewidths=1)
        axes[row_i, 0].set_title(f"Image / box | XAI delta={difference[local_i]:+.3f}", fontsize=8)
        for column, name in ((1, "plain"), (2, "augmix")):
            axes[row_i, column].imshow(images[local_i], cmap="gray", vmin=0, vmax=1)
            axes[row_i, column].imshow(cams[name][local_i], cmap="jet", alpha=.42, vmin=0, vmax=1)
            axes[row_i, column].contour(masks[local_i].astype(float), levels=[0.5], colors=["lime"], linewidths=1)
            axes[row_i, column].set_title(f"{name} p={prediction[name]['scores'][global_i]:.2f}", fontsize=8)
        for axis in axes[row_i]:
            axis.axis("off")
    fig.suptitle("Historical E_final: paired B64 Grad-CAM with public boxes")
    fig.tight_layout()
    fig.savefig(out / "E_final_B64_gradcam_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(project_root: Path, plain_out: Path, augmix_out: Path) -> dict:
    project_root, plain_out, augmix_out = Path(project_root), Path(plain_out), Path(augmix_out)
    stage = project_root / "md6329_step16_b64_external_audit"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = stage / "runs" / run_id
    out.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"plain": step15.load_model(plain_out, device), "augmix": step15.load_model(augmix_out, device)}
    frozen = {name: step15.sha256(path / "best_weights.pt") for name, path in (("plain", plain_out), ("augmix", augmix_out))}
    plan = {
        "status": "frozen_B64_models_before_historical_external_audit",
        "created_utc": utcnow(), "image_size": IMAGE_SIZE, "threshold": 0.5,
        "weights_sha256": frozen,
        "prediction_primary": "Normal vs Lung Opacity AUROC and accuracy at frozen argmax threshold 0.5",
        "explanation_primary": "paired area-corrected Grad-CAM energy within public lesion boxes",
        "intervention": "paired lesion-box minus matched non-overlapping box occlusion effect",
        "E_final_role": "post-hoc transfer audit only; no selection or tuning",
    }
    save_json(out / "audit_plan_before_opening.json", plan)
    files = ext.obtain_rsna(project_root / "md6329_step10_resnet_reproduction")
    members = ext.zip_members(files["images"])
    records, opening = verify_final_records(files)
    missing = [r["sop_instance_uid"] for r in records if r["sop_instance_uid"] not in members]
    if missing:
        raise AssertionError(f"Missing {len(missing)} E_final DICOMs")
    save_json(out / "E_final_B64_opening_audit.json", opening)
    started = time.time()
    prediction = paired_predict(models, files["images"], members, records, device)
    prediction_seconds = time.time() - started
    labels = np.array([r["target"] for r in records], dtype=np.int64)
    np.savez_compressed(
        out / "E_final_B64_paired_predictions.npz", labels=labels,
        patient_ids=np.array([r["patient_id"] for r in records]),
        subset_img_ids=np.array([r["subset_img_id"] for r in records]),
        plain_logits=prediction["plain"]["logits"], plain_scores=prediction["plain"]["scores"],
        augmix_logits=prediction["augmix"]["logits"], augmix_scores=prediction["augmix"]["scores"],
    )
    metrics = {name: ext.prediction_metrics(records, prediction[name]["scores"], 0.5) for name in models}
    plain_primary = metrics["plain"]["primary_normal_vs_lung_opacity_argmax"]
    augmix_primary = metrics["augmix"]["primary_normal_vs_lung_opacity_argmax"]
    differences = {key: augmix_primary[key] - plain_primary[key] for key in (
        "accuracy", "balanced_accuracy", "auroc", "average_precision", "sensitivity", "specificity", "brier"
    )}
    save_json(out / "E_final_B64_prediction_metrics.json", {
        "metrics": metrics, "augmix_minus_plain": differences,
        "historical_B28_argmax_reference": HISTORICAL_B28_ARGMAX,
    })
    xai = paired_explain(models, files["images"], members, records, prediction, out, device)
    summary = {
        "status": "historical_E_final_B64_audit_complete_no_further_tuning",
        "completed_utc": utcnow(), "run_id": run_id, "run_directory": str(out),
        "device": str(device), "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "opening_audit": opening, "weights_sha256": frozen, "prediction_seconds": prediction_seconds,
        "prediction_metrics": metrics, "prediction_augmix_minus_plain": differences,
        "historical_B28_argmax_reference": HISTORICAL_B28_ARGMAX,
        "explainability": xai,
        "E_final_reused_for_selection": False, "further_tuning_permitted_from_this_result": False,
    }
    save_json(out / "E_final_B64_summary.json", summary)
    save_json(stage / "latest_run.json", {
        "run_id": run_id, "relative_path": str(out.relative_to(stage)).replace("\\", "/"),
        "status": summary["status"], "summary_sha256": step15.sha256(out / "E_final_B64_summary.json"),
    })
    return summary


def smoke_test() -> dict:
    geometry = {"left": 0, "top": 0, "side": 100}
    mask = mask_for_boxes64([{"x": 25, "y": 25, "width": 50, "height": 50}], geometry)
    if mask.shape != (64, 64) or not (900 <= int(mask.sum()) <= 1156):
        raise AssertionError((mask.shape, int(mask.sum())))
    prior = center_prior64()
    if not np.isclose(prior.sum(), 1.0):
        raise AssertionError(prior.sum())
    return {"image_size": IMAGE_SIZE, "mask_pixels": int(mask.sum()), "center_prior_sum": float(prior.sum())}


if __name__ == "__main__":
    print(json.dumps(smoke_test(), ensure_ascii=False, indent=2))
