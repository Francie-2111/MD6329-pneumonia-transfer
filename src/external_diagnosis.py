from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import baseline


SPLIT_SEED = 632910
EXTERNAL_EXPLANATION_SEED = 632911
RSNA_IMAGE_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
RSNA_ANNOTATION_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
RSNA_MAPPING_URL = "https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json"
EXPECTED = {
    "images": {"bytes": 3978753654, "etag": "c7d488b2781c73a21c486e8d64f98635-380"},
    "annotations": {"bytes": 52003071, "sha256": "f8f5c54bae158fc0e2f019af979471803151a5922463ff3a378d7734e01b18fb"},
    "mapping": {"bytes": 17587143, "sha256": "803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_resume(url: str, path: Path, expected_bytes: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected_bytes:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            temporary.unlink(missing_ok=True)
            offset = 0
        mode = "ab" if offset else "wb"
        received = offset
        started = time.time()
        with temporary.open(mode) as f:
            for block in response.iter_content(8 * 1024 * 1024):
                if not block:
                    continue
                f.write(block)
                received += len(block)
                if received % (256 * 1024 * 1024) < len(block):
                    print(f"downloaded {received / 2**30:.2f} / {expected_bytes / 2**30:.2f} GiB", flush=True)
        print(f"download finished in {(time.time() - started) / 60:.1f} min", flush=True)
    if temporary.stat().st_size != expected_bytes:
        raise RuntimeError(f"Download size mismatch for {path.name}: {temporary.stat().st_size} != {expected_bytes}")
    temporary.replace(path)
    return path


def obtain_rsna(root: Path) -> dict[str, Path]:
    data = root / "data"
    files = {
        "images": download_resume(RSNA_IMAGE_URL, data / "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip", EXPECTED["images"]["bytes"]),
        "annotations": download_resume(RSNA_ANNOTATION_URL, data / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json", EXPECTED["annotations"]["bytes"]),
        "mapping": download_resume(RSNA_MAPPING_URL, data / "pneumonia-challenge-dataset-mappings_2018.json", EXPECTED["mapping"]["bytes"]),
    }
    for name in ("annotations", "mapping"):
        actual = baseline.sha256(files[name])
        if actual != EXPECTED[name]["sha256"]:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual}")
    return files


def calculated_label_ids(annotation_export: dict) -> dict[str, str]:
    groups = [g for g in annotation_export["labelGroups"] if g["name"] == "Calculated"]
    if len(groups) != 1:
        raise AssertionError(f"Expected one Calculated group, found {len(groups)}")
    labels = {item["name"].strip(): item["id"] for item in groups[0]["labels"]}
    required = {"Normal", "No Lung Opacity / Not Normal", "Lung Opacity"}
    if set(labels) != required:
        raise AssertionError(labels)
    return labels


def build_records(annotation_path: Path, mapping_path: Path) -> list[dict]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    mappings = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_ids = calculated_label_ids(annotations)
    id_to_name = {value: key for key, value in label_ids.items()}
    by_uid: dict[str, dict] = {}
    for item in annotations["datasets"][0]["annotations"]:
        if item["labelId"] not in id_to_name:
            continue
        uid = item["StudyInstanceUID"]
        label = id_to_name[item["labelId"]]
        record = by_uid.setdefault(uid, {"class_name": label, "boxes": []})
        if record["class_name"] != label:
            raise AssertionError((uid, record["class_name"], label))
        if label == "Lung Opacity":
            box = dict(item["data"])
            box["image_height"] = int(item["height"])
            box["image_width"] = int(item["width"])
            record["boxes"].append(box)
    mapping_by_uid = {item["StudyInstanceUID"]: item for item in mappings}
    records = []
    for uid, value in by_uid.items():
        mapping = mapping_by_uid[uid]
        patient_id = mapping["img_id"].split("_", 1)[0]
        records.append({
            "study_uid": uid,
            "sop_instance_uid": mapping["SOPInstanceUID"],
            "subset_img_id": mapping["subset_img_id"],
            "nih_image_id": mapping["img_id"],
            "patient_id": patient_id,
            "class_name": value["class_name"],
            "target": 1 if value["class_name"] == "Lung Opacity" else 0,
            "primary_eligible": value["class_name"] != "No Lung Opacity / Not Normal",
            "boxes": value["boxes"],
        })
    counts = {name: sum(r["class_name"] == name for r in records) for name in ("Normal", "No Lung Opacity / Not Normal", "Lung Opacity")}
    if len(records) != 29684 or counts != {"Normal": 9790, "No Lung Opacity / Not Normal": 12788, "Lung Opacity": 7106}:
        raise AssertionError((len(records), counts))
    return sorted(records, key=lambda r: (r["patient_id"], r["subset_img_id"]))


def patient_split(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    patients = sorted({r["patient_id"] for r in records})
    ranked = sorted(patients, key=lambda p: hashlib.sha256(f"{SPLIT_SEED}|{p}".encode()).hexdigest())
    n_diag = int(math.ceil(0.10 * len(ranked)))
    diag_patients = set(ranked[:n_diag])
    diag = [r for r in records if r["patient_id"] in diag_patients]
    final = [r for r in records if r["patient_id"] not in diag_patients]
    if {r["patient_id"] for r in diag} & {r["patient_id"] for r in final}:
        raise AssertionError("patient leakage")
    summary = {
        "split_seed": SPLIT_SEED,
        "method": "rank unique NIH patient IDs by SHA256(seed|patient_id); first ceil(10%) are E_diag",
        "patients_total": len(patients),
        "patients_diag": len(diag_patients),
        "patients_final_sealed": len(patients) - len(diag_patients),
        "images_diag": len(diag),
        "images_final_sealed": len(final),
        "diag_class_counts": {name: sum(r["class_name"] == name for r in diag) for name in ("Normal", "No Lung Opacity / Not Normal", "Lung Opacity")},
        "final_class_counts_only_for_split_audit": {name: sum(r["class_name"] == name for r in final) for name in ("Normal", "No Lung Opacity / Not Normal", "Lung Opacity")},
        "final_policy": "IDs and class-count audit are frozen before diagnosis; E_final pixels and predictions are not read in this stage",
    }
    return diag, final, summary


def save_split(out: Path, records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    diag, final, summary = patient_split(records)
    manifest = {
        "created_utc": utcnow(),
        "summary": summary,
        "E_diag": [{k: r[k] for k in ("patient_id", "subset_img_id", "study_uid", "sop_instance_uid", "class_name")} for r in diag],
        "E_final_sealed": [{k: r[k] for k in ("patient_id", "subset_img_id", "study_uid", "sop_instance_uid", "class_name")} for r in final],
    }
    baseline.save_json(out / "external_split_manifest.json", manifest)
    summary["manifest_sha256"] = baseline.sha256(out / "external_split_manifest.json")
    baseline.save_json(out / "external_split_summary.json", summary)
    return diag, final, summary


def zip_members(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".dcm")]
    index = {Path(name).stem: name for name in members}
    if len(index) < 29684:
        raise AssertionError(f"Only {len(index)} DICOM members found")
    return index


def read_dicom(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, dict]:
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
    top = (height - side) // 2
    left = (width - side) // 2
    crop = image[top:top + side, left:left + side]
    resized = np.asarray(Image.fromarray(np.uint8(np.round(crop * 255))).resize((28, 28), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return resized, {"height": height, "width": width, "side": side, "top": top, "left": left, "photometric": photometric}


def normalized(images: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(images).float().unsqueeze(1) - 0.5) / 0.5


def mask_for_boxes(boxes: list[dict], geometry: dict) -> np.ndarray:
    mask = np.zeros((28, 28), dtype=bool)
    for box in boxes:
        x0 = (float(box["x"]) - geometry["left"]) / geometry["side"] * 28
        y0 = (float(box["y"]) - geometry["top"]) / geometry["side"] * 28
        x1 = (float(box["x"]) + float(box["width"]) - geometry["left"]) / geometry["side"] * 28
        y1 = (float(box["y"]) + float(box["height"]) - geometry["top"]) / geometry["side"] * 28
        xa, xb = max(0, int(np.floor(x0))), min(28, int(np.ceil(x1)))
        ya, yb = max(0, int(np.floor(y0))), min(28, int(np.ceil(y1)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def load_batch(archive: zipfile.ZipFile, members: dict[str, str], records: list[dict]) -> tuple[torch.Tensor, list[np.ndarray], list[dict]]:
    images, masks, geometries = [], [], []
    for record in records:
        image, geometry = read_dicom(archive, members[record["sop_instance_uid"]])
        images.append(image)
        masks.append(mask_for_boxes(record["boxes"], geometry) if record["boxes"] else np.zeros((28, 28), dtype=bool))
        geometries.append(geometry)
    return normalized(np.stack(images)), masks, geometries


def predict_diag(model: nn.Module, zip_path: Path, members: dict[str, str], records: list[dict], device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    logits, geometries = [], []
    with zipfile.ZipFile(zip_path) as archive, torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            x, _masks, geo = load_batch(archive, members, batch)
            logits.append(model(x.to(device)).cpu())
            geometries.extend(geo)
            if start == 0 or (start // batch_size + 1) % 10 == 0:
                print(f"external predictions: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    logits_np = torch.cat(logits).numpy()
    scores = torch.softmax(torch.from_numpy(logits_np), 1).numpy()[:, 1]
    return logits_np, scores, geometries


def prediction_metrics(records: list[dict], scores: np.ndarray, source_threshold: float) -> dict:
    labels = np.array([r["target"] for r in records])
    primary = np.array([r["primary_eligible"] for r in records], dtype=bool)
    hard = np.array([r["class_name"] == "No Lung Opacity / Not Normal" for r in records])
    result = {
        "primary_normal_vs_lung_opacity_argmax": baseline.metrics(labels[primary], scores[primary], 0.5),
        "primary_normal_vs_lung_opacity_source_youden": baseline.metrics(labels[primary], scores[primary], source_threshold),
        "all_with_abnormal_nonopacity_as_negative_argmax": baseline.metrics(labels, scores, 0.5),
        "all_with_abnormal_nonopacity_as_negative_source_youden": baseline.metrics(labels, scores, source_threshold),
        "abnormal_nonopacity_n": int(hard.sum()),
        "abnormal_nonopacity_false_positive_rate_argmax": float((scores[hard] >= 0.5).mean()),
        "abnormal_nonopacity_false_positive_rate_source_youden": float((scores[hard] >= source_threshold).mean()),
        "abnormal_nonopacity_mean_pneumonia_probability": float(scores[hard].mean()),
    }
    return result


def energy_and_pointing(cam: np.ndarray, mask: np.ndarray) -> tuple[float, int, int]:
    total = float(cam.sum())
    zero = int(total <= 1e-12)
    energy = float(cam[mask].sum() / total) if total > 1e-12 else 0.0
    corrected = energy - float(mask.mean())
    if total > 1e-12:
        y, x = np.unravel_index(int(cam.argmax()), cam.shape)
        pointing = int(mask[y, x])
    else:
        pointing = 0
    return corrected, pointing, zero


def center_prior() -> np.ndarray:
    y, x = np.mgrid[0:28, 0:28]
    prior = np.exp(-((x - 13.5) ** 2 + (y - 13.5) ** 2) / (2 * 6.0 ** 2))
    return prior / prior.sum()


def translated_masks(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    ys, xs = np.where(mask)
    if not len(xs):
        return []
    output, seen = [], set()
    for _ in range(2000):
        dy = int(rng.integers(-int(ys.min()), 28 - int(ys.max())))
        dx = int(rng.integers(-int(xs.min()), 28 - int(xs.max())))
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


def occluded(x: torch.Tensor, mask: np.ndarray, mode: str) -> torch.Tensor:
    y = x.clone()
    m = torch.from_numpy(mask).to(dtype=torch.bool)
    if mode == "mean":
        replacement = torch.tensor(0.0, dtype=y.dtype)
        y[0, m] = replacement
    elif mode == "blur":
        blurred = F.avg_pool2d(y.unsqueeze(0), kernel_size=5, stride=1, padding=2)[0]
        y[0, m] = blurred[0, m]
    else:
        raise ValueError(mode)
    return y


def occlusion_case(model: nn.Module, x: torch.Tensor, mask: np.ndarray, device: torch.device, rng: np.random.Generator) -> dict:
    outside = translated_masks(mask, 20, rng)
    if not outside:
        return {
            "outside_locations": 0,
            "mean_inside_drop": None,
            "mean_outside_drop": None,
            "mean_inside_minus_outside": None,
            "blur_inside_drop": None,
            "blur_outside_drop": None,
            "blur_inside_minus_outside": None,
        }
    with torch.no_grad():
        clean = float(torch.softmax(model(x.unsqueeze(0).to(device)), 1)[0, 1])
        result = {"outside_locations": len(outside)}
        for mode in ("mean", "blur"):
            variants = [occluded(x, mask, mode)] + [occluded(x, m, mode) for m in outside]
            scores = torch.softmax(model(torch.stack(variants).to(device)), 1)[:, 1].cpu().numpy()
            result[f"{mode}_inside_drop"] = clean - float(scores[0])
            result[f"{mode}_outside_drop"] = clean - float(scores[1:].mean())
            result[f"{mode}_inside_minus_outside"] = result[f"{mode}_inside_drop"] - result[f"{mode}_outside_drop"]
        return result


def reset_parameters(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def explain_external(model: nn.Module, zip_path: Path, members: dict[str, str], records: list[dict], scores: np.ndarray, out: Path, device: torch.device, maximum: int = 300) -> dict:
    positives = [i for i, r in enumerate(records) if r["class_name"] == "Lung Opacity"]
    selected = sorted(positives, key=lambda i: hashlib.sha256(f"{EXTERNAL_EXPLANATION_SEED}|{records[i]['subset_img_id']}".encode()).hexdigest())[:maximum]
    selected_records = [records[i] for i in selected]
    cams_all, images_all, masks_all = [], [], []
    with zipfile.ZipFile(zip_path) as archive:
        for start in range(0, len(selected_records), 32):
            batch = selected_records[start:start + 32]
            x, masks, _ = load_batch(archive, members, batch)
            cams, _ = baseline.gradcam(model, x, device)
            cams_all.append(cams)
            images_all.append((x[:, 0].numpy() * 0.5 + 0.5).clip(0, 1))
            masks_all.extend(masks)
    cams = np.concatenate(cams_all)
    images = np.concatenate(images_all)
    center = center_prior()
    localization_rows = []
    for local_i, global_i in enumerate(selected):
        corrected, pointing, zero = energy_and_pointing(cams[local_i], masks_all[local_i])
        center_corrected, center_pointing, _ = energy_and_pointing(center, masks_all[local_i])
        localization_rows.append({
            "subset_img_id": records[global_i]["subset_img_id"],
            "patient_id": records[global_i]["patient_id"],
            "model_score": float(scores[global_i]),
            "prediction_correct_argmax": int(scores[global_i] >= 0.5),
            "area_corrected_energy": corrected,
            "pointing_hit": pointing,
            "zero_cam": zero,
            "center_prior_area_corrected_energy": center_corrected,
            "center_prior_pointing_hit": center_pointing,
            "box_area_fraction": float(masks_all[local_i].mean()),
        })
    baseline.save_csv(out / "external_localization_cases.csv", localization_rows)

    rng = np.random.default_rng(EXTERNAL_EXPLANATION_SEED)
    occlusion_rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for local_i, global_i in enumerate(selected[: min(120, len(selected))]):
            x, masks, _ = load_batch(archive, members, [records[global_i]])
            row = {"subset_img_id": records[global_i]["subset_img_id"], "patient_id": records[global_i]["patient_id"]}
            row.update(occlusion_case(model, x[0], masks[0], device, rng))
            occlusion_rows.append(row)
            if (local_i + 1) % 20 == 0:
                print(f"occlusion analysis: {local_i + 1}/{min(120, len(selected))}", flush=True)
    baseline.save_csv(out / "external_occlusion_cases.csv", occlusion_rows)

    baseline.seed_all(EXTERNAL_EXPLANATION_SEED)
    randomized = copy.deepcopy(model).cpu()
    randomized.apply(reset_parameters)
    randomized.to(device).eval()
    with zipfile.ZipFile(zip_path) as archive:
        x, _, _ = load_batch(archive, members, selected_records[: min(50, len(selected_records))])
    randomized_cams, _ = baseline.gradcam(randomized, x, device)
    trained_subset = cams[:len(randomized_cams)]
    randomization_similarity = baseline.cosine_rows(trained_subset, randomized_cams)

    localization = {
        "n": len(localization_rows),
        "area_corrected_energy_mean": float(np.mean([r["area_corrected_energy"] for r in localization_rows])),
        "area_corrected_energy_median": float(np.median([r["area_corrected_energy"] for r in localization_rows])),
        "pointing_hit_rate": float(np.mean([r["pointing_hit"] for r in localization_rows])),
        "zero_cam_rate": float(np.mean([r["zero_cam"] for r in localization_rows])),
        "center_prior_area_corrected_energy_mean": float(np.mean([r["center_prior_area_corrected_energy"] for r in localization_rows])),
        "center_prior_pointing_hit_rate": float(np.mean([r["center_prior_pointing_hit"] for r in localization_rows])),
    }
    valid_occlusion = [r for r in occlusion_rows if r["outside_locations"]]
    occlusion = {
        "n": len(valid_occlusion),
        "mean_inside_minus_outside": float(np.mean([r["mean_inside_minus_outside"] for r in valid_occlusion])),
        "blur_inside_minus_outside": float(np.mean([r["blur_inside_minus_outside"] for r in valid_occlusion])),
        "both_positive_fraction": float(np.mean([(r["mean_inside_minus_outside"] > 0 and r["blur_inside_minus_outside"] > 0) for r in valid_occlusion])),
    }
    sanity = {
        "n": int(len(randomization_similarity)),
        "trained_vs_fully_randomized_cam_cosine_mean": float(randomization_similarity.mean()),
        "trained_vs_fully_randomized_cam_cosine_median": float(np.median(randomization_similarity)),
        "interpretation": "lower similarity supports model dependence; passing this check is necessary but does not prove correctness",
    }
    summary = {
        "localization": localization,
        "matched_occlusion": occlusion,
        "randomization_sanity": sanity,
        "limits": ["bounding boxes are coarse, not pixel masks", "B28 Grad-CAM is spatially coarse", "occlusion can introduce artifacts", "these results do not establish medical causality"],
    }
    baseline.save_json(out / "external_explainability_summary.json", summary)
    np.savez_compressed(out / "external_explanations.npz", selected_indices=np.array(selected), cams=cams, images=images, masks=np.stack(masks_all), randomized_similarity=randomization_similarity)
    plot_examples(out, records, selected, images, cams, masks_all, scores)
    return summary


def plot_examples(out: Path, records: list[dict], selected: list[int], images: np.ndarray, cams: np.ndarray, masks: list[np.ndarray], scores: np.ndarray) -> None:
    order = np.argsort([scores[i] for i in selected])
    chosen = np.unique(np.linspace(0, len(order) - 1, min(12, len(order)), dtype=int))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax, position in zip(axes.ravel(), chosen):
        local = order[position]
        global_i = selected[local]
        ax.imshow(images[local], cmap="gray", vmin=0, vmax=1)
        ax.imshow(cams[local], cmap="jet", alpha=0.42, vmin=0, vmax=1)
        ax.contour(masks[local].astype(float), levels=[0.5], colors=["lime"], linewidths=1)
        ax.set_title(f"p={scores[global_i]:.2f}  {'TP' if scores[global_i] >= .5 else 'FN'}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(chosen):]:
        ax.axis("off")
    fig.suptitle("E_diag lung-opacity cases: Grad-CAM and public bounding boxes")
    fig.tight_layout()
    fig.savefig(out / "external_gradcam_examples.png", dpi=180)
    plt.close(fig)


def problem_report(out: Path, internal: dict, external_metrics: dict, explanations: dict, split_summary: dict) -> str:
    source = internal["official_argmax_metrics"]
    external = external_metrics["primary_normal_vs_lung_opacity_argmax"]
    drop_auc = external["auroc"] - source["auroc"]
    drop_ba = external["balanced_accuracy"] - source["balanced_accuracy"]
    loc = explanations["localization"]
    occ = explanations["matched_occlusion"]
    sanity = explanations["randomization_sanity"]
    issues = []
    if drop_auc < -0.05:
        issues.append(f"外部 AUROC 比内部低 {abs(drop_auc):.3f}，存在明显排序迁移问题。")
    else:
        issues.append(f"外部与内部 AUROC 差为 {drop_auc:+.3f}；需结合阈值指标判断迁移问题。")
    if drop_ba < -0.05:
        issues.append(f"外部 balanced accuracy 比内部低 {abs(drop_ba):.3f}，固定决策规则迁移较差。")
    threshold_gap = external_metrics["primary_normal_vs_lung_opacity_source_youden"]["balanced_accuracy"] - external["balanced_accuracy"]
    issues.append(f"源域 Youden 阈值与 0.5 在外部 balanced accuracy 上相差 {threshold_gap:+.3f}；这是固定源域阈值比较，不在外部重估阈值。")
    if loc["area_corrected_energy_mean"] <= loc["center_prior_area_corrected_energy_mean"]:
        issues.append("模型框内热量未超过中心先验对照，Grad-CAM 的病灶区域一致性不足。")
    else:
        issues.append("模型框内热量超过中心先验，但仍需匹配遮挡支持其忠实度。")
    if occ["mean_inside_minus_outside"] <= 0 or occ["blur_inside_minus_outside"] <= 0:
        issues.append("至少一种遮挡替换未显示框内区域比匹配框外区域更重要，病灶依赖证据不足。")
    else:
        issues.append("两种遮挡替换的平均方向均支持框内区域更重要，但个案一致性与伪影限制仍需报告。")
    if sanity["trained_vs_fully_randomized_cam_cosine_mean"] > 0.5:
        issues.append("训练模型与完全随机化模型的热图相似度偏高，可视化可能较多反映图像结构而非学习权重。")
    hard_fpr = external_metrics["abnormal_nonopacity_false_positive_rate_argmax"]
    if hard_fpr > 0.5:
        issues.append(f"异常但无肺部阴影病例的假阳性率为 {hard_fpr:.3f}，模型可能学习了非特异异常线索。")
    lines = [
        "# B28 复现与外部 E_diag 诊断报告",
        "",
        f"生成时间：{utcnow()}。训练随机种子：{baseline.SEED}。外部诊断患者：{split_summary['patients_diag']}；封存 E_final 患者：{split_summary['patients_final_sealed']}。",
        "",
        "## 预测结果",
        "",
        f"- 内部 test：AUROC {source['auroc']:.4f}，accuracy {source['accuracy']:.4f}，balanced accuracy {source['balanced_accuracy']:.4f}。",
        f"- 外部 E_diag Normal vs Lung Opacity：AUROC {external['auroc']:.4f}，accuracy {external['accuracy']:.4f}，balanced accuracy {external['balanced_accuracy']:.4f}。",
        f"- 外部异常但无阴影病例：argmax 假阳性率 {hard_fpr:.4f}。",
        "",
        "## 可解释性结果",
        "",
        f"- 面积校正框内热量均值 {loc['area_corrected_energy_mean']:.4f}；中心先验对照 {loc['center_prior_area_corrected_energy_mean']:.4f}。",
        f"- Grad-CAM 最大响应点命中率 {loc['pointing_hit_rate']:.4f}；零热图比例 {loc['zero_cam_rate']:.4f}。",
        f"- 框内减匹配框外遮挡的分数下降差：均值替换 {occ['mean_inside_minus_outside']:.4f}，模糊替换 {occ['blur_inside_minus_outside']:.4f}。",
        f"- 训练模型与完全随机化模型 CAM 余弦相似度均值 {sanity['trained_vs_fully_randomized_cam_cosine_mean']:.4f}。",
        "",
        "## 已发现的问题",
        "",
    ] + [f"- {item}" for item in issues] + [
        "",
        "## 停点",
        "",
        "已完成复现、内部测试和首次外部诊断。按照预先约定，本阶段到此停止；尚未选择或训练任何改进方法，也未读取 E_final 图像或预测结果。下一阶段应先根据以上问题与原始个案表确定一个主要改进方向。",
    ]
    report = "\n".join(lines) + "\n"
    (out / "问题诊断报告.md").write_text(report, encoding="utf-8")
    baseline.save_json(out / "stop_marker.json", {"status": "stopped_after_reproduction_testing_and_problem_diagnosis", "created_utc": utcnow(), "improvement_training_started": False, "E_final_evaluated": False})
    return report


def run(project_root: Path) -> dict:
    base_root = project_root / "md6329_step10_resnet_reproduction"
    latest = json.loads((base_root / "latest_run.json").read_text(encoding="utf-8"))
    out = base_root / latest["relative_path"]
    internal = json.loads((out / "internal_evaluation.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = baseline.load_selected_model(out, device)
    files = obtain_rsna(base_root)
    records = build_records(files["annotations"], files["mapping"])
    diag, final, split_summary = save_split(out, records)
    members = zip_members(files["images"])
    missing = [r["sop_instance_uid"] for r in diag if r["sop_instance_uid"] not in members]
    if missing:
        raise AssertionError(f"Missing DICOMs, first: {missing[:5]}")
    logits, scores, geometries = predict_diag(model, files["images"], members, diag, device)
    source_threshold = json.loads((out / "training_summary.json").read_text(encoding="utf-8"))["source_youden_threshold"]
    prediction_summary = prediction_metrics(diag, scores, source_threshold)
    np.savez_compressed(out / "external_diag_predictions.npz", logits=logits, scores=scores, targets=np.array([r["target"] for r in diag]), subset_img_ids=np.array([r["subset_img_id"] for r in diag]), patient_ids=np.array([r["patient_id"] for r in diag]))
    rows = []
    for record, score in zip(diag, scores):
        rows.append({"patient_id": record["patient_id"], "subset_img_id": record["subset_img_id"], "class_name": record["class_name"], "target": record["target"], "primary_eligible": record["primary_eligible"], "pneumonia_probability": float(score), "pred_argmax": int(score >= 0.5), "pred_source_youden": int(score >= source_threshold)})
    baseline.save_csv(out / "external_diag_predictions.csv", rows)
    baseline.save_json(out / "external_prediction_summary.json", prediction_summary)
    explanations = explain_external(model, files["images"], members, diag, scores, out, device)
    report = problem_report(out, internal, prediction_summary, explanations, split_summary)
    summary = {
        "status": "stopped_after_reproduction_testing_and_problem_diagnosis",
        "run_id": out.name,
        "external_prediction_summary": prediction_summary,
        "external_explainability_summary": explanations,
        "E_final_evaluated": False,
        "improvement_training_started": False,
        "report_sha256": hashlib.sha256(report.encode()).hexdigest(),
    }
    baseline.save_json(out / "diagnosis_summary.json", summary)
    latest.update(status=summary["status"], diagnosis_summary_sha256=baseline.sha256(out / "diagnosis_summary.json"))
    baseline.save_json(base_root / "latest_run.json", latest)
    return summary


if __name__ == "__main__":
    root = Path(os.environ.get("MD6329_PROJECT_ROOT", Path.cwd()))
    print(json.dumps(run(root), ensure_ascii=False, indent=2))
