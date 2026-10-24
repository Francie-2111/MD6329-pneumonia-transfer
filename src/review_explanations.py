"""Step 19: extend the frozen Step 16 explanation audit to four models.
The original sampling, Grad-CAM target, masks, occlusion, and seeds are retained.
Only four-model artifact export and a matched display are added.
"""
from b64_external_audit import *

def plot_four_way(out, selected, records, images, masks, cams, prediction):
    # Deterministic quantiles of AC-JSD minus plain localization, including failures.
    delta = [ext.energy_and_pointing(cams['augmix'][i], masks[i])[0] -
             ext.energy_and_pointing(cams['plain'][i], masks[i])[0] for i in range(len(selected))]
    order = np.argsort(delta)
    chosen = order[np.linspace(0,len(order)-1,6,dtype=int)]
    names = ['plain','aug1','ce3','augmix']
    fig, ax = plt.subplots(6,5,figsize=(12,13))
    for r,i in enumerate(chosen):
        for c,name in enumerate(['image']+names):
            ax[r,c].imshow(images[i],cmap='gray',vmin=0,vmax=1)
            if name!='image': ax[r,c].imshow(cams[name][i],cmap='jet',alpha=.42,vmin=0,vmax=1)
            ax[r,c].contour(masks[i].astype(float),levels=[.5],colors=['lime'],linewidths=.7)
            title = f'Case {i}; delta={delta[i]:+.3f}' if name=='image' else f'{name}: p={prediction[name]["scores"][selected[i]]:.3f}'
            ax[r,c].set_title(title,fontsize=8); ax[r,c].axis('off')
    fig.tight_layout(); fig.savefig(out/'four_way_gradcam.pdf',bbox_inches='tight'); plt.close(fig)

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
        **{name+"_cams": value for name,value in cams.items()},
    )
    plot_examples(out, selected, records, images, masks_all, cams, prediction)
    plot_four_way(out, selected, records, images, masks_all, cams, prediction)
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
