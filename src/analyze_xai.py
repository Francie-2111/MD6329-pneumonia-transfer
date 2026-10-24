from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from xai_validity_audit import patient_bootstrap, save_csv, NAMES, LABELS, SEED, N_DRAWS

P = Path(__file__).resolve().parent
O = P / "results"
S19 = P.parent / "md6329_step19_review_completion/results"

def load(name, root=O):
    with (root / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

random_cases = load("matched_parameter_randomization_cases.csv")
occlusion = load("margin_occlusion_cases.csv")

# Paired patient-cluster contrasts use identical cases and one common random reference.
patient_ids = np.array([r["patient_id"] for r in random_cases])
contrast_rows = []
for stage in ["shared_fully_random", "linear", "layer4_linear", "layer3_to_linear",
              "layer2_to_linear", "layer1_to_linear", "all"]:
    for a, b in (("augmix", "plain"), ("aug1", "plain"), ("ce3", "plain"),
                 ("augmix", "ce3"), ("ce3", "aug1")):
        values = np.array([[float(r[f"{a}_{stage}_cosine"]) - float(r[f"{b}_{stage}_cosine"])]
                           for r in random_cases])
        point, lo, hi = patient_bootstrap(values, patient_ids, SEED + 4, N_DRAWS)
        contrast_rows.append({"stage": stage, "comparison": f"{a} minus {b}",
                              "estimate": point[0], "ci_low": lo[0], "ci_high": hi[0],
                              "n_images": len(values), "n_patients": len(np.unique(patient_ids)),
                              "draws": N_DRAWS})
save_csv(O / "matched_parameter_randomization_contrasts.csv", contrast_rows)

# Diagnose probability saturation and how closely probability- and margin-drop
# case effects agree. Numerical scales are not compared directly.
saturation = {}
correlation = {}
for name in NAMES:
    clean_p = np.array([float(r[f"{name}_clean_probability"]) for r in occlusion])
    clean_m = np.array([float(r[f"{name}_clean_margin"]) for r in occlusion])
    saturation[name] = {
        "n": len(occlusion), "clean_probability_ge_0p99": int((clean_p >= .99).sum()),
        "clean_probability_equal_1_float32": int((clean_p == 1).sum()),
        "clean_probability_median": float(np.median(clean_p)),
        "clean_margin_median": float(np.median(clean_m)),
    }
    correlation[name] = {}
    for mode in ("mean", "blur"):
        p = np.array([float(r[f"{name}_{mode}_probability_inside_minus_outside"]) for r in occlusion])
        m = np.array([float(r[f"{name}_{mode}_margin_inside_minus_outside"]) for r in occlusion])
        saturation[name][f"{mode}_inside_probability_ge_0p99"] = int(np.sum([
            float(r[f"{name}_clean_probability"]) - float(r[f"{name}_{mode}_probability_inside_drop"]) >= .99
            for r in occlusion]))
        saturation[name][f"clean_and_{mode}_inside_probability_ge_0p99"] = int(np.sum([
            float(r[f"{name}_clean_probability"]) >= .99 and
            float(r[f"{name}_clean_probability"]) - float(r[f"{name}_{mode}_probability_inside_drop"]) >= .99
            for r in occlusion]))
        correlation[name][mode] = {
            "spearman_probability_vs_margin_case_effect": float(spearmanr(p, m).statistic),
            "same_sign_fraction": float(np.mean(np.sign(p) == np.sign(m))),
            "probability_near_zero_abs_le_1e-4": int((np.abs(p) <= 1e-4).sum()),
            "margin_near_zero_abs_le_1e-3": int((np.abs(m) <= 1e-3).sum()),
        }

# Selection bias of matched translated masks: excluded boxes are much larger.
saved = np.load(S19 / "E_final_B64_explanations.npz")
selected = saved["selected_indices"].astype(int)
prediction = np.load(S19 / "four_way_predictions.npz")
included = {r["subset_img_id"] for r in occlusion}
area = saved["masks"][:120].mean((1, 2))
ids = prediction["subset_img_ids"][selected[:120]].astype(str)
in_area = area[np.isin(ids, list(included))]
out_area = area[~np.isin(ids, list(included))]
eligibility = {
    "included_n": len(in_area), "excluded_n": len(out_area),
    "included_box_area_median": float(np.median(in_area)),
    "excluded_box_area_median": float(np.median(out_area)),
    "included_box_area_iqr": np.quantile(in_area, [.25, .75]).tolist(),
    "excluded_box_area_iqr": np.quantile(out_area, [.25, .75]).tolist(),
}

diagnostics = {"saturation": saturation, "probability_margin_association": correlation,
               "occlusion_eligibility": eligibility,
               "interpretation": "Margin analysis diagnoses softmax compression; it does not replace probability effects or prove causality."}
(O / "supplementary_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf8")

fig, axes = plt.subplots(2, 4, figsize=(9, 4.5), sharex="row")
for col, name in enumerate(NAMES):
    for row_i, mode in enumerate(("mean", "blur")):
        p = np.array([float(r[f"{name}_{mode}_probability_inside_minus_outside"]) for r in occlusion])
        m = np.array([float(r[f"{name}_{mode}_margin_inside_minus_outside"]) for r in occlusion])
        axes[row_i, col].scatter(p, m, s=9, alpha=.55)
        axes[row_i, col].axhline(0, color=".7", lw=.7); axes[row_i, col].axvline(0, color=".7", lw=.7)
        axes[row_i, col].set_title(LABELS[name] + ("; mean" if row_i == 0 else "; blur"), fontsize=8)
        if col == 0: axes[row_i, col].set_ylabel("Margin effect")
        if row_i == 1: axes[row_i, col].set_xlabel("Probability effect")
        axes[row_i, col].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(O / "probability_vs_margin_occlusion.pdf")
fig.savefig(O / "probability_vs_margin_occlusion.png", dpi=180)
plt.close(fig)

random_intervals = load("matched_parameter_randomization_intervals.csv")
stages = ["linear", "layer4_linear", "layer3_to_linear", "layer2_to_linear", "layer1_to_linear", "all"]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
for name in NAMES:
    model_rows = {r["stage"]: r for r in random_intervals if r["model"] == name}
    ax.plot(range(len(stages)), [float(model_rows[s]["mean_cosine"]) for s in stages],
            marker="o", label=LABELS[name])
ax.set_xticks(range(len(stages)), ["Head", "L4+", "L3+", "L2+", "L1+", "All"])
ax.set_ylabel("CAM cosine vs trained model"); ax.set_xlabel("Cumulative randomized modules")
ax.set_ylim(-.05, 1.02); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(O / "matched_parameter_randomization.png", dpi=180); plt.close(fig)
print(json.dumps(diagnostics, indent=2))
