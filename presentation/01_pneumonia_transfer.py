"""Ten-minute LecTrace presentation for the MD6329 mini-project.

The trace uses a compact teaching example for single-case calculations and the
recorded cohort-level outputs from the full experiment notebook for conclusions.
"""

from __future__ import annotations

import math

from lectrace import image, link, note, plot, table, text
from _data import EXTERNAL_RESULTS, INTERNAL_RESULTS, XAI_RESULTS


def main():
    text("# External Transfer and Spatial Evidence in Pneumonia Classification")
    text("""
        **CUI SHIHAN · G2602393H · MD6329**

        Can higher resolution and acquisition-style consistency improve external
        pneumonia classification while making spatial evidence more lesion-aligned?
    """)
    text("""
        ## Why this healthcare problem matters

        A chest-radiograph classifier can score highly on its source test set yet fail
        at another hospital because patient populations, acquisition settings and label
        definitions change. High accuracy also does not tell us whether the network used
        pulmonary opacity or an unrelated shortcut.
    """)
    text("""
        ## Study design

        We reproduce the 28-pixel PneumoniaMNIST ResNet18, isolate the effect of native
        64-pixel input, then compare plain training with acquisition-style augmentation
        and Jensen–Shannon consistency. Frozen models are evaluated on RSNA and, once,
        on CheXpert. Grad-CAM, randomization and matched occlusion examine spatial evidence.
    """)
    table(
        [
            {"role": "Source", "dataset": "PneumoniaMNIST", "population": "Pediatric", "use": "Training and source evaluation"},
            {"role": "External 1", "dataset": "RSNA", "population": "Adult", "use": "Classification + opacity boxes"},
            {"role": "External 2", "dataset": "CheXpert", "population": "Adult", "use": "Final one-shot classification"},
        ],
        caption="Three-domain evaluation design",
    )
    note("Speaking time: about 2–3 minutes. Frame the project as external transfer plus spatial evidence, not only source accuracy.")

    resolution = state_1_resolution()  # @inspect resolution
    resolution  # @clear resolution @invisible
    forward = state_2_forward_pass()  # @inspect forward
    forward  # @clear forward @invisible
    consistency = state_3_consistency()  # @inspect consistency
    consistency  # @clear consistency @invisible
    external = state_4_external_transfer()  # @inspect external
    external  # @clear external @invisible
    spatial = state_5_spatial_evidence()  # @inspect spatial
    spatial  # @clear spatial @invisible

    text("""
        # Key result and healthcare interpretation

        1. **Resolution explains the largest fixed-threshold accuracy gain.** Moving from
           B28 to B64 raised RSNA accuracy by 8.68 percentage points.
        2. **Consistency mainly improves external ranking.** AC-JSD raised RSNA AUROC from
           0.8720 to 0.9012, while the accuracy interval included zero.
        3. **The final external result transferred.** On CheXpert, AUROC rose from 0.7212
           to 0.9310 and accuracy from 62.24% to 83.92%.
        4. **Spatial evidence improved relative to plain training, but was not conclusive.**
           Aug1 had the best box alignment, and a simple center prior remained stronger.

        The healthcare lesson is that a model should be evaluated across institutions and
        with complementary spatial tests. A better heatmap supports interpretation, but it
        does not by itself establish that the model has learned a causal disease mechanism.
    """)
    note("End at about 10 minutes. Return to the research question: transfer improved, and spatial evidence improved modestly rather than decisively.")


def state_1_resolution():
    text("""
        ## State 1 — The input resolution changes the available evidence

        The architecture is held fixed. The controlled change is the native input from
        $28\times28$ to $64\times64$. This tests whether later gains simply come from
        retaining more radiographic detail.
    """)
    input_28 = {"shape": [1, 1, 28, 28], "pixels": 28 * 28}
    input_64 = {"shape": [1, 1, 64, 64], "pixels": 64 * 64}
    pixel_ratio = input_64["pixels"] / input_28["pixels"]
    internal_accuracy_gain_pp = 100 * (INTERNAL_RESULTS[1]["accuracy"] - INTERNAL_RESULTS[0]["accuracy"])
    rsna_accuracy_gain_pp = 100 * (0.8057 - 0.7189)
    rsna_auroc_gain = 0.8720 - 0.8379
    table(
        [
            {"comparison": "Input pixels", "B28": input_28["pixels"], "B64": input_64["pixels"], "change": f"{pixel_ratio:.2f}×"},
            {"comparison": "Internal accuracy", "B28": 0.8622, "B64": 0.9054, "change": f"+{internal_accuracy_gain_pp:.2f} pp"},
            {"comparison": "RSNA accuracy", "B28": 0.7189, "B64": 0.8057, "change": f"+{rsna_accuracy_gain_pp:.2f} pp"},
            {"comparison": "RSNA AUROC", "B28": 0.8379, "B64": 0.8720, "change": f"+{rsna_auroc_gain:.4f}"},
        ],
        caption="Resolution ablation: same model family, independently fitted",
    )
    plot(metric_chart(
        "Resolution ablation",
        [
            {"split": "Internal", "model": "B28", "metric": "Accuracy", "value": 0.8622},
            {"split": "Internal", "model": "B64 plain", "metric": "Accuracy", "value": 0.9054},
            {"split": "RSNA", "model": "B28", "metric": "Accuracy", "value": 0.7189},
            {"split": "RSNA", "model": "B64 plain", "metric": "Accuracy", "value": 0.8057},
        ],
        x="split",
        color="model",
    ))  # @stepover
    note("What changes: spatial sampling only. Why it matters: this is the largest accuracy gain, so resolution must be separated from the later training-method gain.")
    return {"pixel_ratio": round(pixel_ratio, 2), "internal_gain_pp": round(internal_accuracy_gain_pp, 2), "rsna_gain_pp": round(rsna_accuracy_gain_pp, 2)}


def state_2_forward_pass():
    text("""
        ## State 2 — A ResNet18 forward pass becomes a clinical score

        A one-channel image passes through a $3\times3$ stem and four residual stages.
        Global average pooling produces a 512-dimensional representation, and the linear
        head emits two logits: normal and pneumonia.
    """)
    architecture = [
        {"stage": "Input", "shape": "1 × 1 × 64 × 64", "meaning": "Normalized grayscale radiograph"},
        {"stage": "Residual encoder", "shape": "1 × 512 × 2 × 2", "meaning": "Learned spatial features"},
        {"stage": "Global average pool", "shape": "1 × 512", "meaning": "Image representation"},
        {"stage": "Linear head", "shape": "1 × 2", "meaning": "Normal and pneumonia logits"},
    ]
    table(architecture, caption="Forward path used by every experimental condition")
    logits = [-0.80, 1.60]
    probabilities = softmax(logits)
    logit_margin = logits[1] - logits[0]
    pneumonia_probability = probabilities[1]
    predicted_class = "pneumonia" if pneumonia_probability >= 0.5 else "normal"
    table(
        [
            {"quantity": "Normal logit", "value": round(logits[0], 4)},
            {"quantity": "Pneumonia logit", "value": round(logits[1], 4)},
            {"quantity": "Logit margin", "value": round(logit_margin, 4)},
            {"quantity": "P(pneumonia)", "value": round(pneumonia_probability, 4)},
            {"quantity": "Thresholded output", "value": predicted_class},
        ],
        caption="Compact teaching example of the exact inference calculation",
    )
    note("What changes: features become logits, then probabilities. Why it matters: AUROC uses score ranking, while accuracy additionally depends on the 0.5 threshold.")
    return {"logits": logits, "margin": round(logit_margin, 4), "pneumonia_probability": round(pneumonia_probability, 4), "prediction": predicted_class}


def state_3_consistency():
    text("""
        ## State 3 — Acquisition-style views expose prediction instability

        For each source image, training creates two mild views using brightness,
        contrast, gamma, rotation, translation, blur or low Gaussian noise. Crops and
        horizontal flips are excluded because they can alter anatomy.
    """)
    view_parameters = [
        {"view": "Clean", "brightness": 1.00, "contrast": 1.00, "rotation": "0°"},
        {"view": "View 1", "brightness": 0.92, "contrast": 1.12, "rotation": "+5°"},
        {"view": "View 2", "brightness": 1.08, "contrast": 0.88, "rotation": "−4°"},
    ]
    table(view_parameters, caption="Illustrative views within the prespecified augmentation ranges")
    plain_predictions = [[0.10, 0.90], [0.38, 0.62], [0.18, 0.82]]
    consistent_predictions = [[0.12, 0.88], [0.16, 0.84], [0.10, 0.90]]
    plain_jsd = jensen_shannon(plain_predictions)
    consistent_jsd = jensen_shannon(consistent_predictions)  # @stepover
    plain_ce = mean_cross_entropy(plain_predictions, target=1)  # @stepover
    consistent_ce = mean_cross_entropy(consistent_predictions, target=1)  # @stepover
    selected_lambda = 36
    plain_total = plain_ce + selected_lambda * plain_jsd
    consistent_total = consistent_ce + selected_lambda * consistent_jsd
    table(
        [
            {"example": "Unstable predictions", "mean CE": round(plain_ce, 4), "JSD": round(plain_jsd, 4), "CE + 36×JSD": round(plain_total, 4)},
            {"example": "Consistent predictions", "mean CE": round(consistent_ce, 4), "JSD": round(consistent_jsd, 4), "CE + 36×JSD": round(consistent_total, 4)},
        ],
        caption="How the consistency term penalizes disagreement",
    )
    plot(metric_chart(
        "Pneumonia probability across views",
        [
            {"view": "Clean", "model": "Unstable", "metric": "Pneumonia probability", "value": 0.90},
            {"view": "View 1", "model": "Unstable", "metric": "Pneumonia probability", "value": 0.62},
            {"view": "View 2", "model": "Unstable", "metric": "Pneumonia probability", "value": 0.82},
            {"view": "Clean", "model": "Consistent", "metric": "Pneumonia probability", "value": 0.88},
            {"view": "View 1", "model": "Consistent", "metric": "Pneumonia probability", "value": 0.84},
            {"view": "View 2", "model": "Consistent", "metric": "Pneumonia probability", "value": 0.90},
        ],
        x="view",
        color="model",
    ))  # @stepover
    text(r"""
        The training objective is
        $$
        \mathcal L=\frac{1}{3}\sum_{j=0}^{2}\mathrm{CE}(z_j,y)
        +\lambda\frac{1}{3}\sum_{j=0}^{2}D_{KL}(p_j\Vert\bar p),
        \qquad \bar p=\frac{p_0+p_1+p_2}{3}.
        $$
        The final source-only sweep selected $\lambda=36$ by validation Brier score.
    """)
    link(
        title="AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty",
        url="https://arxiv.org/abs/1912.02781",
        authors=["Hendrycks", "Mu", "Cubuk", "Zoph", "Gilmer", "Lakshminarayanan"],
        date="2020",
        description="The consistency principle that motivated AC-JSD.",
    )
    note("What changes: the same underlying image receives different acquisition-style views. Why it matters: JSD explicitly discourages predictions from changing with these nuisance factors.")
    return {"lambda": selected_lambda, "unstable_jsd": round(plain_jsd, 4), "consistent_jsd": round(consistent_jsd, 4)}


def state_4_external_transfer():
    text("""
        ## State 4 — Frozen models are tested outside the source domain

        No RSNA or CheXpert labels are used for fitting the reported checkpoints. The
        main question is whether source-only consistency survives a population and
        acquisition shift.
    """)
    table(EXTERNAL_RESULTS, caption="Recorded external results from the complete experiment")
    rsna_gain = compare_models(EXTERNAL_RESULTS[0], EXTERNAL_RESULTS[1])
    chexpert_gain = compare_models(EXTERNAL_RESULTS[2], EXTERNAL_RESULTS[3])  # @stepover
    uncertainty = [
        {"endpoint": "RSNA accuracy difference", "estimate": "+0.0036", "95% interval": "[−0.0031, 0.0103]", "interpretation": "Includes zero"},
        {"endpoint": "RSNA AUROC difference", "estimate": "+0.0291", "95% interval": "[0.0255, 0.0330]", "interpretation": "Clear ranking gain"},
        {"endpoint": "CheXpert AUROC difference", "estimate": "+0.2097", "95% interval": "[0.1215, 0.3022]", "interpretation": "Positive one-shot result"},
    ]
    table(uncertainty, caption="Paired uncertainty for AC-JSD minus B64 plain")
    plot(metric_chart(
        "External AUROC",
        [
            {"dataset": "RSNA", "model": "Plain", "metric": "AUROC", "value": 0.8720},
            {"dataset": "RSNA", "model": "AC-JSD", "metric": "AUROC", "value": 0.9012},
            {"dataset": "CheXpert", "model": "Plain", "metric": "AUROC", "value": 0.7212},
            {"dataset": "CheXpert", "model": "AC-JSD", "metric": "AUROC", "value": 0.9310},
        ],
        x="dataset",
        color="model",
    ))  # @stepover
    note("What changes: the hospital and population. Why it matters: AC-JSD improves ranking on both external datasets, while the small RSNA accuracy change is threshold-dependent and uncertain.")
    return {"rsna": rsna_gain, "chexpert": chexpert_gain}


def state_5_spatial_evidence():
    text("""
        ## State 5 — Does the model look closer to the annotated opacity?

        Grad-CAM targets the pneumonia logit at the last residual stage. On the same
        300 RSNA positive images, its energy inside public opacity boxes is compared
        with box area, and the maximum-activation point is tested against the box.
    """)
    example_energy = area_corrected_energy(inside_energy=0.34, total_energy=1.00, box_area_fraction=0.25)
    text(r"""
        Area-corrected energy removes the score expected from box size alone:
        $$E(A,M)=\frac{\sum_{u\in M}A_u}{\sum_u A_u+\epsilon}-\frac{|M|}{HW}.$$
    """)
    table(
        [{"inside heatmap energy": 0.34, "box area fraction": 0.25, "area-corrected energy": round(example_energy, 4)}],
        caption="Small executable example of the localization metric",
    )
    table(XAI_RESULTS, caption="Recorded RSNA explanation results on matched cases")
    image("figures/external_xai.png", width="82%")
    energy_gain = -0.0162 - (-0.0336)
    pointing_gain = 0.0867 - 0.0667
    interpretation = {
        "AC-JSD energy gain vs plain": round(energy_gain, 4),
        "AC-JSD pointing gain vs plain": round(pointing_gain, 4),
        "best box alignment": "Aug1",
        "stronger reference": "Center prior",
    }
    link(
        title="Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
        url="https://arxiv.org/abs/1610.02391",
        authors=["Selvaraju", "Cogswell", "Das", "Vedantam", "Parikh", "Batra"],
        date="2017",
    )
    note("What changes: model weights change where Grad-CAM places energy. Why it matters: AC-JSD improves average alignment over plain, but Aug1 and the center prior prevent an overclaim of lesion localization.")
    return interpretation


def softmax(logits):
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    return probabilities


def mean_cross_entropy(predictions, target):
    losses = [-math.log(max(distribution[target], 1e-12)) for distribution in predictions]
    mean_loss = sum(losses) / len(losses)
    return mean_loss


def jensen_shannon(predictions):
    mean_distribution = [sum(view[k] for view in predictions) / len(predictions) for k in range(len(predictions[0]))]
    divergences = []
    for distribution in predictions:
        divergence = sum(p * math.log(p / mean_distribution[k]) for k, p in enumerate(distribution) if p > 0)
        divergences.append(divergence)
    jsd = sum(divergences) / len(divergences)
    return jsd


def compare_models(plain, improved):
    accuracy_difference = improved["accuracy"] - plain["accuracy"]
    auroc_difference = improved["auroc"] - plain["auroc"]
    return {
        "dataset": plain["dataset"],
        "accuracy_difference": round(accuracy_difference, 4),
        "auroc_difference": round(auroc_difference, 4),
    }


def area_corrected_energy(inside_energy, total_energy, box_area_fraction):
    observed_fraction = inside_energy / (total_energy + 1e-12)
    corrected_energy = observed_fraction - box_area_fraction
    return corrected_energy


def metric_chart(title, values, x, color):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 520,
        "height": 250,
        "data": {"values": values},
        "mark": {"type": "bar", "tooltip": True},
        "encoding": {
            "x": {"field": x, "type": "nominal", "axis": {"labelAngle": 0}},
            "xOffset": {"field": color},
            "y": {"field": "value", "type": "quantitative", "scale": {"domain": [0.5, 1.0]}},
            "color": {"field": color, "type": "nominal"},
            "tooltip": [
                {"field": x, "type": "nominal"},
                {"field": color, "type": "nominal"},
                {"field": "value", "type": "quantitative", "format": ".4f"},
            ],
        },
    }


if __name__ == "__main__":
    main()
