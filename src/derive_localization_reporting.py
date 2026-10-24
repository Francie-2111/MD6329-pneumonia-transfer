"""Derive a nonnegative Grad-CAM reporting metric from frozen case-level outputs.

This script does not rerun a model or alter any heatmap. It expresses the same
Grad-CAM result as the fraction of total heatmap energy inside the union of the
public opacity boxes, then computes patient-cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def clustered_summary(rows: list[dict], draws: int = 5000, seed: int = 632920) -> dict:
    patient_ids = np.asarray([row["patient_id"] for row in rows])
    _, inverse = np.unique(patient_ids, return_inverse=True)
    n_patients = int(inverse.max() + 1)

    box_area = np.asarray([float(row["box_area_fraction"]) for row in rows])
    plain_share = np.asarray(
        [float(row["plain_area_corrected_energy"]) for row in rows]
    ) + box_area
    acjsd_share = np.asarray(
        [float(row["augmix_area_corrected_energy"]) for row in rows]
    ) + box_area

    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 4), dtype=float)
    for draw in range(draws):
        cluster_weights = np.bincount(
            rng.integers(0, n_patients, size=n_patients), minlength=n_patients
        )[inverse]
        denominator = cluster_weights.sum()
        plain_mean = float(np.sum(cluster_weights * plain_share) / denominator)
        acjsd_mean = float(np.sum(cluster_weights * acjsd_share) / denominator)
        area_mean = float(np.sum(cluster_weights * box_area) / denominator)
        samples[draw] = [
            acjsd_mean,
            area_mean,
            acjsd_mean - plain_mean,
            100.0 * (acjsd_mean / plain_mean - 1.0),
        ]

    point = np.asarray(
        [
            acjsd_share.mean(),
            box_area.mean(),
            (acjsd_share - plain_share).mean(),
            100.0 * (acjsd_share.mean() / plain_share.mean() - 1.0),
        ]
    )
    labels = [
        "acjsd_in_box_energy_share",
        "mean_box_area_fraction",
        "absolute_share_gain_vs_plain",
        "relative_share_gain_percent_vs_plain",
    ]
    summary = {
        label: {
            "estimate": float(point[index]),
            "ci95": np.quantile(samples[:, index], [0.025, 0.975]).tolist(),
        }
        for index, label in enumerate(labels)
    }
    summary.update(
        {
            "n_images": len(rows),
            "n_patients": n_patients,
            "bootstrap_draws": draws,
            "definition": "sum of Grad-CAM energy inside union opacity boxes divided by total Grad-CAM energy",
            "interpretation": (
                "This nonnegative descriptive metric is derived from the same frozen heatmaps. "
                "The mean box-area fraction remains the area-only reference; the metric does not "
                "establish causal lesion recognition."
            ),
        }
    )
    return summary


def make_figure(summary: dict, output: Path) -> None:
    share = summary["acjsd_in_box_energy_share"]
    area = summary["mean_box_area_fraction"]
    gain = summary["relative_share_gain_percent_vs_plain"]

    blue = "#1677b8"
    orange = "#c65a25"
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7))

    estimate = 100.0 * share["estimate"]
    low, high = [100.0 * value for value in share["ci95"]]
    axes[0].bar([0], [estimate], width=0.48, color=blue)
    axes[0].errorbar(
        [0], [estimate], yerr=[[estimate - low], [high - estimate]],
        fmt="none", ecolor="#16334f", capsize=6, linewidth=1.8,
    )
    axes[0].axhline(100.0 * area["estimate"], color="#55724b", linestyle="--", linewidth=1.5)
    axes[0].text(0, estimate / 2, f"{estimate:.2f}%", ha="center", va="center", color="white", fontsize=15, weight="bold")
    axes[0].text(0.38, 100.0 * area["estimate"] + 0.35, "Mean annotated area 12.26%", ha="right", color="#3f5f38", fontsize=9)
    axes[0].set_xticks([0], ["AC-JSD"])
    axes[0].set_ylim(0, 16)
    axes[0].set_ylabel("Grad-CAM energy inside opacity boxes (%)")
    axes[0].set_title("Absolute in-box energy share")

    gain_estimate = gain["estimate"]
    gain_low, gain_high = gain["ci95"]
    axes[1].barh([0], [gain_estimate], height=0.42, color=orange)
    axes[1].errorbar(
        [gain_estimate], [0], xerr=[[gain_estimate - gain_low], [gain_high - gain_estimate]],
        fmt="none", ecolor="#5f2c18", capsize=6, linewidth=1.8,
    )
    axes[1].text(gain_estimate / 2, 0, f"+{gain_estimate:.1f}%", ha="center", va="center", color="white", fontsize=15, weight="bold")
    axes[1].set_xlim(0, 32)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Relative change in in-box energy share (%)")
    axes[1].set_title("Paired gain over matched plain training")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is axes[0] else "x", color="#d9e2ea", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

    fig.suptitle("Grad-CAM spatial evidence on 300 matched RSNA-positive images", fontsize=13, weight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_csv", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    with args.case_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = clustered_summary(rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(summary, args.figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
