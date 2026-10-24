# Line-by-line LecTrace speaking script

This speaker document is separate from the LecTrace viewer. Each row corresponds to one press of the right arrow in the 99-step trace.

| Step | LecTrace line | Function | What to say |
|---:|---:|---|---|
| 1 | 15 | `main` | The presentation begins inside `main`, which controls the five selected execution states. |
| 2 | 16 | `main` | My project studies external transfer and spatial evidence in pneumonia classification. |
| 3 | 17 | `main` | I am CUI SHIHAN. The question is whether higher resolution and acquisition-style consistency improve transfer, and how much spatial evidence overlaps annotated opacity. |
| 4 | 23 | `main` | The clinical motivation is that a classifier can perform well on its source hospital yet fail elsewhere, and accuracy alone cannot reveal shortcut learning. |
| 5 | 31 | `main` | The design has three stages: reproduce the baseline, isolate resolution, then test augmentation plus Jensen–Shannon consistency with frozen external evaluation. |
| 6 | 39 | `main` | PneumoniaMNIST is the pediatric source, RSNA supplies an adult external task and opacity boxes, and CheXpert is the final one-shot external cohort. |
| 7 | 47 | `main` | I now enter the first selected state: the resolution ablation. |
| 8 | 77 | `state_1_resolution` | This function keeps the model family fixed and changes only native input resolution. |
| 9 | 78 | `state_1_resolution` | The first state asks how much evidence is retained at 28 versus 64 pixels. |
| 10 | 85 | `state_1_resolution` | The B28 tensor contains one grayscale channel and 784 pixels. |
| 11 | 86 | `state_1_resolution` | The B64 tensor has the same batch and channel structure but 4,096 pixels. |
| 12 | 87 | `state_1_resolution` | Dividing those counts shows that B64 retains 5.22 times as many sampled pixels. |
| 13 | 88 | `state_1_resolution` | On the internal test set, accuracy rises by 4.32 percentage points. |
| 14 | 89 | `state_1_resolution` | On RSNA, accuracy rises by 8.68 percentage points. |
| 15 | 90 | `state_1_resolution` | RSNA AUROC also rises by 0.0341. |
| 16 | 91 | `state_1_resolution` | This table puts the controlled input change and the three measured gains side by side. |
| 17 | 100 | `state_1_resolution` | The chart makes the important pattern visible: the resolution change already explains most of the fixed-threshold accuracy improvement. |
| 18 | 111 | `state_1_resolution` | This matters because later gains cannot be attributed to AC-JSD unless resolution is controlled first. |
| 19 | 112 | `state_1_resolution` | The state returns only the three quantities that we need to carry forward. |
| 20 | 49 | `main` | The second selected state follows one B64 prediction from representation to decision. |
| 21 | 115 | `state_2_forward_pass` | We now enter the forward-pass function. |
| 22 | 116 | `state_2_forward_pass` | The model is a one-channel ResNet18 with four residual stages, global average pooling and a two-logit head. |
| 23 | 123 | `state_2_forward_pass` | These four records summarize the tensor shape at each major stage. |
| 24 | 129 | `state_2_forward_pass` | The table shows how a 64-pixel image becomes a 512-dimensional representation and finally two class scores. |
| 25 | 130 | `state_2_forward_pass` | For a compact teaching case, the normal and pneumonia logits are minus 0.8 and 1.6. |
| 26 | 131 | `state_2_forward_pass` | The logits are passed to softmax to obtain normalized class probabilities. |
| 27 | 282 | `softmax` | Inside softmax, the input is the two raw scores. |
| 28 | 283 | `softmax` | Subtracting the maximum keeps the exponential calculation numerically stable. |
| 29 | 284 | `softmax` | Exponentiating converts relative logit differences into positive weights. |
| 30 | 285 | `softmax` | Their sum is the normalization denominator. |
| 31 | 286 | `softmax` | Dividing by the denominator produces probabilities that sum to one. |
| 32 | 287 | `softmax` | Softmax returns those two probabilities to the forward pass. |
| 33 | 132 | `state_2_forward_pass` | The pneumonia logit exceeds the normal logit by 2.4; this margin is also useful for occlusion analysis. |
| 34 | 133 | `state_2_forward_pass` | The pneumonia probability is 0.9168. |
| 35 | 134 | `state_2_forward_pass` | At the prespecified 0.5 threshold, this example is classified as pneumonia. |
| 36 | 135 | `state_2_forward_pass` | The table connects logits, margin, probability and final class in one view. |
| 37 | 145 | `state_2_forward_pass` | This distinction matters: AUROC uses continuous ranking, whereas accuracy also depends on the chosen threshold. |
| 38 | 146 | `state_2_forward_pass` | The forward state returns the compact prediction summary shown in the variable panel. |
| 39 | 51 | `main` | The third state introduces the training change: consistency across acquisition-style views. |
| 40 | 149 | `state_3_consistency` | We now enter the AC-JSD calculation. |
| 41 | 150 | `state_3_consistency` | The same anatomy is shown as a clean image and two mildly transformed views; crops and flips are excluded. |
| 42 | 157 | `state_3_consistency` | These records specify brightness, contrast and rotation for the teaching example. |
| 43 | 162 | `state_3_consistency` | The table confirms that all transformations stay within the prespecified mild ranges. |
| 44 | 163 | `state_3_consistency` | The first prediction set changes substantially across views, with pneumonia probability falling from 0.90 to 0.62. |
| 45 | 164 | `state_3_consistency` | The second set remains close across views, between 0.84 and 0.90. |
| 46 | 165 | `state_3_consistency` | We calculate Jensen–Shannon divergence for the unstable set. |
| 47 | 296 | `jensen_shannon` | This helper exposes the exact consistency calculation. |
| 48 | 297 | `jensen_shannon` | First, average the three predictive distributions to form the reference distribution. |
| 49 | 298 | `jensen_shannon` | Then calculate KL divergence from the clean prediction to that reference. |
| 50 | 299 | `jensen_shannon` | Repeat the KL calculation for the first augmented view. |
| 51 | 300 | `jensen_shannon` | Repeat it for the second augmented view. |
| 52 | 301 | `jensen_shannon` | Averaging the three KL terms gives the Jensen–Shannon divergence. |
| 53 | 302 | `jensen_shannon` | The unstable-set JSD is returned to the training state. |
| 54 | 166 | `state_3_consistency` | The same calculation is applied to the consistent set, but stepped over because the arithmetic is identical. |
| 55 | 167 | `state_3_consistency` | Mean cross-entropy is calculated for the unstable predictions. |
| 56 | 168 | `state_3_consistency` | The same label loss is calculated for the consistent predictions. |
| 57 | 169 | `state_3_consistency` | Source-only validation selected lambda equal to 36 using Brier score. |
| 58 | 170 | `state_3_consistency` | For the unstable set, the JSD penalty is added to cross-entropy and raises the total loss. |
| 59 | 171 | `state_3_consistency` | For the consistent set, the smaller JSD contributes a much smaller penalty. |
| 60 | 172 | `state_3_consistency` | The table quantifies why the optimizer prefers stable predictions for the same anatomy. |
| 61 | 179 | `state_3_consistency` | The chart shows the same effect visually: the consistent probabilities stay close across all three views. |
| 62 | 192 | `state_3_consistency` | This is the exact objective used in training: mean cross-entropy plus lambda times mean Jensen–Shannon divergence. |
| 63 | 201 | `state_3_consistency` | The linked AugMix paper supplies the consistency principle; this project uses milder radiograph-specific transformations rather than its mixture construction. |
| 64 | 208 | `state_3_consistency` | This state matters because acquisition changes should be treated as nuisance variation, not as evidence for pneumonia. |
| 65 | 209 | `state_3_consistency` | The state returns the selected lambda and both divergence values. |
| 66 | 53 | `main` | The fourth state changes the hospital and population while keeping checkpoints frozen. |
| 67 | 212 | `state_4_external_transfer` | We now enter external transfer evaluation. |
| 68 | 213 | `state_4_external_transfer` | Neither RSNA nor CheXpert labels are used to fit these reported checkpoints. |
| 69 | 220 | `state_4_external_transfer` | The table contains the recorded external accuracy and AUROC from the complete notebook. |
| 70 | 221 | `state_4_external_transfer` | We first compare AC-JSD with plain training on RSNA. |
| 71 | 305 | `compare_models` | This helper computes paired model differences. |
| 72 | 306 | `compare_models` | RSNA accuracy changes by only about 0.0036. |
| 73 | 307 | `compare_models` | RSNA AUROC increases by about 0.0291. |
| 74 | 308 | `compare_models` | The helper returns both differences as one external-transfer summary. |
| 75 | 222 | `state_4_external_transfer` | The same subtraction gives the much larger CheXpert gains, and we step over repeated arithmetic. |
| 76 | 223 | `state_4_external_transfer` | These intervals were specified before interpreting the final contrast. |
| 77 | 228 | `state_4_external_transfer` | The RSNA accuracy interval includes zero, but the RSNA and CheXpert AUROC intervals are clearly positive. |
| 78 | 229 | `state_4_external_transfer` | The chart highlights that AC-JSD improves score ranking on both external datasets. |
| 79 | 240 | `state_4_external_transfer` | This matters clinically because ranking transfers more consistently than a single fixed threshold; operating thresholds still require local calibration. |
| 80 | 241 | `state_4_external_transfer` | The state returns separate summaries for RSNA and CheXpert. |
| 81 | 55 | `main` | The fifth state audits where the final model's Grad-CAM evidence falls. |
| 82 | 244 | `state_5_spatial_evidence` | We now enter the spatial-evidence audit. |
| 83 | 245 | `state_5_spatial_evidence` | Grad-CAM targets the pneumonia logit on the same 300 RSNA-positive images. |
| 84 | 252 | `state_5_spatial_evidence` | A small executable example shows the nonnegative in-box energy-share definition. |
| 85 | 315 | `energy_in_box_share` | This helper receives the Grad-CAM energy inside the union of opacity boxes and the total heatmap energy. |
| 86 | 316 | `energy_in_box_share` | Dividing inside energy by total energy gives a bounded share between zero and one. |
| 87 | 317 | `energy_in_box_share` | The helper returns that share without changing the saved heatmap. |
| 88 | 253 | `state_5_spatial_evidence` | The displayed equation applies the same definition to every pixel in the heatmap and mask. |
| 89 | 257 | `state_5_spatial_evidence` | The example table verifies the arithmetic before the cohort-level result. |
| 90 | 261 | `state_5_spatial_evidence` | Across 300 matched images, AC-JSD places 10.65 percent of Grad-CAM energy inside the annotated opacity boxes. |
| 91 | 262 | `state_5_spatial_evidence` | The figure shows a patient-bootstrap interval of 9.42 to 11.92 percent and a 19.6 percent paired relative gain. |
| 92 | 263 | `state_5_spatial_evidence` | This variable records the final model's absolute in-box energy share. |
| 93 | 264 | `state_5_spatial_evidence` | This variable records the paired relative gain over the matched plain-training reference. |
| 94 | 265 | `state_5_spatial_evidence` | The mean opacity-box area is 12.26 percent, which gives the essential area-only context. |
| 95 | 266 | `state_5_spatial_evidence` | The interpretation object collects the final absolute value, relative gain, area context and role of the audit. |
| 96 | 272 | `state_5_spatial_evidence` | This link identifies the Grad-CAM method used to construct the class-specific maps. |
| 97 | 278 | `state_5_spatial_evidence` | The result provides a compact spatial audit, while the area reference limits the strength of any localization claim. |
| 98 | 279 | `state_5_spatial_evidence` | The state returns the restrained interpretation: spatial evidence supports the analysis, while external transfer remains the main contribution. |
| 99 | 58 | `main` | The conclusion is: resolution explains the largest fixed-threshold accuracy gain; AC-JSD improves external ranking and transfers strongly to CheXpert. Its in-box Grad-CAM energy is 10.65 percent, a 19.6 percent relative gain, while the 12.26 percent mean box area shows that localization remains limited. |
