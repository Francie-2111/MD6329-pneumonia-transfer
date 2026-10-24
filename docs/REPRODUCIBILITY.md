# Reproducing the experiments

1. Open `notebooks/MD6329_Final_Complete_Experiment_Notebook.ipynb` in Colab.
2. Place the repository at `/content/md6329_project` or update `PROJECT_ROOT` in the first code cell.
3. Install `requirements.txt` and place datasets outside Git as described in `DATA.md`.
4. Run the notebook in order. Training and external evaluation are controlled by the `RUN_*` switches.
5. For the final source comparison, run plain, Aug1, CE3 and AC-JSD with λ in `{1, 3, 12, 36}`.
6. Use the saved best-Brier plain and selected AC-JSD checkpoints for CheXpert evaluation.

The published results use seed 632901. Paired bootstrap intervals use 5,000 draws. External RSNA resampling is patient-clustered.
