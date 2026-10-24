# Data

The repository does not redistribute medical images.

- **PneumoniaMNIST v2:** obtain the official 28×28 and 64×64 releases from [MedMNIST](https://medmnist.com/).
- **RSNA Pneumonia Detection Challenge:** obtain the public image archive, expert annotations and mapping files from the sources documented in `src/external_diagnosis.py`.
- **CheXpert / CheXlocalize:** request access from [Stanford AIMI](https://aimi.stanford.edu/datasets/chexlocalize). The files are subject to their research-use terms and must remain outside Git.

Expected local layout for Colab is `/content/md6329_project`. The complete notebook creates the stage directories it needs. Keep raw images, checkpoints and patient-level predictions in ignored folders.
