# Project Development Files

This directory contains the complete development pipeline for the Embedded and
Edge AI project: exploratory analysis, preprocessing experiments, model
training, cross-validation, reusable Python modules, experiment metadata, and
trained checkpoints.

When the project is finalised, these files belong in the repository's `src/`
directory. 

## Recommended execution order

| Order | File | Purpose |
|---:|---|---|
| 1 | `01_eda_huggingface.ipynb` | Load the private Hugging Face dataset and perform exploratory data analysis. |
| 2 | `02_boundary_analysis.ipynb` | Investigate the out-of-room ground-truth observations identified during EDA. |
| 3a | `03a_preprocessing_magnitude.ipynb` | Define the log-magnitude preprocessing used by V1-V4. |
| 3b | `03b_preprocessing_signed_iq.ipynb` | Define the signed I/Q representation used by V5. |
| 3c | `03c_preprocessing_fused_magnitude_iq.ipynb` | Define the magnitude and signed-I/Q representation used by V6. |
| 3d | `03d_v7_alternative_split_and_preprocessing.ipynb` | Define the alternative acquisition-level split used by V7-V11. |
| 3e | `03e_preprocessing_phase_ema_temporal.ipynb` | Define the phase-aware, EMA and previous-frame representation used by V10-V11. |
| 4 | `04_baseline_model.ipynb` | Train and compare model versions V1-V12, then export the selected INT8 model. |
| 5 | `05_v11_stratified_group_cross_validation.ipynb` | Run the grouped two-fold robustness assessment for V11. |
| 6 | `06_edge_optimization_audit.ipynb` | Audit float-versus-INT8 size, activation memory, operations, streaming preprocessing memory, and batch-1 desktop latency. |

The notebooks retain their outputs so that the experimental decisions and
results can be inspected without retraining every model.

## Reusable modules

- `eda.py`: validation and aggregation utilities used by the EDA.
- `preprocessing.py`: preprocessing, target construction, and split utilities.
- `modeling.py`: model builders, custom losses, and evaluation metrics.

## Environment setup

The development environment is intentionally separate from the protected root
`requirements.txt` in the GitHub Classroom repository. That protected file
defines only the libraries allowed during final inference.

From the project root, run:

```powershell
.\src\setup_training_env.ps1
```

Then select the `EEAI Training` Jupyter kernel. The script creates
`.venv_training`, installs `src/requirements-training.txt`, verifies the core
packages, and registers the kernel. EDA-only dependencies are listed in
`requirements-eda.txt`.

Virtual environments are machine-specific and must not be copied into `src/`
or committed to Git.

## Dataset access

The private dataset is loaded from Hugging Face. Set a read-only token before
starting Jupyter:

```powershell
$env:HF_TOKEN = "your_read_only_token"
```

## Preprocessing experiment map

| Representation | Model versions | Main reproducibility artifact |
|---|---|---|
| Log magnitude | V1-V4 | `artifacts/preprocessing_config.json` |
| Signed log-I/Q | V5 | `artifacts/v5_iq_normalization.json` |
| Magnitude and signed I/Q | V6 | `artifacts/v6_fused_normalization.json` |
| Fused input with alternative grouped split | V7-V9 | `artifacts/data_split_v7.json`, `artifacts/v7_fused_normalization.json` |
| Magnitude, causal EMA residual, signed I/Q, phase differences, and previous-frame context | V10-V11 | `artifacts/v10_phase_ema_temporal_normalization.json` |

Training-only normalisation statistics are estimated using the corresponding
training acquisition windows. Ground-truth coordinates are preserved for
analysis and clipped to the official room bounds only when targets or final
predictions are constructed.

## Artifacts and models

- `artifacts/` contains saved splits, normalisation statistics, training
  histories, thresholds, and evaluation summaries.
- `models/` contains development checkpoints for the documented experiments.
- `models/baseline_v11_full_int8.tflite` is the current deployment candidate.
- `outputs/eda/` contains the compact EDA summaries used by the notebooks and
  report.
- `outputs/edge_audit/` contains the reproducible float-versus-INT8 comparison,
  memory summary, latency measurements, and report-ready figure.


## Final repository placement

At finalisation time:

1. Copy the contents of this directory into the repository's `src/` folder.
2. Keep `submission/code.py` and `submission/model.tflite` in `submission/`.
3. Keep the final report at `report/report.pdf`.
4. Do not modify `evaluation/` or the protected root `requirements.txt`.
5. Do not copy `.venv_training`, secrets, caches, or temporary evaluator output.
