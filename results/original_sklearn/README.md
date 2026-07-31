# Archived CPU sklearn results

This directory preserves the original CPU sklearn SVM outputs produced by the
corresponding scripts in `experiments/original_sklearn/`. These files are kept
as provenance and for historical comparisons only. The main public README
benchmark tables use the GPU/cuML SVM summaries under `results/strict/svm/` and
`results/ablation/gpu_svm/singlem/`.

- strict LOSO results for BENDR, BIOT, CBraMod, and LaBraM;
- strict LOSO results for downstream-excluded and downstream-included SingLEM;
- strict LOSO results for the SingLEM no-feature-embedding ablation;
- individual-channel results for downstream-excluded SingLEM.

TXT files contain the original readable logs. Optuna PKL files retain the
full-precision test metrics, validation macro-F1, and selected SVM parameters.
Each result directory also contains standardized `per_subject_metrics.csv`,
`summary.csv`, and `run_metadata.json` files so the archived outputs can be
inspected with the same schema as the main results.
