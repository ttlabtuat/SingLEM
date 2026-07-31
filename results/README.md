# Results

This directory uses the same layout as the revised public experiment runners:

```text
strict/          strict LOSO SVM, MLP, and neural results
adapted_30/      30% calibration MLP and neural results
ablation/        GPU/cuML SVM SingLEM ablation results
single_channel/  GPU/cuML SVM SingLEM per-electrode results
manifests/       published target-calibration trial IDs
```

Each completed model/dataset directory contains `per_subject_metrics.csv`,
`summary.csv`, and `run_metadata.json`. Single-channel result directories
contain `per_channel_subject_metrics.csv`, `channel_summary.csv`, and
`run_metadata.json`.

The main SVM result source is `results/strict/svm/`, which contains strict LOSO
GPU/cuML SVM outputs for foundation models and classical feature baselines. The
SingLEM GPU-SVM ablation results are stored under
`results/ablation/gpu_svm/singlem/`: the main `downstream_excluded` checkpoint
is represented by the strict SingLEM result, while `downstream_included` and
`no_feature_embedding` have separate ablation directories.

The single-channel hierarchy under
`results/single_channel/gpu_svm/singlem/downstream_excluded/` stores the
per-electrode values underlying the topographical maps. Old submitted CPU
sklearn archives are intentionally not stored in this revised `main` result
tree.

Generate all root README tables with:

```bash
python analysis/build_result_tables.py
```

Regenerate summaries and portable metadata after copying or rerunning metrics
with:

```bash
python analysis/summarize_results.py --result_root results
```

Accuracy and macro-F1 are displayed as percentages. Cohen's kappa remains on
its original scale. Sample standard deviation is used for generated summaries.
