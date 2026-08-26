# Results

This directory uses the same layout as the revised public experiment runners:

```text
strict/          strict LOSO SVM, MLP, and neural results
adapted_30/      30% calibration MLP and neural results
ablation/        GPU/cuML SVM SingLEM ablation results
single_channel/  GPU/cuML SVM SingLEM per-electrode results
statistical_significance/ paired subject-level Wilcoxon-Holm analysis
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

Regenerate the manuscript significance analysis and all root README tables with:

```bash
python analysis/statistical_significance.py --verify-manuscript
python analysis/build_result_tables.py
```

Regenerate summaries and portable metadata after copying or rerunning metrics
with:

```bash
python analysis/summarize_results.py --result_root results
```

Accuracy and macro-F1 are displayed as percentages. Cohen's kappa remains on
its original scale. Sample standard deviation is used for generated summaries.

## Statistical Significance

`statistical_significance/` contains the complete inferential analysis behind
the `*` markers in manuscript Tables II--V and the root README. Each observation
is one held-out-subject accuracy, and SingLEM and each applicable baseline are
paired by exact subject ID. Tests are two-sided paired Wilcoxon signed-rank
tests; Holm correction is applied independently within each dataset and
evaluation setting. Accuracy is the inferential metric, while macro-F1 and
Cohen's kappa remain descriptive.

The manuscript tables use only `*` for compact presentation. The result folder
provides raw and Holm-adjusted p-values, effect sizes, signed differences, tie
diagnostics, aligned subject accuracies, source-file hashes, and validation
metadata. Regenerate the statistics before rebuilding README tables:

```bash
python analysis/statistical_significance.py --verify-manuscript
python analysis/build_result_tables.py
```

For results from an independent rerun, omit `--verify-manuscript`. The analysis
will use the same statistical procedure without requiring exact equality to the
committed manuscript means.
