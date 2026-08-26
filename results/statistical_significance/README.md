# Statistical Significance Results

These files compare SingLEM (primary) with every applicable non-SingLEM
baseline in manuscript Tables II--V. Each observation is one held-out-subject
accuracy, paired by exact subject ID. Tests are two-sided paired Wilcoxon
signed-rank tests. Holm correction is applied independently within each
evaluation-setting and dataset family, and significance is defined as
Holm-adjusted `p < 0.05`.

The manuscript and root README use `*` as a compact marker for a baseline whose
accuracy is significantly lower than SingLEM. Full raw and adjusted p-values,
effect sizes, difference summaries, tie diagnostics, and test methods are in
`paired_comparisons.csv`; `p_value_report.md` provides readable p-value tables.
`aligned_subject_accuracies.csv` records every paired value, and
`input_inventory.csv` records source paths, hashes, and selected subject IDs.
`dataset_summary.csv`, `analysis_config.json`, and `validation_report.json`
provide family-level summaries, analysis settings, and integrity checks.

Regenerate and verify the committed manuscript results from the repository root:

```bash
python analysis/statistical_significance.py --verify-manuscript
python analysis/build_result_tables.py
```

For independently reproduced result files, omit `--verify-manuscript`; the same
statistical procedure will run without requiring exact equality to the committed
manuscript means.
