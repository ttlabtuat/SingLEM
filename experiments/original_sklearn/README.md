# Archived CPU sklearn experiments

These portable scripts preserve the original strict LOSO and single-channel
classifier logic. Each script accepts `--data_dir`, `--n_trials`, and
`--output`. The output prefix produces the matching `*_results.txt` and
`*_optuna_results.pkl` files under `results/original_sklearn/`.

These scripts are archival provenance for the earlier CPU sklearn SVM runs. The
main public SVM reproduction path uses RAPIDS/cuML through
`experiments/run_all.py` and writes to `results/strict/svm/` or
`results/ablation/gpu_svm/singlem/`.

Example:

```bash
python experiments/original_sklearn/singlem/downstream_excluded/strict_loso/dreyer/SVM_LOSO_all_DREYER.py \
  --data_dir datasets/features/dreyer/singlem/downstream_excluded \
  --n_trials 40 \
  --output results/original_sklearn/singlem/downstream_excluded/strict_loso/dreyer/DREYER
```
