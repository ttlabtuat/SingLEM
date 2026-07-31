# RAPIDS environment

The main public SVM results use RAPIDS/cuML. A RAPIDS environment is required
to reproduce the strict GPU-SVM benchmark tables and the SingLEM GPU-SVM
ablation table. Use a RAPIDS version compatible with the installed CUDA driver.
Follow the official [RAPIDS installation guide](https://docs.rapids.ai/install/)
and verify:

```bash
python -c "import cupy, cuml, optuna, sklearn; print(cuml.__version__)"
```

Point the launcher to that interpreter while retaining the PyTorch interpreter
for MLP and neural experiments:

```bash
export PYTORCH_PYTHON=/path/to/pytorch/python
export RAPIDS_PYTHON=/path/to/rapids/python
```

CPU-only machines can still run package validation, preprocessing, feature
extraction on CPU-capable models, and non-SVM experiment groups. They cannot
fully reproduce the main SVM tables without a CUDA-capable RAPIDS/cuML setup.
