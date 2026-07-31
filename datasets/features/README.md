# Extracted foundation-model features

This directory is populated by
`preprocessing/extract_features/extract_foundation_features.py`. It contains
zero-byte placeholders that mirror the generated feature layout. The extractor
overwrites zero-byte placeholders by default and skips only non-empty files
unless `--overwrite` is passed.

```text
datasets/features/<dataset>/<model>/<subject>.pkl
```

SingLEM keeps the benchmark variants separate:

```text
datasets/features/<dataset>/singlem/downstream_excluded/<subject>.pkl
datasets/features/<dataset>/singlem/downstream_included/<subject>.pkl
datasets/features/<dataset>/singlem/no_feature_embedding/<subject>.pkl
```

Each file contains frozen features, labels, and extraction metadata. Feature
trial identifiers are resolved against the corresponding generated trial file.
