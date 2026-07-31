# Generated trials

This directory contains zero-byte placeholders that mirror the generated trial
layout. `preprocessing/build_trials/build_all_trials.py` overwrites zero-byte
placeholders by default and skips only non-empty files unless `--overwrite` is
passed.

```text
datasets/trials/<dataset>/<model>/<subject>.pkl
```

Each subject file contains trial data, labels, channel names, sampling rate,
trial identifiers, and preprocessing metadata.
