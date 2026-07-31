# Foundation models

This directory contains the external foundation-model files used by the
frozen-feature adapters. The feature extractors use a small canonical runtime
layout for each model. Source is included only when the upstream license permits
redistribution. A zero-byte file is an intentional placeholder and must be
replaced or normalized before feature extraction.

| Model | Source in this repository | License | Checkpoint |
|---|---|---|---|
| BENDR | Placeholder | Not stated | Placeholder |
| BIOT | Included | MIT | Placeholder |
| CBraMod | Included | MIT | Placeholder |
| LaBraM | Included | MIT | Placeholder |
| CodeBrain | Placeholder | Not stated | Placeholder |
| CSBrain | Placeholder | Not stated | Placeholder |
| LUNA-large | Included | Apache-2.0 | Placeholder; upstream weight terms apply |
| MIRepNet | Included | MIT | Placeholder |

SingLEM is maintained separately under `SingLEM/`, with checkpoints under
`SingLEM/checkpoints/`.

Each external model directory has a README with its official source, required
paths, checkpoint location, and checksum. After replacing placeholders, verify
the installation with:

```bash
python models/foundation/setup_models.py --verify
```

Two setup styles are supported:

1. Replace the exact placeholder files listed in each model README. This keeps
   only the minimal files required by SingLEM.
2. Download or clone the full upstream repository and run `setup_models.py
   --install`. The setup script copies only the files needed by SingLEM into the
   canonical paths and ignores extra upstream files.

Keep the canonical paths unchanged. If your file manager creates folders such as
`BENDR-main copy` or `CSBrain-main copy`, `setup_models.py` may not find them.
Rename or merge downloaded folders so they match the expected names, or pass the
download location explicitly with `--source_root` and `--checkpoint_root`.

Example for a full upstream repository plus a separate checkpoint download
folder:

```bash
python models/foundation/setup_models.py \
  --models biot \
  --install \
  --source_root /path/to/BIOT \
  --checkpoint_root /path/to/BIOT/pretrained-models \
  --verify
```

If the full upstream repository was pasted directly under the corresponding
model folder, the same command can usually be shortened to:

```bash
python models/foundation/setup_models.py --models biot --install --verify
```

The setup script also searches a local staging folder named `foundation_models/`
at the repository root, so `foundation_models/BIOT-main`,
`foundation_models/CSBrain-main`, and similar downloaded repository folders can
be used without passing `--source_root`.

Recommended full-repository placement:

```text
SingLEM/
  SingLEM/                 included SingLEM code and checkpoints
  foundation_models/
    BENDR-main/
    BIOT-main/
    LaBraM-main/
    CBraMod-main/
    CodeBrain-main/
    CSBrain-main/
    BioFoundation-main/
    MIRepNet-main/
```

Do not place SingLEM itself under `foundation_models/`. SingLEM is part of this
repository and its checkpoints are kept under `SingLEM/checkpoints/`.
Do not paste full upstream repositories over `models/foundation/<model>/`
unless you are following that model's README to replace the exact placeholder
files. The recommended full-repository workflow is to keep full repos under
root-level `foundation_models/` and let `setup_models.py --install` copy the
required files into `models/foundation/`.

Then run:

```bash
python models/foundation/setup_models.py --install --verify
```

`--install` overwrites zero-byte placeholders by default. It does not replace
non-empty files unless `--overwrite` is also passed. Use `--dry_run` to inspect
which files would be copied.

To inspect one model only:

```bash
python models/foundation/setup_models.py --models cbramod --verify
```

Downloaded weights modify tracked placeholder files. Keep those replacements
local and do not commit them. `analysis/validate.py --raw_package` enforces this
rule for a portable public package. Some upstream repositories do not include
pretrained weights on GitHub; for those models, download the checkpoint from the
source listed in the model README and pass that folder with `--checkpoint_root`
or place the file at the exact placeholder path.
