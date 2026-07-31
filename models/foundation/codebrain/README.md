# CodeBrain setup

- Official source: <https://github.com/jingyingma01/CodeBrain>
- Paper: <https://arxiv.org/abs/2506.09110>
- License: no explicit source-code license was identified; source is not redistributed here.

Small setup: obtain the benchmark source and checkpoint from the official source
or model authors and replace:

```text
Models/SGConv.py
Models/SSSM.py
Checkpoints/CodeBrain.pth
```

Then run `python models/foundation/setup_models.py --models codebrain --verify`.

Full-repository setup is also supported. The upstream GitHub repository contains
the required source files under `Models/`; the checkpoint must be downloaded or
obtained separately if it is not included in the source tree:

```bash
python models/foundation/setup_models.py \
  --models codebrain \
  --install \
  --source_root /path/to/CodeBrain \
  --checkpoint_root /path/to/codebrain-checkpoint-folder \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root`. The checkpoint must still be placed at
`Checkpoints/CodeBrain.pth` or in a folder passed with `--checkpoint_root`.

The expected checkpoint checksum is recorded in `../manifest.json`.
