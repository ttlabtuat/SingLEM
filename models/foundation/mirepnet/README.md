# MIRepNet setup

- Official source: <https://github.com/staraink/MIRepNet>
- Checkpoint: <https://huggingface.co/starself/MIRepNet>
- License: MIT; the minimal required source and upstream license are included.

Small setup: keep the included source files, download the checkpoint, replace
`weight/MIRepNet.pth`, and run:

```bash
python models/foundation/setup_models.py --models mirepnet --verify
```

Full-repository setup is also supported. The upstream GitHub repository contains
the required source files and may contain the expected `weight/MIRepNet.pth`
path; otherwise download the checkpoint from HuggingFace:

```bash
python models/foundation/setup_models.py \
  --models mirepnet \
  --install \
  --source_root /path/to/MIRepNet \
  --checkpoint_root /path/to/mirepnet-checkpoint-folder \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root`. The checkpoint must still be placed at
`weight/MIRepNet.pth` or in a folder passed with `--checkpoint_root`.

The expected SHA-256 checksum is recorded in `../manifest.json`.
