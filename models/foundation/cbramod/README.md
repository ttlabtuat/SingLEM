# CBraMod setup

- Official source: <https://github.com/wjq-learning/CBraMod>
- Checkpoint: <https://huggingface.co/weighting666/CBraMod>
- License: MIT; the minimal required source and upstream license are included.

Small setup: keep the included source files, replace
`pretrained_weights/pretrained_weights.pth` with the original checkpoint, then
run:

```bash
python models/foundation/setup_models.py --models cbramod --verify
```

Full-repository setup is also supported. The upstream GitHub repository contains
the required source files but not the pretrained checkpoint. Download the
checkpoint from HuggingFace and run:

```bash
python models/foundation/setup_models.py \
  --models cbramod \
  --install \
  --source_root /path/to/cbramod \
  --checkpoint_root /path/to/cbramod-checkpoint-folder \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root`. The HuggingFace checkpoint must still be placed at
`pretrained_weights/pretrained_weights.pth` or in a folder passed with
`--checkpoint_root`.

The expected SHA-256 checksum is recorded in `../manifest.json`.
