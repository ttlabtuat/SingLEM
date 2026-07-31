# LaBraM setup

- Official source: <https://github.com/935963004/LaBraM>
- License: MIT; the minimal required source and upstream license are included.

Small setup: keep the included source files, obtain the pretrained
`labram-base.pth` following the official repository, replace
`checkpoints/labram-base.pth`, and verify with:

```bash
python models/foundation/setup_models.py --models labram --verify
```

Full-repository setup is also supported:

```bash
python models/foundation/setup_models.py \
  --models labram \
  --install \
  --source_root /path/to/LaBraM \
  --checkpoint_root /path/to/LaBraM \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root` and `--checkpoint_root`. The setup script uses only
`modeling_finetune.py`, `utils.py`, and `checkpoints/labram-base.pth`.

The expected SHA-256 checksum is recorded in `../manifest.json`.
