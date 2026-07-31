# BIOT setup

- Official source: <https://github.com/ycq091044/BIOT>
- Checkpoints: <https://github.com/ycq091044/BIOT/tree/main/pretrained-models>
- License: MIT; the required `biot.py` and upstream license are included.

Small setup: keep the included `biot.py`, replace
`EEG-six-datasets-18-channels.ckpt` with the upstream checkpoint, then run:

```bash
python models/foundation/setup_models.py --models biot --verify
```

Full-repository setup is also supported. The upstream BIOT repository stores the
source as `model/biot.py` and the checkpoint under `pretrained-models/`. To copy
those files into the canonical paths used by this repository, run:

```bash
python models/foundation/setup_models.py \
  --models biot \
  --install \
  --source_root /path/to/BIOT \
  --checkpoint_root /path/to/BIOT/pretrained-models \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root` and `--checkpoint_root`. The feature extractor also accepts
the direct upstream path `model/biot.py`, but `--install` keeps the public layout
consistent by copying it to `biot.py`.

The expected SHA-256 checksum is recorded in `../manifest.json`.
