# CSBrain setup

- Official source: <https://github.com/yuchen2199/CSBrain>
- Checkpoint: <https://drive.google.com/drive/folders/1-GsVVewRM0B93H08yts5m53yU2whxYvj>
- License: no explicit repository license was identified; source is not redistributed here.

Small setup: replace the placeholder model source files under `models/` and
`CSBrain.pth` with the official files. `models/__init__.py` may remain empty;
the required source filenames are recorded in `../manifest.json`. Verify with:

```bash
python models/foundation/setup_models.py --models csbrain --verify
```

Full-repository setup is also supported. The upstream GitHub repository contains
the required source files under `models/`; the checkpoint is downloaded
separately from the Google Drive folder:

```bash
python models/foundation/setup_models.py \
  --models csbrain \
  --install \
  --source_root /path/to/CSBrain \
  --checkpoint_root /path/to/csbrain-checkpoint-folder \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root`. The checkpoint must still be placed at `CSBrain.pth` or in
a folder passed with `--checkpoint_root`.

The checkpoint checksum is recorded in `../manifest.json`.
