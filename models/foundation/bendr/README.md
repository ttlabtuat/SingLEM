# BENDR setup

- Official source: <https://github.com/SPOClab-ca/BENDR>
- Pretrained release: <https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha>
- License: no explicit repository license was identified; source is not redistributed here.

Replace these placeholders with the original files:

```text
dn3_ext.py
encoder.pt
contextualizer.pt
```

Alternatively, use the full upstream repository as the source input and the
release download folder as the checkpoint input:

```bash
python models/foundation/setup_models.py \
  --models bendr \
  --install \
  --source_root /path/to/BENDR \
  --checkpoint_root /path/to/bendr-release-files \
  --verify
```

If the full upstream repository contents were pasted directly into this folder,
omit `--source_root`. The release checkpoints are separate from the GitHub source
tree, so `encoder.pt` and `contextualizer.pt` must still be downloaded from the
pretrained release or placed in a folder passed with `--checkpoint_root`.

Expected SHA-256 checksums are recorded in `../manifest.json`.
