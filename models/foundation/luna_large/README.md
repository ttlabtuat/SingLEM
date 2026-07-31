# LUNA-large setup

- Official source: <https://github.com/pulp-bio/BioFoundation>
- Checkpoint: <https://huggingface.co/PulpBio/LUNA>
- Source license: Apache-2.0; the minimal required source and license are included.
- Weight license: follow the terms stated on the upstream model card.

Small setup: keep the included source files, download
`LUNA_large.safetensors`, replace `weights/LUNA_large.safetensors`, and run:

```bash
python models/foundation/setup_models.py --models luna_large --verify
```

Full-repository setup is also supported. The BioFoundation GitHub repository
contains the required LUNA source files, while the LUNA-large checkpoint is
downloaded from HuggingFace:

```bash
python models/foundation/setup_models.py \
  --models luna_large \
  --install \
  --source_root /path/to/BioFoundation \
  --checkpoint_root /path/to/luna-checkpoint-folder \
  --verify
```

If the full BioFoundation repository contents were pasted directly into this
folder, omit `--source_root`. The checkpoint must still be placed at
`weights/LUNA_large.safetensors` or in a folder passed with `--checkpoint_root`.

The expected SHA-256 checksum is recorded in `../manifest.json`.
