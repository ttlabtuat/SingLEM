# Extract Features

Extract frozen foundation-model features from model-specific trial files.

Run all benchmark feature extraction jobs with:

```bash
python preprocessing/extract_features/extract_all_features.py \
  --datasets dreyer,wbcic_2c,wbcic_3c,atten_nback,atten_dsr,atten_word \
  --models singlem,bendr,biot,labram,cbramod,csbrain,codebrain,luna_large,mirepnet \
  --singlem_variants all \
  --channel_policy pretrained_matched \
  --input_root datasets/trials \
  --output_root datasets/features \
  --gpu 0 \
  --overwrite
```

`extract_all_features.py` skips MIRepNet on cognitive datasets. The
`--singlem_variants all` option extracts the downstream-excluded SingLEM
features used by the main benchmark and the downstream-included and
no-feature-embedding features used by the GPU-SVM ablation runner. It delegates
each dataset/model job to the single-pair extractor below.

```bash
python preprocessing/extract_features/extract_foundation_features.py \
  --dataset <dataset> \
  --model <model> \
  --channel_policy pretrained_matched \
  --input_root datasets/trials \
  --output_root datasets/features \
  --gpu 0 \
  --overwrite
```

SingLEM supports three pretrained encoders:

```bash
python preprocessing/extract_features/extract_foundation_features.py \
  --dataset <dataset> --model singlem \
  --singlem_variant downstream_excluded \
  --input_root datasets/trials --output_root datasets/features --gpu 0
```

The default is `downstream_excluded`, which is SingLEM (primary) in the revised
manuscript. Use `--singlem_variants all` with `extract_all_features.py` to
extract all SingLEM variants. The alternatives are `downstream_included`
for SingLEM (all 71 datasets) and `no_feature_embedding` for SingLEM
(w/o feature emb.).
Outputs are separated under
`datasets/features/<dataset>/singlem/<variant>/`.
