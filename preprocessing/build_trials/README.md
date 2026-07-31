# Build Trials

Build model-specific trial files from raw datasets.

The builders implement the preprocessing used in the revised manuscript:
notch/band-pass filtering, resampling, unit scaling, model-specific channel
handling, and artifact rejection. Samples exceeding the configured amplitude
threshold are removed and treated as boundaries between continuous clean EEG
segments, so extracted trials do not cross artifact boundaries.

```bash
python preprocessing/build_trials/build_all_trials.py \
  --raw_root raw_datasets \
  --output_root datasets/trials \
  --datasets dreyer,wbcic_2c,wbcic_3c,atten_nback,atten_dsr,atten_word \
  --models singlem,bendr,biot,labram,cbramod,csbrain,codebrain,luna_large,mirepnet \
  --n_jobs 32 \
  --overwrite
```
