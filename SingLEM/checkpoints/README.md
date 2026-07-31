# SingLEM checkpoints

This directory contains the three final retrained SingLEM encoder checkpoints
used in the revised manuscript reproduction package:

| Checkpoint | Source archive | Pretraining corpus | Use |
| --- | --- | --- | --- |
| `singlem_downstream_excluded.pt` | `final_public_encoder/downstream_excluded` | **SingLEM (primary)**. The full architecture pretrained on 68 datasets after excluding `Dreyer_MI_25`, `WBCIC_MI_23`, and `ATTEN_28`. | Main benchmark and single-channel analysis |
| `singlem_downstream_included.pt` | `final_public_encoder/downstream_included` | **SingLEM (all 71 datasets)**. The same architecture and training procedure pretrained on the complete 71-dataset corpus. | General reuse and downstream-included ablation |
| `singlem_no_feature_embedding.pt` | `final_public_encoder/no_feature_embedding` | **SingLEM (w/o feature emb.)**. The 68-dataset leakage-controlled corpus with the feature embedding module removed. | No-feature-embedding ablation |

`singlem_pretrained.pt` is not part of the revised `main` branch and is not
used by the public reproduction code.

This public package reproduces the downstream benchmark and provides the
released pretrained SingLEM checkpoints. It does not package the full
large-scale pretraining pipeline for rebuilding these checkpoints from scratch.

Run `python models/foundation/setup_models.py --verify` to verify external
foundation-model placeholders, and run `python analysis/validate.py
--raw_package --portable` to verify these SingLEM checkpoints.
