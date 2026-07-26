# RB-MOL

This repository contains the core code for **Region-Balanced Mask-Aware
Occlusion Learning (RB-MOL)** for occluded person re-identification. The code is
built on the TransReID training framework and adds edge-to-body occlusion
synthesis, mask-aware reliable token selection, region-balanced token selection,
and prototype-based occluded representation learning.

## Main Files

- `train.py`: training entry.
- `test.py`: evaluation entry.
- `config/defaults.py`: default configuration, including RB-MOL options.
- `processor/processor.py`: RB-MOL training logic.
- `configs/rbmol/occ_duke_rbmol.yml`: final Occluded-Duke RB-MOL configuration.


## Data

Place datasets under `./data`. For Occluded-Duke, the expected structure is:

```text
data/
  Occluded_Duke/
    bounding_box_train/
    query/
    bounding_box_test/
```

## Pretrained Weights

Place the ImageNet-pretrained ViT-B/16 weight at:


You may also change `MODEL.PRETRAIN_PATH` in the YAML config.

## Training

Train the main RB-MOL model on Occluded-Duke:

```bash
python train.py --config_file configs/rbmol/occ_duke_rbmol.yml
```

## Evaluation

Set `TEST.WEIGHT` to a trained checkpoint path:

```bash
python test.py --config_file configs/rbmol/occ_duke_rbmol.yml TEST.WEIGHT ./logs/your_run/transformer_best.pth
```
