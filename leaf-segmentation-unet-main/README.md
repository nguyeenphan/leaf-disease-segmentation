# Leaf Disease Segmentation U-Net

Semantic segmentation of leaf disease images using **ResNet50-UNet with Attention Gates**. Trained with combined BCE, Focal Tversky, and Laplacian loss for binary leaf/background masks.

---

## Features

- **Architecture:** ResNet50 encoder + U-Net decoder with attention gates
- **Loss:** BCE + Focal Tversky + Laplacian (configurable weights)
- **Training:** Early stopping, Dice/IoU validation, optional mixed precision
- **Inference:** Batch prediction with GPU/CUDA, MPS (Apple Silicon), or CPU

---

## Requirements

- Python 3.8+
- PyTorch, torchvision
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── dataset/
│   ├── augmented/          # train/val (after preprocessing)
│   │   ├── train/          # *_img.png, *_seg.png
│   │   └── val/
│   └── original/
│       └── test/           # *_img.png for prediction
├── model/
│   ├── resnet18_unet.py
│   ├── resnet50_unet.py    # main model + loss functions
│   └── resnet50-unet-aug/  # trained weights (e.g. loss1_bce_dice/)
├── preprocess/
│   ├── generate_dataset.py
│   └── README.md
├── utilities/
├── evaluation/
├── modify/                  # dataset & evaluation scripts
├── train.py
├── predict.py
└── requirements.txt
```

**Dataset naming:** For each sample use `*_img.png` (image) and `*_seg.png` (binary mask). Example: `00001_img.png`, `00001_seg.png`.

---

## Dataset Preparation

### Option A: From DenseLeaves (original pipeline)

1. Go to `preprocess/` and download [DenseLeaves.zip](https://www.egr.msu.edu/denseleaves/Data/DenseLeaves.zip).
2. Unzip into `preprocess/DenseLeaves/`.
3. Run: `python generate_dataset.py` (from inside `preprocess/`).
4. Dataset is written to `./dataset` (adjust paths in `train.py` if needed).

### Option B: Custom dataset

- **Train/val:** Put images and masks in `dataset/augmented/train/` and `dataset/augmented/val/` with filenames `*_img.png` and `*_seg.png`.
- **Test:** Put test images in `dataset/original/test/` as `*_img.png`.

Update `CONFIG` in `train.py` and `TEST_DIR` / `SAVE_PATH` in `predict.py` if your paths differ.

---

## Training

Main settings are in the `CONFIG` dict at the top of `train.py`:

| Key           | Default                     | Description               |
| ------------- | --------------------------- | ------------------------- |
| `TRAIN_PATH`  | `./dataset/augmented/train` | Training images & masks   |
| `VAL_PATH`    | `./dataset/augmented/val`   | Validation set            |
| `WEIGHT_PATH` | `./model/resnet50`          | Where to save checkpoints |
| `BATCH_SIZE`  | 16                          | Batch size                |
| `EPOCHS`      | 100                         | Max epochs                |
| `IMG_SIZE`    | (224, 224)                  | Input size                |

Run training:

```bash
python train.py
```

Saved files under `WEIGHT_PATH`:

- `resnet50_best_dice.pth` — best validation Dice
- `resnet50_latest_weights.pth` — last epoch

Training uses early stopping (patience 7) on validation Dice.

---

## Prediction

Edit the following at the top of `predict.py`:

| Variable       | Default                                    | Description                   |
| -------------- | ------------------------------------------ | ----------------------------- |
| `WEIGHT_PATH`  | `./model/resnet50-unet-aug/loss1_bce_dice` | Folder containing `.pth` file |
| `USE_BEST_VAL` | `True`                                     | Use best Dice weights         |
| `TEST_DIR`     | `./dataset/original/test`                  | Directory of `*_img.png`      |
| `SAVE_PATH`    | `./output/res50-aug`                       | Output directory              |

Then run:

```bash
python predict.py
```

Outputs are written to `SAVE_PATH` with optional `PREFIX` (e.g. `seg_00001_img.png`). Input images are resized to 224×224 and normalized with ImageNet stats.

**Tip:** If segmentation is too strict or loose, adjust the binary threshold in `predict.py` (e.g. the `np.where(seg_np > 220, 1, 0)` logic or the sigmoid threshold).

---

## Pretrained Weights

- **From this repo:** Use weights under `model/resnet50-unet-aug/` (e.g. `loss1_bce_dice/resnet50_best_dice.pth`). Set `WEIGHT_PATH` in `predict.py` to the folder that contains the `.pth` file.
- **Legacy / external:** If you have weights from the old pipeline (e.g. `get_pretrained.sh` or `model/pretrained/download.md`), place them in a directory and set `WEIGHT_PATH` to that directory; ensure the filename in `predict.py` matches (`resnet50_best_dice.pth` or `resnet50_latest_weights.pth`).

---

## Evaluation & Scripts

- **`evaluation/`** — Evaluation results and scripts (e.g. Dice/IoU on test sets).
- **`modify/`** — Utilities for dataset modification, mask processing, and visualization.

See scripts inside those folders for usage.

---

## Device

- **CUDA:** Used automatically if available.
- **MPS (Apple Silicon):** Used if CUDA is not available.
- **CPU:** Fallback if neither is available.

---

## Acknowledgments

- [DenseLeaves](https://www.egr.msu.edu/denseleaves/) — Michigan State University
- [Plant Pathology 2021 FGVC8](https://www.kaggle.com/c/plant-pathology-2021-fgvc8) — Kaggle (sample inputs)

---

## License

See `LICENSE` in the repository.
