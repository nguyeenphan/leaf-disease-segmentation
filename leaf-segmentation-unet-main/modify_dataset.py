import os
import random
import shutil
from pathlib import Path

# --- Config ---
ORI_IMG = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/ori_dataset")
MASKS   = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/masks")
OUT     = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/split_ori")

VAL_RATIO  = 0.1
TEST_RATIO = 0.1
SEED = 42
# --------------

random.seed(SEED)

for split in ["train", "val", "test"]:
    (OUT / split).mkdir(parents=True, exist_ok=True)

# Lấy tất cả id từ ori_dataset
all_imgs = sorted(list(ORI_IMG.glob("*.png")))
ids = [p.stem for p in all_imgs]  # "00000", "00001", ...
random.shuffle(ids)

n_total = len(ids)
n_val = int(n_total * VAL_RATIO)
n_test = int(n_total * TEST_RATIO)
n_train = n_total - n_val - n_test

train_ids = ids[:n_train]
val_ids   = ids[n_train:n_train+n_val]
test_ids  = ids[n_train+n_val:]

def copy_ids(id_list, split):
    for id in id_list:
        img_src  = ORI_IMG / f"{id}.png"
        mask_src = MASKS   / f"{id}.png"
        img_dst  = OUT / split / f"{id}_img.png"
        mask_dst = OUT / split / f"{id}_seg.png"
        if img_src.exists() and mask_src.exists():
            shutil.copy2(img_src, img_dst)
            shutil.copy2(mask_src, mask_dst)

copy_ids(train_ids, "train")
copy_ids(val_ids, "val")
copy_ids(test_ids, "test")

print(f"Done! Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
