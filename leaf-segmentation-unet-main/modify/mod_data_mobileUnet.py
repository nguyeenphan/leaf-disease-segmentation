import os
import random
import shutil
import re
from pathlib import Path

# --- Config ---
IMG_DIR = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/aug_dataset")
MASK_DIR = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/masks_aug")
OUT_DIR = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/split_aug_mobileUnet")

VAL_RATIO  = 0.1
TEST_RATIO = 0.1
SEED = 42
# --------------

random.seed(SEED)

for split in ["train", "val", "test"]:
    (OUT_DIR / split).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{split}_mask").mkdir(parents=True, exist_ok=True)

def base_id_from_name(name: str) -> str:
    m = re.match(r"^(\d{5})", name)
    return m.group(1) if m else None

all_imgs = sorted(list(IMG_DIR.glob("*.png")))
all_ids = [base_id_from_name(p.stem) for p in all_imgs if base_id_from_name(p.stem) is not None]

random.shuffle(all_ids)

n_total = len(all_ids)
n_val = int(n_total * VAL_RATIO)
n_test = int(n_total * TEST_RATIO)
n_train = n_total - n_val - n_test

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train+n_val]
test_ids  = all_ids[n_train+n_val:]

def copy_group(id_list, split):
    for bid in id_list:
        img_src = IMG_DIR / f"{bid}.png"
        mask_src = MASK_DIR / f"{bid}.png"
        assert img_src.exists(), f"Missing image for {bid}"
        assert mask_src.exists(), f"Missing mask for {bid}"

        img_dst = OUT_DIR / split / f"{bid}.png"
        shutil.copy2(img_src, img_dst)

        mask_dst = OUT_DIR / f"{split}_mask" / f"{bid}.png"
        shutil.copy2(mask_src, mask_dst)

copy_group(train_ids, "train")
copy_group(val_ids, "val")
copy_group(test_ids, "test")

print(f"Done!")
print(f"Train IDs={len(train_ids)}, Val IDs={len(val_ids)}, Test IDs={len(test_ids)}")
print(f"Total images processed={len(all_ids)}")
