import os
import random
import shutil
import re
from pathlib import Path

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
all_ids = sorted(set(base_id_from_name(p.stem) for p in all_imgs if base_id_from_name(p.stem)))

random.shuffle(all_ids)
n_total = len(all_ids)
n_val = int(n_total * VAL_RATIO)
n_test = int(n_total * TEST_RATIO)
n_train = n_total - n_val - n_test

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train + n_val]
test_ids  = all_ids[n_train + n_val:]

def copy_group_augmented(id_list, split):
    for bid in id_list:
        img_variants = list(IMG_DIR.glob(f"{bid}*.png"))
        if not img_variants:
            print(f"[WARNING] No images found for ID {bid}")
            continue

        mask_src = MASK_DIR / f"{bid}.png"
        if not mask_src.exists():
            print(f"[WARNING] Missing mask for ID {bid}")
            continue

        for img_path in img_variants:
            img_name = img_path.name
            mask_dst_name = img_name
            img_dst = OUT_DIR / split / img_name
            mask_dst = OUT_DIR / f"{split}_mask" / mask_dst_name

            shutil.copy2(img_path, img_dst)
            shutil.copy2(mask_src, mask_dst)

copy_group_augmented(train_ids, "train")
copy_group_augmented(val_ids, "val")
copy_group_augmented(test_ids, "test")

print("Done!")
print(f"Train IDs={len(train_ids)}, Val IDs={len(val_ids)}, Test IDs={len(test_ids)}")
print(f"Total unique base IDs={len(all_ids)}")
