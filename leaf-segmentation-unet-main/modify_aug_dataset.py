import os
import random
import shutil
import re
from pathlib import Path

# --- Config ---
AUG_IMG = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/aug_dataset")
MASKS   = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/masks")
OUT     = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/split_aug")

VAL_RATIO  = 0.1
TEST_RATIO = 0.1
SEED = 42
# --------------

random.seed(SEED)

for split in ["train", "val", "test"]:
    (OUT / split).mkdir(parents=True, exist_ok=True)

# Helper: lấy base_id từ tên file (00000, 00000_style3 → 00000)
def base_id_from_name(name: str) -> str:
    m = re.match(r"^(\d{5})", name)
    return m.group(1) if m else None

# Gom ảnh theo base_id
all_imgs = sorted(list(AUG_IMG.glob("*.png")))
by_id = {}
for p in all_imgs:
    bid = base_id_from_name(p.stem)
    by_id.setdefault(bid, []).append(p)

all_ids = sorted(by_id.keys())
random.shuffle(all_ids)

n_total = len(all_ids)
n_val = int(n_total * VAL_RATIO)
n_test = int(n_total * TEST_RATIO)
n_train = n_total - n_val - n_test

train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train+n_val]
test_ids  = all_ids[n_train+n_val:]

def copy_id_group(id_list, split):
    for bid in id_list:
        mask_src = MASKS / f"{bid}.png"
        assert mask_src.exists(), f"Missing mask for {bid}"
        for img in by_id[bid]:
            # Ảnh
            img_dst = OUT / split / f"{img.stem}_img.png"
            shutil.copy2(img, img_dst)
            # Mask (đổi tên theo ảnh)
            mask_dst = OUT / split / f"{img.stem}_seg.png"
            shutil.copy2(mask_src, mask_dst)

copy_id_group(train_ids, "train")
copy_id_group(val_ids, "val")
copy_id_group(test_ids, "test")

print(f"Done! Train IDs={len(train_ids)}, Val IDs={len(val_ids)}, Test IDs={len(test_ids)}")
print(f"Total images moved: {sum(len(v) for v in by_id.values())}")
