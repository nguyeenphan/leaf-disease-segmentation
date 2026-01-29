import cv2, glob, os
from pathlib import Path
import numpy as np
from preprocess.generate_dataset import resize2square, check_dir

IN_ROOT  = Path("dataset/split_aug")     
OUT_ROOT = Path("dataset/split_aug_sq")
SIZE = 224

splits = ["train", "val", "test"]

for split in splits:
    in_dir  = IN_ROOT / split
    out_dir = OUT_ROOT / split
    check_dir(out_dir)

    mask_paths = sorted(glob.glob(str(in_dir / "*_seg.png")))
    img_paths  = [p.replace("_seg.png", "_img.png") for p in mask_paths]

    print(f"[{split}] {len(img_paths)} pairs")

    for img_path, mask_path in zip(img_paths, mask_paths):
        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img  = resize2square(img, SIZE)
        mask = resize2square(mask, SIZE)

        fname_img  = os.path.basename(img_path)
        fname_mask = os.path.basename(mask_path)

        cv2.imwrite(str(out_dir / fname_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / fname_mask), mask)
