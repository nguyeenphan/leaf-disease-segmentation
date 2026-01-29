from PIL import Image
import os
from pathlib import Path

ORI_IMG = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/ori_dataset")

for img_path in ORI_IMG.glob("*.jpg"):
    with Image.open(img_path) as img:
        png_path = img_path.with_suffix('.png')
        img.save(png_path, "PNG")
        os.remove(img_path)

print("Done!")