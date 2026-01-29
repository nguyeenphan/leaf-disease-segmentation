import os
import shutil
import re

base_dir = "dataset"
images_dir = os.path.join(base_dir, "aug_dataset")
masks_dir = os.path.join(base_dir, "masks")
output_dir = os.path.join(base_dir, "masks_aug")

os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r"^(\d{5})")

for filename in os.listdir(images_dir):
    if not filename.endswith(".png"):
        continue

    match = pattern.match(filename)
    if not match:
        continue

    base_name = match.group(1)
    mask_src = os.path.join(masks_dir, f"{base_name}.png")
    mask_dst = os.path.join(output_dir, filename)

    if os.path.exists(mask_src):
        shutil.copy(mask_src, mask_dst)
    else:
        print(f"[WARNING] Missing mask for {filename}")

print("Done!")
