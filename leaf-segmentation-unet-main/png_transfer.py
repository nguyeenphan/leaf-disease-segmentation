from PIL import Image
import os
from pathlib import Path

# Đường dẫn đến thư mục ori_dataset
ORI_IMG = Path("/Users/nguyenphan/Developer/Thesis/leaf-segmentation-unet-main/dataset/ori_dataset")

# Lặp qua tất cả các tệp .jpg trong thư mục
for img_path in ORI_IMG.glob("*.jpg"):
    # Mở tệp ảnh
    with Image.open(img_path) as img:
        # Đổi tên tệp với phần mở rộng .png
        png_path = img_path.with_suffix('.png')
        # Lưu tệp ảnh dưới định dạng .png
        img.save(png_path, "PNG")
        # Xóa tệp .jpg gốc nếu cần
        os.remove(img_path)

print("Đã chuyển đổi tất cả các tệp .jpg thành .png")