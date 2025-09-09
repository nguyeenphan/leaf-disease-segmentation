import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import ResNetUNet


IMAGE_SIZE = 224


def collect_pairs(images_dir: str, masks_dir: str, allowed_exts=("png", "jpg", "jpeg", "bmp")) -> Tuple[List[str], List[str]]:
    mask_index = {}
    for root, _, files in os.walk(masks_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            ext = ext.lower().lstrip('.')
            if ext not in allowed_exts:
                continue
            mask_index[name] = os.path.join(root, f)

    image_paths, mask_paths = [] , []
    for root, _, files in os.walk(images_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            ext = ext.lower().lstrip('.')
            if ext not in allowed_exts:
                continue
            img_path = os.path.join(root, f)
            if name in mask_index:
                image_paths.append(img_path)
                mask_paths.append(mask_index[name])
    paired = sorted(zip(image_paths, mask_paths), key=lambda x: os.path.basename(x[0]))
    return [p[0] for p in paired], [p[1] for p in paired]


class EvalDataset(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str]):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        image = self.to_tensor(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        mask = self.to_tensor(mask)
        return image, mask


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()


def iou_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def evaluate(weights_path: str, images_dir: str, masks_dir: str, batch_size: int = 16, threshold: float = 0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(n_class=1).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    img_paths, msk_paths = collect_pairs(images_dir, masks_dir)
    if len(img_paths) == 0:
        raise RuntimeError("Không tìm thấy cặp ảnh–mask hợp lệ cho evaluate.")
    ds = EvalDataset(img_paths, msk_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dices, ious = [], []
    with torch.no_grad():
        for images, masks in dl:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            dices.append(dice_coeff(preds, masks))
            ious.append(iou_coeff(preds, masks))

    dice_mean = torch.stack(dices).mean().item()
    iou_mean = torch.stack(ious).mean().item()
    print({"dice": dice_mean, "iou": iou_mean})
    return dice_mean, iou_mean


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet on custom image/mask directories")
    parser.add_argument('--weights', type=str, required=True, help='Path to weights .pth')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--masks_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    evaluate(args.weights, args.images_dir, args.masks_dir, args.batch_size, args.threshold)


if __name__ == "__main__":
    main()


