import os
import argparse
import random
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim

from model import ResNetUNet, dice_loss
from preprocess import check_dir


IMAGE_SIZE = 224
WEIGHT_PATH = "./model/pretrained"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mask_index(masks_dir: str, allowed_exts: Tuple[str, ...]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for root, _, files in os.walk(masks_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            ext = ext.lower().lstrip('.')
            if ext not in allowed_exts:
                continue
            index[name] = os.path.join(root, f)
    return index


def collect_pairs(images_dir: str, masks_dir: str, allowed_exts: Tuple[str, ...]) -> Tuple[List[str], List[str]]:
    mask_index = build_mask_index(masks_dir, allowed_exts)
    image_paths: List[str] = []
    mask_paths: List[str] = []

    for root, _, files in os.walk(images_dir):
        for f in files:
            name, ext = os.path.splitext(f)
            ext = ext.lower().lstrip('.')
            if ext not in allowed_exts:
                continue
            img_path = os.path.join(root, f)
            if name not in mask_index:
                # try some common suffix conversions
                # e.g., if image is foo_img -> mask might be foo_seg, and vice versa
                candidates = [
                    name.replace('_img', '_seg'),
                    name.replace('-img', '-seg'),
                    name.replace('image', 'mask'),
                    name.replace('Image', 'Mask'),
                    name.replace('IMG', 'SEG'),
                    name.rstrip('_img').rstrip('-img'),
                ]
                found = None
                for c in candidates:
                    if c in mask_index:
                        found = mask_index[c]
                        break
                if found is None:
                    # skip if no corresponding mask
                    continue
                image_paths.append(img_path)
                mask_paths.append(found)
            else:
                image_paths.append(img_path)
                mask_paths.append(mask_index[name])

    # sort by name for deterministic pairing
    paired = sorted(zip(image_paths, mask_paths), key=lambda x: os.path.basename(x[0]))
    image_paths, mask_paths = [p[0] for p in paired], [p[1] for p in paired]
    return image_paths, mask_paths


class ImageMaskDataset(Dataset):
    def __init__(self, img_paths: List[str], mask_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        mask = cv2.imread(self.mask_paths[idx])
        if mask is None:
            raise FileNotFoundError(f"Mask not found for {self.img_paths[idx]}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

        return [image, mask]


trans = transforms.Compose([
    transforms.ToTensor()
])


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)
    d = dice_loss(pred, target)
    loss = bce * bce_weight + d * (1 - bce_weight)
    metrics['bce'] += bce.detach().cpu().numpy() * target.size(0)
    metrics['dice'] += d.detach().cpu().numpy() * target.size(0)
    metrics['loss'] += loss.detach().cpu().numpy() * target.size(0)
    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append(f"{k}: {metrics[k] / epoch_samples:4f}")
    print(f"{phase}: ", ", ".join(outputs))


def train_model(model, dataloaders, optimizer, scheduler, device, num_epochs=25):
    import copy
    from collections import defaultdict

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            from collections import defaultdict
            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'best_val_weights.pth'))
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                scheduler.step()

        torch.save(model.state_dict(), os.path.join(WEIGHT_PATH, 'latest_weights.pth'))
    print(f"Best val loss: {best_loss:4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train UNet from separate image/mask directories")
    parser.add_argument('--images_dir', type=str, required=True, help='Path to images directory')
    parser.add_argument('--masks_dir', type=str, required=True, help='Path to masks directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weights_out', type=str, default=WEIGHT_PATH)
    args = parser.parse_args()

    check_dir(args.weights_out)

    set_seed(args.seed)
    allowed_exts = ('png', 'jpg', 'jpeg', 'bmp')
    img_paths, msk_paths = collect_pairs(args.images_dir, args.masks_dir, allowed_exts)
    if len(img_paths) == 0:
        raise RuntimeError("Không tìm thấy cặp ảnh–mask hợp lệ. Kiểm tra lại tên file tương ứng giữa images_dir và masks_dir.")

    # deterministic split
    indices = list(range(len(img_paths)))
    random.Random(args.seed).shuffle(indices)
    split_idx = int(len(indices) * (1 - args.val_split))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_imgs = [img_paths[i] for i in train_idx]
    train_msks = [msk_paths[i] for i in train_idx]
    val_imgs = [img_paths[i] for i in val_idx]
    val_msks = [msk_paths[i] for i in val_idx]

    train_set = ImageMaskDataset(train_imgs, train_msks, transform=trans)
    val_set = ImageMaskDataset(val_imgs, val_msks, transform=trans)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on: {device}')

    model = ResNetUNet(n_class=1).to(device)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    global WEIGHT_PATH
    WEIGHT_PATH = args.weights_out

    _ = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, device, num_epochs=args.epochs)


if __name__ == "__main__":
    main()


