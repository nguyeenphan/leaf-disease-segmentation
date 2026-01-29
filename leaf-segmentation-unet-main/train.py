import glob
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from model.resnet50_unet import ResNet50UNet, focal_tversky_loss, laplacian_loss
from preprocess import check_dir

# Configuration
CONFIG: Dict[str, Any] = {
    "TRAIN_PATH": "./dataset/augmented/train",
    "VAL_PATH": "./dataset/augmented/val",
    "WEIGHT_PATH": "./model/resnet50",
    "BATCH_SIZE": 16,
    "NUM_WORKERS": 4,
    "LR": 1e-4,
    "EPOCHS": 100,
    "IMG_SIZE": (224, 224),
}


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        delta: float = 0,
        path: str = "checkpoint.pt",
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.mode = mode
        self.val_score_best = -np.inf if mode == "max" else np.inf

    def __call__(self, score: float, model: torch.nn.Module) -> None:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        else:
            improved = (
                score > self.best_score + self.delta
                if self.mode == "max"
                else score < self.best_score - self.delta
            )
            
            if improved:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score: float, model: torch.nn.Module) -> None:
        torch.save(model.state_dict(), self.path)
        self.val_score_best = score


def read_imgs_and_masks(folder_path: str) -> Tuple[List[str], List[str]]:
    mask_paths = glob.glob(os.path.join(folder_path, "*seg.png"))
    img_paths = [path.replace("seg", "img") for path in mask_paths]
    return img_paths, mask_paths


class ParseDataset(Dataset):
    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        augment: Optional[A.Compose] = None,
    ) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.normalize = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        if self.augment:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        normalized = self.normalize(image=image, mask=mask)
        image, mask = normalized["image"], normalized["mask"]
        mask = mask.float().unsqueeze(0) / 255.0

        return image, mask


train_aug = A.Compose(
    [
        A.RandomResizedCrop(
            height=CONFIG["IMG_SIZE"][0],
            width=CONFIG["IMG_SIZE"][1],
            scale=(160 / 224, 1.0),
            p=0.5,
        ),
        A.PadIfNeeded(min_height=CONFIG["IMG_SIZE"][0], min_width=CONFIG["IMG_SIZE"][1], p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
    ]
)

val_aug = A.Compose([A.Resize(height=CONFIG["IMG_SIZE"][0], width=CONFIG["IMG_SIZE"][1])])


def calc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    metrics: Dict[str, float],
    bce_weight: float = 0.3,
    focal_tv_weight: float = 0.4,
    lap_weight: float = 0.3,
) -> torch.Tensor:
    pred_sigmoid = torch.sigmoid(pred)
    batch_size = target.size(0)
    
    losses = {
        "bce": F.binary_cross_entropy_with_logits(pred, target),
        "focal_tversky": focal_tversky_loss(pred_sigmoid, target, alpha=0.4, beta=0.6, gamma=1.5),
        "laplacian": laplacian_loss(pred_sigmoid, target),
    }
    
    loss = losses["bce"] * bce_weight + losses["focal_tversky"] * focal_tv_weight + losses["laplacian"] * lap_weight
    
    for key, val in losses.items():
        metrics[key] += val.item() * batch_size
    metrics["loss"] += loss.item() * batch_size

    return loss


def compute_dice_iou(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
    iou = (intersection + 1e-5) / (union - intersection + 1e-5)

    return dice.mean().item(), iou.mean().item()


def print_metrics(metrics: Dict[str, float], epoch_samples: int, phase: str) -> None:
    outputs = [f"{key}: {metrics[key] / epoch_samples:.4f}" for key in metrics.keys()]
    print(f"{phase}: {', '.join(outputs)}")


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int = 100,
    patience: int = 7,
) -> torch.nn.Module:
    best_dice = 0.0
    scaler = amp.GradScaler()
    save_path = os.path.join(CONFIG["WEIGHT_PATH"], "resnet50_best_dice.pth")
    early_stopping = EarlyStopping(patience=patience, path=save_path, mode="max")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        since = time.time()

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            metrics: Dict[str, float] = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    with amp.autocast():
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)
                        dice, iou = compute_dice_iou(outputs, labels)

                    metrics["dice"] += dice * inputs.size(0)
                    metrics["iou"] += iou * inputs.size(0)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_dice = metrics["dice"] / epoch_samples

            if phase == "train":
                scheduler.step()

            if phase == "val":
                early_stopping(epoch_dice, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    model.load_state_dict(torch.load(early_stopping.path))
                    return model
                if epoch_dice > best_dice:
                    best_dice = epoch_dice

        time_elapsed = time.time() - since
        print(f"Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        torch.save(
            model.state_dict(),
            os.path.join(CONFIG["WEIGHT_PATH"], "resnet50_latest_weights.pth"),
        )

    print(f"\nBest val Dice: {best_dice:.4f}")
    model.load_state_dict(torch.load(early_stopping.path))
    return model


def main() -> None:
    check_dir(CONFIG["WEIGHT_PATH"])

    train_img_paths, train_img_masks = read_imgs_and_masks(CONFIG["TRAIN_PATH"])
    val_img_paths, val_img_masks = read_imgs_and_masks(CONFIG["VAL_PATH"])

    print(f"Train size: {len(train_img_paths)}, Val size: {len(val_img_paths)}")

    train_set = ParseDataset(train_img_paths, train_img_masks, augment=train_aug)
    val_set = ParseDataset(val_img_paths, val_img_masks, augment=val_aug)

    global dataloaders
    dataloaders = {
        "train": DataLoader(
            train_set,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=True,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=True,
            persistent_workers=True,
        ),
        "val": DataLoader(
            val_set,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False,
            num_workers=CONFIG["NUM_WORKERS"],
            pin_memory=True,
            persistent_workers=True,
        ),
    }

    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Device: GPU - {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    model = ResNet50UNet(n_class=1).to(device)
    optimizer_ft = optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

    model = train_model(
        model, optimizer_ft, exp_lr_scheduler, num_epochs=CONFIG["EPOCHS"], patience=10
    )


if __name__ == "__main__":
    main()
