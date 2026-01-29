import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

# Import models
from model.resnet18_unet import ResNetUNet as ResNet18UNet
from model.resnet50_unet import ResNet50UNet

MODEL_TYPE = 'resnet18'
# MODEL_PATH = './model/resnet50-unet-aug/loss3_bce_ftv_lap/resnet50_best_dice.pth'
MODEL_PATH = './model/resnet18-unet-aug/resnet18_best_dice.pth'
TEST_DIR = './dataset/augmented/test'
IMAGE_SIZE = 256
OUTPUT_DIR = './evaluation_results_aug'
USE_GPU = True

def calculate_metrics(pred, target):
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    dice = (2 * intersection + 1e-8) / (pred_flat.sum() + target_flat.sum() + 1e-8)
    
    correct = (pred_flat == target_flat).sum()
    accuracy = correct / pred_flat.numel()
    
    tp = intersection
    fp = pred_flat.sum() - intersection
    fn = target_flat.sum() - intersection
    
    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def load_test_data(test_dir):
    test_path = Path(test_dir)
    img_files = sorted(test_path.glob('*_img.png'))
    
    pairs = []
    for img_path in img_files:
        base_name = img_path.stem.replace('_img', '')
        mask_path = test_path / f"{base_name}_seg.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path, base_name))
    
    print(f"Found {len(pairs)} test samples")
    return pairs


def main():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"\n Loading {MODEL_TYPE} model...")
    if MODEL_TYPE == 'resnet18':
        model = ResNet18UNet(n_class=1)
    elif MODEL_TYPE == 'resnet50':
        model = ResNet50UNet(n_class=1)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Loaded from: {MODEL_PATH}")
    
    img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print(f"\nLoading test data from: {TEST_DIR}")
    test_pairs = load_test_data(TEST_DIR)
    
    print(f"\nEvaluating on {len(test_pairs)} samples...")
    all_metrics = []
    per_image_results = []
    
    with torch.no_grad():
        for img_path, mask_path, name in tqdm(test_pairs, desc="Testing"):
            image = Image.open(img_path).convert('RGB')
            image_tensor = img_transform(image).unsqueeze(0).to(device)
            
            output = model(image_tensor)
            pred = torch.sigmoid(output).squeeze(0)
            
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask)  # Values: 0-38
            
            pred_resized = F.interpolate(
                pred.unsqueeze(0),
                size=mask.size[::-1],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            mask_tensor = torch.from_numpy(mask_np).float()
            
            metrics = calculate_metrics(pred_resized.cpu(), mask_tensor)
            all_metrics.append(metrics)
            per_image_results.append({'name': name, **metrics})
    
    print("RESULTS")
    print("="*70)
    
    for key in ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']:
        values = [m[key] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"\n{key.upper()}:")
        print(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'results.txt'
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Checkpoint: {MODEL_PATH}\n")
        f.write(f"Test set: {TEST_DIR}\n")
        f.write(f"Num samples: {len(test_pairs)}\n")
        f.write(f"Mask encoding: 0=background, >0=disease\n\n")
        
        for key in ['iou', 'dice', 'accuracy', 'precision', 'recall', 'f1']:
            values = [m[key] for m in all_metrics]
            f.write(f"{key.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
        
        f.write("\n\nPER-IMAGE RESULTS:\n")
        f.write(f"{'Name':<15} {'IoU':>8} {'Dice':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}\n")
        f.write("-"*70 + "\n")
        
        for result in per_image_results:
            f.write(f"{result['name']:<15} "
                   f"{result['iou']:>8.4f} "
                   f"{result['dice']:>8.4f} "
                   f"{result['accuracy']:>8.4f} "
                   f"{result['precision']:>8.4f} "
                   f"{result['recall']:>8.4f} "
                   f"{result['f1']:>8.4f}\n")
    
    print(f"\nResults saved: {results_file}")
    
    print("\n" + "="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()

