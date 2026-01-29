import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Import model
sys.path.append(str(Path(__file__).parent / 'model'))
from model.resnet50_unet import ResNet50UNet


def create_heatmap_from_features(model, image_tensor, device):
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())
    
    # Hook into decoder layers
    hooks = []
    hooks.append(model.conv_up0.register_forward_hook(hook_fn))
    hooks.append(model.conv_up1.register_forward_hook(hook_fn))
    hooks.append(model.conv_up2.register_forward_hook(hook_fn))
    hooks.append(model.conv_up3.register_forward_hook(hook_fn))
    
    # Forward
    with torch.no_grad():
        output = model(image_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Create heatmap from features
    heatmaps = []
    for feat in feature_maps:
        # Spatial attention: mean across channels
        heatmap = feat.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        heatmap = F.interpolate(heatmap, size=(256, 256), mode='bilinear', align_corners=True)
        heatmap = heatmap.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmaps.append(heatmap)
    
    return output, heatmaps


def create_visualization(image_path, mask_path=None, model_path=None, 
                        output_path='attention_result.png'):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = ResNet50UNet(n_class=1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Load ground truth
    gt_mask = None
    if mask_path:
        gt = Image.open(mask_path).convert('L')
        gt_mask = transform(gt).squeeze().numpy()
    
    # Get prediction and heatmaps
    print("Generating prediction và heatmap...")
    output, heatmaps = create_heatmap_from_features(model, image_tensor, device)
    pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()
    
    # Average all heatmaps
    avg_heatmap = np.mean(heatmaps, axis=0)
    
    avg_heatmap = np.power(avg_heatmap, 0.5)  # Increase contrast in the middle region
    
    # Create figure
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    img_np = np.array(image.resize((256, 256)))
    
    # 1. Input
    axes[0].imshow(img_np)
    axes[0].set_title('Input Image', fontsize=18, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # 2. Ground Truth
    if gt_mask is not None:
        axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=18, fontweight='bold', pad=15)
    else:
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, color='gray')
        axes[1].set_title('Ground Truth', fontsize=18, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # 3. Heatmap (Feature Importance)
    axes[2].imshow(img_np)
    im = axes[2].imshow(avg_heatmap, cmap='jet', alpha=0.7, vmin=0, vmax=1)  # jet = tương phản cao
    axes[2].set_title('Feature Importance\n(Attention-like Heatmap)', 
                     fontsize=18, fontweight='bold', pad=15)
    axes[2].axis('off')
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Importance', fontsize=14, fontweight='bold')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    
    # 4. Prediction
    axes[3].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Prediction', fontsize=18, fontweight='bold', pad=15)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def create_comparison_grid(images_info, model_path, output_path='comparison_grid.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = ResNet50UNet(n_class=1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    n_images = len(images_info)
    fig, axes = plt.subplots(n_images, 4, figsize=(20, 5*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    for idx, info in enumerate(images_info):
        print(f"Processing image {idx+1}/{n_images}...")
        
        # Load image
        image = Image.open(info['image']).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        img_np = np.array(image.resize((256, 256)))
        
        # Load GT
        gt_mask = None
        if 'mask' in info and info['mask']:
            gt = Image.open(info['mask']).convert('L')
            gt_mask = transform(gt).squeeze().numpy()
        
        # Get prediction and heatmap
        output, heatmaps = create_heatmap_from_features(model, image_tensor, device)
        pred_mask = torch.sigmoid(output).cpu().squeeze().numpy()
        avg_heatmap = np.mean(heatmaps, axis=0)
        avg_heatmap = np.power(avg_heatmap, 0.5)  # Increase contrast in the middle region
        
        # Plot
        axes[idx, 0].imshow(img_np)
        if idx == 0:
            axes[idx, 0].set_title('Input', fontsize=16, fontweight='bold')
        axes[idx, 0].set_ylabel(info.get('title', f'Image {idx+1}'), 
                               fontsize=14, fontweight='bold')
        axes[idx, 0].axis('off')
        
        if gt_mask is not None:
            axes[idx, 1].imshow(gt_mask, cmap='gray')
        else:
            axes[idx, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
        if idx == 0:
            axes[idx, 1].set_title('Ground Truth', fontsize=16, fontweight='bold')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(img_np)
        axes[idx, 2].imshow(avg_heatmap, cmap='jet', alpha=0.7, vmin=0, vmax=1)  # jet = tương phản cao
        if idx == 0:
            axes[idx, 2].set_title('Heatmap', fontsize=16, fontweight='bold')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(pred_mask, cmap='gray')
        if idx == 0:
            axes[idx, 3].set_title('Prediction', fontsize=16, fontweight='bold')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Grid saved: {output_path}")
    plt.close()


# ============== EXAMPLE USAGE ==============

if __name__ == '__main__':
    MODEL_PATH = 'model/resnet50-unet-aug/loss2_bce_dice_ftv_lap/resnet50_best_dice.pth'
    
    print("=" * 60)
    print("SIMPLE ATTENTION-STYLE VISUALIZATION")
    print("=" * 60)
    
    # Example 1: Single image
    print("\nExample 1: Single image")
    create_visualization(
        image_path='dataset/augmented/test/00006_img.png',
        mask_path='dataset/augmented/test/00006_seg.png',
        model_path=MODEL_PATH,
        output_path='demo_heatmap_single.png'
    )
    
    # Example 2: Grid
    print("\nExample 2: Grid comparison")
    images_info = [
        {
            'image': 'dataset/augmented/test/00373_img.png',
            'mask': 'dataset/augmented/test/00373_seg.png',
            'title': 'Case 1'
        },
        {
            'image': 'dataset/augmented/test/00470_img.png',
            'mask': 'dataset/augmented/test/00470_seg.png',
            'title': 'Case 2'
        },
        {
            'image': 'dataset/augmented/test/00027_img.png',
            'mask': 'dataset/augmented/test/00027_seg.png',
            'title': 'Case 3'
        },
    ]
    
    create_comparison_grid(
        images_info=images_info,
        model_path=MODEL_PATH,
        output_path='demo_heatmap_grid.png'
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("   - demo_heatmap_single.png")
    print("   - demo_heatmap_grid.png")
    print("=" * 60)

