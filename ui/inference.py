import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional
import time

from model_loader import get_model_loader, get_multi_model_loader
from utils import (
    preprocess_image,
    postprocess_mask,
    create_overlay,
    create_masked_image,
    pil_to_numpy,
    numpy_to_pil,
    resize_keeping_aspect_ratio
)


class SegmentationPipeline:
    
    def __init__(self, weight_path: str = None, device: str = None):
        self.model_loader = get_model_loader(weight_path, device)
        self.model = self.model_loader.load_model()
        self.device = self.model_loader.get_device()
        
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.6,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.5,
        return_all: bool = True
    ) -> Dict[str, Image.Image]:

        start_time = time.time()
        
        original_np = pil_to_numpy(image)
        
        display_image = resize_keeping_aspect_ratio(original_np, max_size=800)
        
        print(f" Input image shape: {original_np.shape}")
        
        input_tensor = preprocess_image(image, target_size=(224, 224))
        input_tensor = input_tensor.to(self.device)
        
        print(f"Running inference on {self.device}...")
        
        output = self.model(input_tensor)
        
        mask_binary = postprocess_mask(output, threshold=threshold)
        
        mask_resized = np.array(Image.fromarray(mask_binary).resize(
            (display_image.shape[1], display_image.shape[0]),
            Image.Resampling.BILINEAR
        ))
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f}s")
        
        results = {
            'original': numpy_to_pil(display_image),
            'mask': numpy_to_pil(mask_resized)
        }
        
        if return_all:
            overlay = create_overlay(
                display_image,
                mask_resized,
                color=overlay_color,
                alpha=overlay_alpha
            )
            results['overlay'] = numpy_to_pil(overlay)
            
            masked = create_masked_image(display_image, mask_resized)
            results['masked'] = numpy_to_pil(masked)
        
        mask_percentage = (mask_resized > 127).sum() / mask_resized.size * 100
        results['stats'] = {
            'inference_time': inference_time,
            'mask_percentage': mask_percentage,
            'image_size': display_image.shape[:2]
        }
        
        print(f"Segmentation complete! Mask covers {mask_percentage:.2f}% of image")
        
        return results


_pipeline_instance = None


def get_pipeline(weight_path: str = None, device: str = None) -> SegmentationPipeline:

    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = SegmentationPipeline(weight_path, device)
    
    return _pipeline_instance


def segment_image(
    image: Image.Image,
    threshold: float = 0.5,
    overlay_alpha: float = 0.5,
    mask_color: str = "green"
) -> Tuple[Image.Image, Image.Image, Image.Image, str]:

    if image is None:
        return None, None, None, "Please upload an image"
    
    # Color mapping
    color_map = {
        "green": (0, 255, 0),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255)
    }
    
    overlay_color = color_map.get(mask_color.lower(), (0, 255, 0))
    
    pipeline = get_pipeline()
    
    try:
        # Run segmentation
        results = pipeline.predict(
            image,
            threshold=threshold,
            overlay_color=overlay_color,
            overlay_alpha=overlay_alpha,
            return_all=True
        )
        
        # Format statistics
        stats = results['stats']
        stats_text = f"""
### Segmentation Results

- **Inference Time:** {stats['inference_time']:.3f}s
- **Image Size:** {stats['image_size'][1]} × {stats['image_size'][0]} px
- **Segmented Area:** {stats['mask_percentage']:.2f}%
- **Device:** {pipeline.device}
- **Threshold:** {threshold:.2f}
"""
        
        return (
            results['original'],
            results['mask'],
            results['overlay'],
            stats_text
        )
        
    except Exception as e:
        error_msg = f"Error during segmentation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, None, None, error_msg


def segment_image_multi_model(
    image: Image.Image,
    threshold: float = 0.5,
    overlay_alpha: float = 0.5,
    mask_color: str = "green"
) -> tuple:

    if image is None:
        empty_results = [None] * 9 + ["Please upload an image"] * 4
        return tuple(empty_results)
    
    color_map = {
        "green": (0, 255, 0),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255)
    }
    
    overlay_color = color_map.get(mask_color.lower(), (0, 255, 0))
    
    multi_loader = get_multi_model_loader()
    models = multi_loader.get_all_models()
    device = multi_loader.get_device()
    
    original_np = pil_to_numpy(image)
    display_image = resize_keeping_aspect_ratio(original_np, max_size=800)
    
    all_results = {}
    model_order = ['ResNet18-UNet-Ori', 'ResNet18-UNet-Aug', 'ResNet50-UNet-Aug']
    
    try:
        for model_name in model_order:
            if model_name not in models:
                print(f"Warning: {model_name} not loaded")
                continue
            
            model_info = models[model_name]
            model = model_info['model']
            display_name = model_info['display_name']
            
            print(f"\nRunning inference with {display_name}...")
            start_time = time.time()
            
            input_tensor = preprocess_image(image, target_size=(224, 224))
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            mask_binary = postprocess_mask(output, threshold=threshold)
            
            mask_resized = np.array(Image.fromarray(mask_binary).resize(
                (display_image.shape[1], display_image.shape[0]),
                Image.Resampling.BILINEAR
            ))
            
            inference_time = time.time() - start_time
            
            overlay = create_overlay(
                display_image,
                mask_resized,
                color=overlay_color,
                alpha=overlay_alpha
            )
            
            mask_percentage = (mask_resized > 127).sum() / mask_resized.size * 100
            
            all_results[model_name] = {
                'original': numpy_to_pil(display_image),
                'mask': numpy_to_pil(mask_resized),
                'overlay': numpy_to_pil(overlay),
                'inference_time': inference_time,
                'mask_percentage': mask_percentage,
                'display_name': display_name
            }
            
            print(f"✓ {display_name}: {inference_time:.3f}s, {mask_percentage:.2f}% segmented")
        
        outputs = []
        stats_texts = []
        
        for model_name in model_order:
            if model_name in all_results:
                result = all_results[model_name]
                outputs.extend([
                    result['original'],
                    result['mask'],
                    result['overlay']
                ])
                
                # Individual model stats
                stats_text = f"""
**{result['display_name']}**
- Inference Time: {result['inference_time']:.3f}s
- Segmented Area: {result['mask_percentage']:.2f}%
"""
                stats_texts.append(stats_text)
            else:
                # Model not loaded
                outputs.extend([None, None, None])
                stats_texts.append(f"**{model_name}**: Not loaded")
        
        # Comparison statistics
        if len(all_results) > 0:
            total_time = sum(r['inference_time'] for r in all_results.values())
            avg_time = total_time / len(all_results)
            
            comparison_text = f"""
### Overall Statistics

- **Total Inference Time**: {total_time:.3f}s
- **Average Time per Model**: {avg_time:.3f}s
- **Image Size**: {display_image.shape[1]} × {display_image.shape[0]} px
- **Device**: {device}
- **Threshold**: {threshold:.2f}
- **Models Loaded**: {len(all_results)}/3

---

**Segmentation Area Comparison:**
"""
            for model_name in model_order:
                if model_name in all_results:
                    result = all_results[model_name]
                    comparison_text += f"\n- {result['display_name']}: {result['mask_percentage']:.2f}%"
        else:
            comparison_text = "No models loaded"

        return tuple(outputs + stats_texts + [comparison_text])
        
    except Exception as e:
        error_msg = f"Error during multi-model segmentation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        empty_results = [None] * 9 + [error_msg] * 4
        return tuple(empty_results)


if __name__ == "__main__":
    print("Testing inference pipeline...")
    
    test_image = Image.new('RGB', (512, 512), color='green')
    
    pipeline = get_pipeline()
    results = pipeline.predict(test_image)
    
    print("Results keys:", results.keys())
    print("Stats:", results['stats'])
    print("Test passed!")

