import os
import sys
import torch
from pathlib import Path

# Add parent directory to path to import model
PARENT_DIR = Path(__file__).parent.parent
sys.path.append(str(PARENT_DIR / "leaf-segmentation-unet-main"))

from model.resnet50_unet import ResNet50UNet
from model.resnet18_unet import ResNetUNet


class ModelLoader:
    def __init__(self, weight_path: str, device: str = None):
        self.weight_path = weight_path
        self.device = self._setup_device(device)
        self.model = None
        
    def _setup_device(self, device: str = None) -> torch.device:
        if device:
            return torch.device(device)
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
        
        print(f"Using device: {device_name}")
        return torch.device(device_name)
    
    def load_model(self) -> ResNet50UNet:
        if self.model is not None:
            print("Model already loaded")
            return self.model
        
        print("Loading model architecture...")
        
        num_classes = 1  # Binary segmentation
        self.model = ResNet50UNet(n_class=num_classes)
        
        # Load weights
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Weight file not found: {self.weight_path}")
        
        print(f"Loading weights from: {self.weight_path}")
        
        state_dict = torch.load(
            self.weight_path,
            map_location=self.device
        )
        
        self.model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        return self.model
    
    def get_device(self) -> torch.device:
        """Get the device model is running on"""
        return self.device

# Singleton instance for easy access
_model_loader_instance = None


def get_model_loader(weight_path: str = None, device: str = None):
    global _model_loader_instance
    
    if _model_loader_instance is None:
        if weight_path is None:
            # Default weight path
            weight_path = str(
                PARENT_DIR / "leaf-segmentation-unet-main" / "model" / 
                "resnet50-unet-aug" / "loss3_bce_ftv_lap" / "resnet50_best_dice.pth"
            )
        
        _model_loader_instance = ModelLoader(weight_path, device)
        _model_loader_instance.load_model()
    
    return _model_loader_instance


class MultiModelLoader:
    """Load and manage multiple models for comparison"""
    
    def __init__(self, device: str = None):
        self.device = self._setup_device(device)
        self.models = {}
        self.model_configs = {
            'ResNet50-UNet-Aug': {
                'architecture': 'resnet50',
                'weight_path': str(
                    PARENT_DIR / "leaf-segmentation-unet-main" / "model" / 
                    "resnet50-unet-aug" / "loss3_bce_ftv_lap" / "resnet50_best_dice.pth"
                ),
                'display_name': 'ResNet50-UNet (Augmented)'
            },
            'ResNet18-UNet-Ori': {
                'architecture': 'resnet18',
                'weight_path': str(
                    PARENT_DIR / "leaf-segmentation-unet-main" / "model" / 
                    "resnet18-unet-ori" / "resnet18_best_dice.pth"
                ),
                'display_name': 'ResNet18-UNet (Original)'
            },
            'ResNet18-UNet-Aug': {
                'architecture': 'resnet18',
                'weight_path': str(
                    PARENT_DIR / "leaf-segmentation-unet-main" / "model" / 
                    "resnet18-unet-aug" / "resnet18_best_dice.pth"
                ),
                'display_name': 'ResNet18-UNet (Augmented)'
            }
        }
    
    def _setup_device(self, device: str = None) -> torch.device:
        if device:
            return torch.device(device)
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
        
        print(f"Using device: {device_name}")
        return torch.device(device_name)
    
    def load_all_models(self):
        """Load all models"""
        for model_name, config in self.model_configs.items():
            print(f"\nLoading {config['display_name']}...")
            
            # Check if weight file exists
            if not os.path.exists(config['weight_path']):
                print(f"Warning: Weight file not found for {model_name}: {config['weight_path']}")
                continue
            
            # Create model based on architecture
            if config['architecture'] == 'resnet50':
                model = ResNet50UNet(n_class=1)
            else:  # resnet18
                model = ResNetUNet(n_class=1)
            
            # Load weights
            state_dict = torch.load(
                config['weight_path'],
                map_location=self.device
            )
            model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            self.models[model_name] = {
                'model': model,
                'display_name': config['display_name']
            }
            
            print(f"✓ {config['display_name']} loaded successfully")
        
        print(f"\nTotal models loaded: {len(self.models)}/{len(self.model_configs)}")
        return self.models
    
    def get_model(self, model_name: str):
        """Get a specific model"""
        return self.models.get(model_name, {}).get('model')
    
    def get_all_models(self):
        """Get all loaded models"""
        return self.models
    
    def get_device(self):
        """Get the device models are running on"""
        return self.device


# Global multi-model instance
_multi_model_instance = None


def get_multi_model_loader(device: str = None) -> MultiModelLoader:
    """Get or create multi-model loader instance"""
    global _multi_model_instance
    
    if _multi_model_instance is None:
        _multi_model_instance = MultiModelLoader(device)
        _multi_model_instance.load_all_models()
    
    return _multi_model_instance


if __name__ == "__main__":
    # Test model loading
    print("Testing multi-model loader...")
    loader = get_multi_model_loader()
    models = loader.get_all_models()
    print(f"\nLoaded models: {list(models.keys())}")
    print(f"Device: {loader.get_device()}")
    print("Test passed!")

