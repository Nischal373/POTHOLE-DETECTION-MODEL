"""
Inference script to predict if an image contains potholes.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys
from pathlib import Path
from functools import lru_cache

class PotholeDetector(nn.Module):
    """Pothole detection model using transfer learning."""
    
    def __init__(self, model_name='resnet18', num_classes=2):
        super(PotholeDetector, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=False)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path="best_pothole_model.pth", device=None):
    """Load the trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = str(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'resnet18')
    classes = checkpoint.get('classes', ['no_pothole', 'pothole'])
    
    model = PotholeDetector(model_name=model_name, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, classes, device


@lru_cache(maxsize=16)
def _get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image_with_model(
    image_path,
    model,
    classes,
    device,
    img_size=224,
    threshold=0.5,
):
    """Predict using an already-loaded model (recommended for APIs)."""
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {
            'success': False,
            'error': f"Error loading image: {str(e)}"
        }

    transform = _get_transform(int(img_size))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_idx = predicted.item()

    predicted_class = classes[predicted_idx]
    has_pothole = predicted_class == 'pothole' and confidence >= threshold

    # Get probability for pothole class
    pothole_prob = probabilities[0][classes.index('pothole')].item() if 'pothole' in classes else 0.0

    return {
        'success': True,
        'has_pothole': has_pothole,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'pothole_probability': pothole_prob,
        'no_pothole_probability': 1.0 - pothole_prob
    }

def predict_image(image_path, model_path="best_pothole_model.pth", 
                 img_size=224, threshold=0.5):
    """
    Predict if an image contains a pothole.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        img_size: Size to resize image to
        threshold: Confidence threshold for pothole detection
    
    Returns:
        dict with prediction results
    """
    # Load model (for CLI use). For APIs, prefer predict_image_with_model().
    model, classes, device = load_model(model_path)
    return predict_image_with_model(
        image_path=image_path,
        model=model,
        classes=classes,
        device=device,
        img_size=img_size,
        threshold=threshold,
    )

def predict_batch(image_dir, model_path="best_pothole_model.pth", 
                 img_size=224, threshold=0.5):
    """Predict on a directory of images."""
    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    images = []
    for ext in image_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    results = []
    for img_path in images:
        result = predict_image(str(img_path), model_path, img_size, threshold)
        if result['success']:
            result['image_path'] = str(img_path)
        results.append(result)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path_or_dir> [model_path]")
        print("Examples:")
        print("  python predict.py image.jpg best_pothole_model.pth")
        print("  python predict.py test_img best_pothole_model.pth")
        sys.exit(1)
    
    input_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "best_pothole_model.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    p = Path(input_path)
    if not p.exists():
        print(f"Error: Path '{input_path}' not found!")
        sys.exit(1)

    # If a directory is provided, run batch prediction on all images inside it.
    if p.is_dir():
        results = predict_batch(str(p), model_path)
        successes = [r for r in results if r.get("success")]
        failures = [r for r in results if not r.get("success")]
        pothole_yes = [r for r in successes if r.get("has_pothole")]
        pothole_no = [r for r in successes if not r.get("has_pothole")]

        print(f"\nDirectory: {str(p)}")
        print(f"Model: {model_path}")
        print(f"Images processed: {len(results)} (ok: {len(successes)}, failed: {len(failures)})")
        print(f"Pothole YES: {len(pothole_yes)} | Pothole NO: {len(pothole_no)}\n")

        for r in successes:
            verdict = "YES" if r["has_pothole"] else "NO"
            print(f"{r['image_path']} -> {verdict} (conf: {r['confidence']:.2%}, pothole_prob: {r['pothole_probability']:.2%})")

        if failures:
            print("\nFailed images:")
            for r in failures:
                print(f"- {r.get('image_path', '(unknown)')}: {r.get('error', 'Unknown error')}")
        sys.exit(0)

    # Otherwise treat it as a single image file.
    result = predict_image(str(p), model_path)

    if not result['success']:
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nImage: {str(p)}")
    print(f"Has Pothole: {'YES' if result['has_pothole'] else 'NO'}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Pothole Probability: {result['pothole_probability']:.2%}")
    print(f"No Pothole Probability: {result['no_pothole_probability']:.2%}")
    