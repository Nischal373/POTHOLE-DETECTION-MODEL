"""
Training script for pothole detection model using transfer learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import os
from pathlib import Path

class PotholeDetector(nn.Module):
    """Pothole detection model using transfer learning."""
    
    def __init__(self, model_name='resnet18', num_classes=2, pretrained=True):
        super(PotholeDetector, self).__init__()
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model_name = model_name
    
    def forward(self, x):
        return self.backbone(x)

def get_data_loaders(data_dir="dataset", batch_size=32, img_size=224):
    """Create data loaders for training, validation, and testing."""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation/test
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_model(data_dir="dataset", model_name='resnet18', num_epochs=20, 
                batch_size=32, learning_rate=0.001, img_size=224):
    """Main training function."""
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found!")
        print("Please run prepare_data.py first to organize your images.")
        return
    
    # Check for both classes
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(os.path.join(train_dir, 'pothole')):
        print("Error: 'pothole' class not found in training data!")
        return
    
    if not os.path.exists(os.path.join(train_dir, 'no_pothole')):
        print("Warning: 'no_pothole' class not found in training data!")
        print("The model needs both pothole and non-pothole images for binary classification.")
        print("Please add non-pothole road images to the dataset.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_dir, batch_size, img_size
    )
    print(f"Classes: {classes}")
    
    # Create model
    model = PotholeDetector(model_name=model_name, num_classes=2)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = "best_pothole_model.pth"
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes,
                'model_name': model_name
            }, best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    # Test on test set
    print("\n" + "=" * 50)
    print("Evaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print(f"\nTraining complete! Best model saved to: {best_model_path}")

if __name__ == "__main__":
    train_model(
        data_dir="dataset",
        model_name='resnet18',  # Options: 'resnet18', 'resnet50', 'efficientnet'
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001,
        img_size=224
    )
