"""
Simplified Temporal Median CNN - Quick Start
============================================

A streamlined version for quick experimentation.
Use this for rapid prototyping and testing.

Usage:
    python temporal_median_simple.py
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - MODIFY THESE FOR YOUR SETUP
REAL_DIR = "/Users/mac/Documents/FMT/original-faces"
FAKE_DIRS = [
    "/Users/mac/Documents/FMT/fake-faces/Deepfakes",
    "/Users/mac/Documents/FMT/fake-faces/Face2Face", 
    "/Users/mac/Documents/FMT/fake-faces/FaceShifter",
    "/Users/mac/Documents/FMT/fake-faces/FaceSwap",
    "/Users/mac/Documents/FMT/fake-faces/NeuralTextures"
]

# Parameters
MAX_FRAMES = 128
IMG_SIZE = 256  # Smaller for faster training
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_video(video_dir, max_frames=128, target_size=(128, 128)):
    """Load video frames and compute temporal median residual."""
    
    if not os.path.exists(video_dir):
        return None
    
    # Get image files
    files = sorted([f for f in os.listdir(video_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))])[:max_frames]
    
    if len(files) < 16:
        return None
    
    # Load frames
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(video_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            frames.append(img.astype(np.float32))
    
    if len(frames) < 16:
        return None
    
    frames = np.array(frames)
    
    # Compute temporal median residual
    median = np.median(frames, axis=0)
    residual = frames - median[np.newaxis, :, :]
    
    # Normalize to [-1, 1]
    residual = residual / 127.5
    
    return residual


def collect_dataset(real_dir, fake_dirs, videos_per_method=5):
    """Collect video paths and labels."""
    
    paths = []
    labels = []
    
    # Real videos
    print("Collecting real videos...")
    real_path = os.path.join(real_dir)
    real_subdirs = sorted([d for d in os.listdir(real_path) 
                          if os.path.isdir(os.path.join(real_path, d))])[:25]
    
    for subdir in real_subdirs:
        paths.append(os.path.join(real_path, subdir))
        labels.append(0)
    
    print(f"  Real: {len(real_subdirs)} videos")
    
    # Fake videos
    print("Collecting fake videos...")
    for fake_dir in fake_dirs:
        if not os.path.exists(fake_dir):
            print(f"  ⚠ Not found: {fake_dir}")
            continue
        
        fake_subdirs = sorted([d for d in os.listdir(fake_dir)
                              if os.path.isdir(os.path.join(fake_dir, d))])[:videos_per_method]
        
        for subdir in fake_subdirs:
            paths.append(os.path.join(fake_dir, subdir))
            labels.append(1)
        
        method_name = os.path.basename(fake_dir)
        print(f"  {method_name}: {len(fake_subdirs)} videos")
    
    print(f"\nTotal: {len(paths)} videos (Real: {labels.count(0)}, Fake: {labels.count(1)})")
    
    return paths, labels


class VideoDataset(Dataset):
    """Simple dataset for temporal median residuals."""
    
    def __init__(self, video_paths, labels, max_frames=128, img_size=128):
        self.data = []
        self.labels = []
        
        print("Loading videos...")
        for path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
            residual = load_video(path, max_frames, (img_size, img_size))
            
            if residual is not None:
                self.data.append(residual)
                self.labels.append(label)
        
        print(f"✓ Loaded {len(self.data)} videos successfully")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        residual = torch.from_numpy(self.data[idx]).float().unsqueeze(0)  # Add channel
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return residual, label


# ============================================================================
# LIGHTWEIGHT 3D CNN MODEL
# ============================================================================

class Lightweight3DCNN(nn.Module):
    """
    Lightweight 3D CNN optimized for speed and efficiency.
    Good balance between accuracy and computational cost.
    """
    
    def __init__(self):
        super(Lightweight3DCNN, self).__init__()
        
        # Input: (1, 128, 128, 128)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (16, 64, 64, 64)
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (32, 32, 32, 32)
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # -> (64, 16, 16, 16)
            
            # Block 4
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)  # -> (128, 1, 1, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-4):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        for data, target in tqdm(train_loader, desc="Training"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(DEVICE), target.to(DEVICE)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model! Accuracy: {accuracy:.4f}")
    
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.4f}")
    
    return all_preds, all_targets


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Temporal Median CNN - Simplified Pipeline")
    print("="*70)
    
    # Collect data
    video_paths, labels = collect_dataset(REAL_DIR, FAKE_DIRS)
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create datasets
    train_dataset = VideoDataset(train_paths, train_labels, MAX_FRAMES, IMG_SIZE)
    val_dataset = VideoDataset(val_paths, val_labels, MAX_FRAMES, IMG_SIZE)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    model = Lightweight3DCNN().to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    preds, targets = train_model(model, train_loader, val_loader, EPOCHS, LR)
    
    # Final report
    print("\n" + "="*70)
    print("Final Classification Report")
    print("="*70)
    print(classification_report(targets, preds, target_names=['Real', 'Fake']))


if __name__ == "__main__":
    main()