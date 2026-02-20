"""
Temporal Median CNN
============================================================
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

REAL_DIR = "/Users/mac/Documents/FMT/original-faces"
FAKE_DIRS = [
    "/Users/mac/Documents/FMT/fake-faces/Deepfakes",
    "/Users/mac/Documents/FMT/fake-faces/Face2Face", 
    "/Users/mac/Documents/FMT/fake-faces/FaceShifter",
    "/Users/mac/Documents/FMT/fake-faces/FaceSwap",
    "/Users/mac/Documents/FMT/fake-faces/NeuralTextures"
]

MAX_FRAMES = 64  # Reduced for faster training
IMG_SIZE = 112   # Smaller but still effective
BATCH_SIZE = 4   # Larger batch
EPOCHS = 50      # More epochs
LR = 5e-4        # Higher learning rate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# IMPROVED DATA PROCESSING
# ============================================================================

def load_video_improved(video_dir, max_frames=64, target_size=(112, 112)):
    """
    Improved video loading with better preprocessing.
    """
    
    if not os.path.exists(video_dir):
        return None
    
    files = sorted([f for f in os.listdir(video_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(files) < 16:
        return None
    
    # Sample frames uniformly if we have too many
    if len(files) > max_frames:
        indices = np.linspace(0, len(files)-1, max_frames).astype(int)
        files = [files[i] for i in indices]
    
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(video_dir, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)
            frames.append(img.astype(np.float32))
    
    if len(frames) < 16:
        return None
    
    frames = np.array(frames)
    
    # IMPROVED NORMALIZATION
    # 1. Compute temporal median
    median = np.median(frames, axis=0)
    
    # 2. Compute residual
    residual = frames - median[np.newaxis, :, :]
    
    # 3. Better normalization: standardize per video
    # This ensures each video has similar scale
    mean = residual.mean()
    std = residual.std()
    
    if std > 1e-5:  # Avoid division by zero
        residual = (residual - mean) / (std + 1e-8)
    else:
        residual = residual / 127.5  # Fallback
    
    # 4. Clip outliers
    residual = np.clip(residual, -3, 3)
    
    return residual


def augment_video(residual):
    """
    Data augmentation for temporal median residuals.
    """
    
    # Random temporal crop (if longer than needed)
    if len(residual) > 32:
        max_start = len(residual) - 32
        start = random.randint(0, max_start)
        residual = residual[start:start+32]
    
    # Random horizontal flip (50%)
    if random.random() > 0.5:
        residual = np.flip(residual, axis=2).copy()
    
    # Random temporal reversal (25%)
    if random.random() > 0.75:
        residual = residual[::-1].copy()
    
    # Small random noise (regularization)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, residual.shape)
        residual = residual + noise
    
    return residual


class ImprovedVideoDataset(Dataset):
    """Dataset with augmentation and better preprocessing."""
    
    def __init__(self, video_paths, labels, max_frames=64, img_size=112, augment=False):
        self.data = []
        self.labels = []
        self.augment = augment
        
        print(f"Loading videos (augment={augment})...")
        for path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
            residual = load_video_improved(path, max_frames, (img_size, img_size))
            
            if residual is not None:
                self.data.append(residual)
                self.labels.append(label)
        
        print(f"✓ Loaded {len(self.data)} videos")
        
        # Print class distribution
        n_real = self.labels.count(0)
        n_fake = self.labels.count(1)
        print(f"  Real: {n_real}, Fake: {n_fake}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        residual = self.data[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augment:
            residual = augment_video(residual)
        
        # Convert to tensor
        residual_tensor = torch.from_numpy(residual).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return residual_tensor, label_tensor


# ============================================================================
# IMPROVED MODEL
# ============================================================================

class ImprovedCNN(nn.Module):
    """
    Improved 3D CNN with:
    - Better initialization
    - Dropout for regularization
    - Batch normalization
    - Skip connections
    """
    
    def __init__(self, dropout=0.5):
        super(ImprovedCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(2)
        
        # Block 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(2)
        
        # Block 3
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Block 4
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Better weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# IMPROVED TRAINING
# ============================================================================

def train_improved(model, train_loader, val_loader, epochs=50, lr=5e-4):
    """
    Improved training with:
    - Learning rate warmup
    - Gradient clipping
    - Better optimization
    - Early stopping
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        for data, target in tqdm(train_loader, desc="Training"):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            pred = torch.argmax(output, dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
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
        val_acc = accuracy_score(all_targets, all_preds)
        
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_improved.pth')
            print(f"✓ New best! Accuracy: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs)")
            break
    
    print(f"\n✅ Training complete! Best accuracy: {best_acc:.4f}")
    
    return all_preds, all_targets


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("IMPROVED Temporal Median CNN")
    print("="*70)
    
    # Collect data
    video_paths = []
    labels = []
    
    # Real videos
    real_subdirs = sorted([d for d in os.listdir(REAL_DIR) 
                          if os.path.isdir(os.path.join(REAL_DIR, d))])[:25]
    for subdir in real_subdirs:
        video_paths.append(os.path.join(REAL_DIR, subdir))
        labels.append(0)
    
    print(f"Real videos: {len(real_subdirs)}")
    
    # Fake videos
    for fake_dir in FAKE_DIRS:
        if not os.path.exists(fake_dir):
            continue
        
        fake_subdirs = sorted([d for d in os.listdir(fake_dir)
                              if os.path.isdir(os.path.join(fake_dir, d))])[:5]
        
        for subdir in fake_subdirs:
            video_paths.append(os.path.join(fake_dir, subdir))
            labels.append(1)
        
        print(f"{os.path.basename(fake_dir)}: {len(fake_subdirs)}")
    
    print(f"\nTotal: {len(video_paths)} videos (Real: {labels.count(0)}, Fake: {labels.count(1)})")
    
    # Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\nTrain: {len(train_paths)}, Val: {len(val_paths)}")
    
    # Create datasets WITH AUGMENTATION
    train_dataset = ImprovedVideoDataset(
        train_paths, train_labels, MAX_FRAMES, IMG_SIZE, augment=True
    )
    val_dataset = ImprovedVideoDataset(
        val_paths, val_labels, MAX_FRAMES, IMG_SIZE, augment=False
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create improved model
    model = ImprovedCNN().to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Train
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    preds, targets = train_improved(model, train_loader, val_loader, EPOCHS, LR)
    
    # Final report
    print("\n" + "="*70)
    print("Final Classification Report")
    print("="*70)
    print(classification_report(targets, preds, target_names=['Real', 'Fake']))


if __name__ == "__main__":
    main()