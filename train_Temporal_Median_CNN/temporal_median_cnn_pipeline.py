"""
Complete Pipeline: Temporal Median Residuals + CNN Classification
==================================================================

This script:
1. Loads 25 real + 25 fake videos (5 from each deepfake method)
2. Computes temporal median residuals
3. Trains a performant CNN (3D ResNet or EfficientNet)
4. Evaluates and visualizes results

Dataset Structure:
    original-faces/000, 001, 002, ...
    fake-faces/Deepfakes/000, 001, ...
    fake-faces/Face2Face/000, 001, ...
    fake-faces/FaceShifter/000, 001, ...
    fake-faces/FaceSwap/000, 001, ...
    fake-faces/NeuralTextures/000, 001, ...
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters."""
    
    # Paths
    REAL_DIR = "original-faces"
    FAKE_DIRS = {
        'Deepfakes': 'fake-faces/Deepfakes',
        'Face2Face': 'fake-faces/Face2Face',
        'FaceShifter': 'fake-faces/FaceShifter',
        'FaceSwap': 'fake-faces/FaceSwap',
        'NeuralTextures': 'fake-faces/NeuralTextures'
    }
    
    # Data parameters
    MAX_FRAMES = 128
    TARGET_SIZE = (224, 224)  # Resize frames for CNN
    VIDEOS_PER_METHOD = 5
    
    # Training parameters
    BATCH_SIZE = 4  # Small batch due to video memory
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    TRAIN_SPLIT = 0.8
    
    # Model parameters
    MODEL_TYPE = '3d_resnet'  # Options: '3d_resnet', '2d_cnn', 'efficient_3d'
    
    # Output
    OUTPUT_DIR = "temporal_median_cnn_results"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class TemporalMedianProcessor:
    """Process videos to extract temporal median residuals."""
    
    def __init__(self, max_frames=128, target_size=(224, 224)):
        self.max_frames = max_frames
        self.target_size = target_size
    
    def load_video_frames(self, video_dir: str) -> np.ndarray:
        """
        Load frames from a video directory.
        
        Args:
            video_dir: Path to directory containing frame images
            
        Returns:
            frames: Array of shape (T, H, W) or None if failed
        """
        if not os.path.exists(video_dir):
            print(f"⚠ Directory not found: {video_dir}")
            return None
        
        files = sorted(os.listdir(video_dir))
        
        # Filter image files
        files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(files) == 0:
            print(f"⚠ No image files in: {video_dir}")
            return None
        
        # Limit to max_frames
        files = files[:self.max_frames]
        
        frames = []
        for f in files:
            img_path = os.path.join(video_dir, f)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Resize to target size
            img = cv2.resize(img, self.target_size)
            frames.append(img.astype(np.float32))
        
        if len(frames) < 16:  # Need minimum frames
            print(f"⚠ Only {len(frames)} frames in: {video_dir}")
            return None
        
        return np.array(frames)
    
    def compute_temporal_median_residual(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute temporal median and subtract from frames.
        
        This isolates motion information by removing static content.
        
        Args:
            frames: Array of shape (T, H, W)
            
        Returns:
            residual: Array of shape (T, H, W)
        """
        # Compute median across time axis
        median_frame = np.median(frames, axis=0)  # Shape: (H, W)
        
        # Subtract median from each frame
        residual = frames - median_frame[np.newaxis, :, :]  # Broadcasting
        
        # Normalize to [-1, 1] range
        residual = residual / 127.5  # Assuming original range is [0, 255]
        
        return residual
    
    def process_video(self, video_dir: str) -> np.ndarray:
        """
        Complete processing pipeline for one video.
        
        Returns:
            residual: Temporal median residual, shape (T, H, W)
        """
        frames = self.load_video_frames(video_dir)
        
        if frames is None:
            return None
        
        residual = self.compute_temporal_median_residual(frames)
        
        return residual


class VideoDataset(Dataset):
    """PyTorch dataset for temporal median residuals."""
    
    def __init__(self, video_paths, labels, processor):
        """
        Args:
            video_paths: List of paths to video directories
            labels: List of labels (0=real, 1=fake)
            processor: TemporalMedianProcessor instance
        """
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        
        # Pre-process all videos
        print("Pre-processing videos...")
        self.data = []
        self.valid_labels = []
        
        for video_path, label in tqdm(zip(video_paths, labels), total=len(video_paths)):
            residual = processor.process_video(video_path)
            
            if residual is not None:
                self.data.append(residual)
                self.valid_labels.append(label)
        
        print(f"✓ Successfully processed {len(self.data)} videos")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        residual = self.data[idx]
        label = self.valid_labels[idx]
        
        # Convert to tensor
        # For 3D CNN: (C, T, H, W) where C=1 (grayscale)
        residual_tensor = torch.from_numpy(residual).float()
        residual_tensor = residual_tensor.unsqueeze(0)  # Add channel dimension
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return residual_tensor, label_tensor


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class Simple3DCNN(nn.Module):
    """
    Simple 3D CNN for video classification.
    Operates on temporal median residuals.
    """
    
    def __init__(self, num_classes=2):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        # Input: (1, T, H, W) where T=128, H=W=224
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        # Output: (32, T/2, H/4, W/4)
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        # Output: (64, T/4, H/16, W/16)
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )
        # Output: (128, T/8, H/64, W/64)
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        # Output: (256, 1, 1, 1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 1, T, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.fc(x)
        
        return x


class ResNet3D_18(nn.Module):
    """
    3D ResNet-18 inspired architecture.
    More powerful than Simple3DCNN, better for complex patterns.
    """
    
    def __init__(self, num_classes=2):
        super(ResNet3D_18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # First block (may downsample)
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock3D(nn.Module):
    """3D Residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class Efficient2Plus1D(nn.Module):
    """
    (2+1)D CNN: Factorized 3D convolutions.
    Decomposes 3D conv into spatial 2D + temporal 1D.
    More efficient than full 3D conv.
    """
    
    def __init__(self, num_classes=2):
        super(Efficient2Plus1D, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            SpatialTemporalConv(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(SpatialTemporalBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(SpatialTemporalBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SpatialTemporalConv(nn.Module):
    """(2+1)D convolution: spatial 2D + temporal 1D."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SpatialTemporalConv, self).__init__()
        
        # Decompose kernel
        t, h, w = kernel_size
        st, sh, sw = stride
        pt, ph, pw = padding
        
        # Spatial convolution
        self.spatial = nn.Conv3d(in_channels, out_channels, 
                                kernel_size=(1, h, w),
                                stride=(1, sh, sw),
                                padding=(0, ph, pw),
                                bias=False)
        self.bn_spatial = nn.BatchNorm3d(out_channels)
        
        # Temporal convolution
        self.temporal = nn.Conv3d(out_channels, out_channels,
                                 kernel_size=(t, 1, 1),
                                 stride=(st, 1, 1),
                                 padding=(pt, 0, 0),
                                 bias=False)
        self.bn_temporal = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.spatial(x)
        x = self.bn_spatial(x)
        x = self.relu(x)
        
        x = self.temporal(x)
        x = self.bn_temporal(x)
        x = self.relu(x)
        
        return x


class SpatialTemporalBlock(nn.Module):
    """Residual block with (2+1)D convolutions."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(SpatialTemporalBlock, self).__init__()
        
        self.conv1 = SpatialTemporalConv(in_channels, out_channels, 
                                         kernel_size=(3, 3, 3),
                                         stride=(stride, stride, stride),
                                         padding=(1, 1, 1))
        
        self.conv2 = SpatialTemporalConv(out_channels, out_channels,
                                         kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1),
                                         padding=(1, 1, 1))
        
        # Shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Train and evaluate the model."""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.config.DEVICE), target.to(self.config.DEVICE)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(output, dim=1)
                preds = torch.argmax(output, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)
        
        return avg_loss, accuracy, auc, all_preds, all_targets, all_probs
    
    def train(self):
        """Complete training loop."""
        print("="*70)
        print("Starting Training")
        print("="*70)
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, accuracy, auc, preds, targets, probs = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(accuracy)
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_checkpoint('best_model.pth')
                print(f"✓ New best model saved! Accuracy: {accuracy:.4f}")
        
        print("\n" + "="*70)
        print(f"Training Complete! Best Accuracy: {self.best_accuracy:.4f}")
        print("="*70)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, filepath)


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_video_paths(config):
    """
    Collect paths to 25 real and 25 fake videos.
    
    Returns:
        video_paths: List of video directory paths
        labels: List of labels (0=real, 1=fake)
    """
    video_paths = []
    labels = []
    
    # Collect real videos
    print("Collecting REAL videos...")
    real_dir = Path(config.REAL_DIR)
    real_subdirs = sorted([d for d in real_dir.iterdir() if d.is_dir()])[:25]
    
    for video_dir in real_subdirs:
        video_paths.append(str(video_dir))
        labels.append(0)  # Real
    
    print(f"✓ Found {len(real_subdirs)} real videos")
    
    # Collect fake videos (5 from each method)
    print("\nCollecting FAKE videos...")
    for method_name, method_dir in config.FAKE_DIRS.items():
        method_path = Path(method_dir)
        
        if not method_path.exists():
            print(f"⚠ Directory not found: {method_dir}")
            continue
        
        method_subdirs = sorted([d for d in method_path.iterdir() if d.is_dir()])[:config.VIDEOS_PER_METHOD]
        
        for video_dir in method_subdirs:
            video_paths.append(str(video_dir))
            labels.append(1)  # Fake
        
        print(f"✓ Found {len(method_subdirs)} videos from {method_name}")
    
    print(f"\n📊 Total dataset: {len(video_paths)} videos")
    print(f"   Real: {labels.count(0)}")
    print(f"   Fake: {labels.count(1)}")
    
    return video_paths, labels


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(trainer, output_dir):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(trainer.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(trainer.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontweight='bold', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(trainer.val_accuracies, label='Val Accuracy', linewidth=2, color='green')
    axes[1].axhline(trainer.best_accuracy, color='r', linestyle='--', 
                   label=f'Best: {trainer.best_accuracy:.4f}', linewidth=2)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Validation Accuracy', fontweight='bold', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves")


def plot_confusion_matrix(targets, preds, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix")


def visualize_sample_residuals(dataset, output_dir, num_samples=4):
    """Visualize sample temporal median residuals."""
    fig, axes = plt.subplots(2, num_samples, figsize=(16, 8))
    
    for i in range(num_samples):
        # Get sample
        residual, label = dataset[i]
        residual_np = residual.squeeze(0).numpy()  # Remove channel dim
        
        # Show first frame
        axes[0, i].imshow(residual_np[0], cmap='seismic', vmin=-1, vmax=1)
        axes[0, i].set_title(f"Frame 0 - {'Fake' if label == 1 else 'Real'}", fontweight='bold')
        axes[0, i].axis('off')
        
        # Show middle frame
        mid_frame = len(residual_np) // 2
        axes[1, i].imshow(residual_np[mid_frame], cmap='seismic', vmin=-1, vmax=1)
        axes[1, i].set_title(f"Frame {mid_frame} - {'Fake' if label == 1 else 'Real'}", fontweight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle('Sample Temporal Median Residuals', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_residuals.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sample visualizations")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    
    # Configuration
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("Temporal Median CNN Pipeline")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*70)
    
    # Step 1: Collect video paths
    video_paths, labels = collect_video_paths(config)
    
    # Step 2: Split into train/val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        video_paths, labels, 
        test_size=(1-config.TRAIN_SPLIT),
        stratify=labels,
        random_state=42
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_paths)} videos")
    print(f"   Validation: {len(val_paths)} videos")
    
    # Step 3: Create datasets
    processor = TemporalMedianProcessor(
        max_frames=config.MAX_FRAMES,
        target_size=config.TARGET_SIZE
    )
    
    print("\n" + "="*70)
    print("Creating Training Dataset")
    print("="*70)
    train_dataset = VideoDataset(train_paths, train_labels, processor)
    
    print("\n" + "="*70)
    print("Creating Validation Dataset")
    print("="*70)
    val_dataset = VideoDataset(val_paths, val_labels, processor)
    
    # Step 4: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging; increase for performance
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Step 5: Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    if config.MODEL_TYPE == '3d_resnet':
        model = ResNet3D_18(num_classes=2)
        print("✓ Using ResNet3D-18")
    elif config.MODEL_TYPE == 'efficient_3d':
        model = Efficient2Plus1D(num_classes=2)
        print("✓ Using Efficient (2+1)D CNN")
    else:
        model = Simple3DCNN(num_classes=2)
        print("✓ Using Simple 3D CNN")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")
    
    # Step 6: Train
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    
    # Step 7: Final evaluation
    print("\n" + "="*70)
    print("Final Evaluation on Validation Set")
    print("="*70)
    
    val_loss, accuracy, auc, preds, targets, probs = trainer.validate()
    
    print(f"\n📊 Final Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
    print(f"   Loss: {val_loss:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(targets, preds, target_names=['Real', 'Fake']))
    
    # Step 8: Visualizations
    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)
    
    plot_training_curves(trainer, config.OUTPUT_DIR)
    plot_confusion_matrix(targets, preds, config.OUTPUT_DIR)
    visualize_sample_residuals(train_dataset, config.OUTPUT_DIR)
    
    # Step 9: Save results
    results = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'best_accuracy': float(trainer.best_accuracy),
        'num_train': len(train_dataset),
        'num_val': len(val_dataset),
        'model_type': config.MODEL_TYPE,
        'num_epochs': config.NUM_EPOCHS
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✅ All results saved to: {config.OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()