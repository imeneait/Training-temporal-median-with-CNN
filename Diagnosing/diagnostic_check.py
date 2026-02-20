"""
Diagnostic Script: Find Why Model Isn't Learning
=================================================

This will help us identify the problem:
1. Check if data is loaded correctly
2. Verify temporal median is computed properly
3. Inspect model gradients
4. Visualize what the model sees
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Your paths
REAL_DIR = "/Users/mac/Documents/FMT/original-faces"
FAKE_DIR = "/Users/mac/Documents/FMT/fake-faces/Deepfakes"  # Start with one method

def load_video(video_dir, max_frames=128, target_size=(128, 128)):
    """Load video frames and compute temporal median residual."""
    
    if not os.path.exists(video_dir):
        print(f"⚠ Directory not found: {video_dir}")
        return None
    
    files = sorted([f for f in os.listdir(video_dir) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))])[:max_frames]
    
    if len(files) < 16:
        print(f"⚠ Only {len(files)} files found")
        return None
    
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
    
    # Normalize
    residual = residual / 127.5
    
    return frames, median, residual


def diagnostic_check():
    """Run comprehensive diagnostics."""
    
    print("="*70)
    print("DIAGNOSTIC CHECK")
    print("="*70)
    
    # Check 1: Can we load videos?
    print("\n1. CHECKING VIDEO LOADING...")
    print("-" * 70)
    
    real_subdirs = [d for d in os.listdir(REAL_DIR) if os.path.isdir(os.path.join(REAL_DIR, d))]
    fake_subdirs = [d for d in os.listdir(FAKE_DIR) if os.path.isdir(os.path.join(FAKE_DIR, d))]
    
    print(f"Real directories found: {len(real_subdirs)}")
    print(f"Fake directories found: {len(fake_subdirs)}")
    
    if len(real_subdirs) == 0 or len(fake_subdirs) == 0:
        print("❌ NO VIDEOS FOUND! Check your paths.")
        return
    
    # Load one real and one fake video
    real_path = os.path.join(REAL_DIR, real_subdirs[0])
    fake_path = os.path.join(FAKE_DIR, fake_subdirs[0])
    
    print(f"\nTesting with:")
    print(f"  Real: {real_path}")
    print(f"  Fake: {fake_path}")
    
    real_frames, real_median, real_residual = load_video(real_path)
    fake_frames, fake_median, fake_residual = load_video(fake_path)
    
    if real_frames is None or fake_frames is None:
        print("❌ FAILED TO LOAD VIDEOS!")
        return
    
    print(f"\n✓ Real video: {real_frames.shape}")
    print(f"✓ Fake video: {fake_frames.shape}")
    
    # Check 2: Are residuals different?
    print("\n2. CHECKING RESIDUAL STATISTICS...")
    print("-" * 70)
    
    print(f"Real residual - Min: {real_residual.min():.3f}, Max: {real_residual.max():.3f}, "
          f"Mean: {real_residual.mean():.3f}, Std: {real_residual.std():.3f}")
    print(f"Fake residual - Min: {fake_residual.min():.3f}, Max: {fake_residual.max():.3f}, "
          f"Mean: {fake_residual.mean():.3f}, Std: {fake_residual.std():.3f}")
    
    # Check if they're too similar
    real_std = real_residual.std()
    fake_std = fake_residual.std()
    
    if abs(real_std - fake_std) < 0.01:
        print("⚠ WARNING: Real and fake residuals are very similar!")
        print("  This will make learning difficult.")
    
    # Check if residuals are too small
    if real_std < 0.05:
        print("⚠ WARNING: Residuals are very small (low signal)!")
        print("  Try different normalization or preprocessing.")
    
    # Check 3: Visualize
    print("\n3. GENERATING VISUALIZATIONS...")
    print("-" * 70)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Real video
    axes[0, 0].imshow(real_frames[0], cmap='gray')
    axes[0, 0].set_title('Real Frame 0', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(real_frames[len(real_frames)//2], cmap='gray')
    axes[0, 1].set_title('Real Frame Mid', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(real_median, cmap='gray')
    axes[0, 2].set_title('Real Temporal Median', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(real_residual[len(real_residual)//2], cmap='seismic', vmin=-1, vmax=1)
    axes[0, 3].set_title('Real Residual', fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Fake video
    axes[1, 0].imshow(fake_frames[0], cmap='gray')
    axes[1, 0].set_title('Fake Frame 0', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fake_frames[len(fake_frames)//2], cmap='gray')
    axes[1, 1].set_title('Fake Frame Mid', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(fake_median, cmap='gray')
    axes[1, 2].set_title('Fake Temporal Median', fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(fake_residual[len(fake_residual)//2], cmap='seismic', vmin=-1, vmax=1)
    axes[1, 3].set_title('Fake Residual', fontweight='bold')
    axes[1, 3].axis('off')
    
    # Row 3: Histograms
    axes[2, 0].hist(real_residual.flatten(), bins=100, alpha=0.7, color='blue')
    axes[2, 0].set_title('Real Residual Distribution', fontweight='bold')
    axes[2, 0].set_xlabel('Residual Value')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].hist(fake_residual.flatten(), bins=100, alpha=0.7, color='red')
    axes[2, 1].set_title('Fake Residual Distribution', fontweight='bold')
    axes[2, 1].set_xlabel('Residual Value')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Temporal variance (how much motion)
    real_temporal_var = np.var(real_residual, axis=0)
    fake_temporal_var = np.var(fake_residual, axis=0)
    
    axes[2, 2].imshow(real_temporal_var, cmap='hot')
    axes[2, 2].set_title('Real Temporal Variance\n(Motion Map)', fontweight='bold')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(fake_temporal_var, cmap='hot')
    axes[2, 3].set_title('Fake Temporal Variance\n(Motion Map)', fontweight='bold')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostic_visualization.png', dpi=200, bbox_inches='tight')
    print("✓ Saved: diagnostic_visualization.png")
    plt.close()
    
    # Check 4: Model gradient check
    print("\n4. CHECKING MODEL GRADIENTS...")
    print("-" * 70)
    
    # Simple model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(1, 8, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(8, 2)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = TinyModel()
    criterion = nn.CrossEntropyLoss()
    
    # Create batch
    real_tensor = torch.from_numpy(real_residual).float().unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
    fake_tensor = torch.from_numpy(fake_residual).float().unsqueeze(0).unsqueeze(0)
    
    batch = torch.cat([real_tensor, fake_tensor], dim=0)  # (2, 1, T, H, W)
    labels = torch.tensor([0, 1])
    
    # Forward pass
    output = model(batch)
    loss = criterion(output, labels)
    
    print(f"Output logits: {output.detach().numpy()}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: gradient norm = {grad_norm:.6f}")
            if grad_norm > 1e-6:
                has_gradients = True
    
    if not has_gradients:
        print("❌ NO GRADIENTS! Model cannot learn.")
    else:
        print("✓ Gradients are flowing")
    
    # Check 5: Data variance
    print("\n5. CHECKING DATA VARIANCE...")
    print("-" * 70)
    
    # Load multiple videos
    real_stds = []
    fake_stds = []
    
    for i in range(min(5, len(real_subdirs))):
        path = os.path.join(REAL_DIR, real_subdirs[i])
        result = load_video(path)
        if result is not None:
            _, _, residual = result
            real_stds.append(residual.std())
    
    for i in range(min(5, len(fake_subdirs))):
        path = os.path.join(FAKE_DIR, fake_subdirs[i])
        result = load_video(path)
        if result is not None:
            _, _, residual = result
            fake_stds.append(residual.std())
    
    print(f"Real residual stds: {[f'{s:.3f}' for s in real_stds]}")
    print(f"Fake residual stds: {[f'{s:.3f}' for s in fake_stds]}")
    
    real_mean_std = np.mean(real_stds)
    fake_mean_std = np.mean(fake_stds)
    
    print(f"\nMean std - Real: {real_mean_std:.3f}, Fake: {fake_mean_std:.3f}")
    print(f"Difference: {abs(real_mean_std - fake_mean_std):.3f}")
    
    if abs(real_mean_std - fake_mean_std) < 0.02:
        print("⚠ WARNING: Real and fake have very similar variance!")
        print("  The model may struggle to distinguish them.")
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    issues = []
    
    if len(real_subdirs) < 10 or len(fake_subdirs) < 10:
        issues.append("⚠ Not enough videos (need 25+ each)")
    
    if real_std < 0.05:
        issues.append("⚠ Residuals too small (weak signal)")
    
    if abs(real_std - fake_std) < 0.01:
        issues.append("⚠ Real and fake residuals too similar")
    
    if not has_gradients:
        issues.append("❌ No gradients (model cannot learn)")
    
    if len(issues) == 0:
        print("✅ No obvious issues detected!")
        print("\nPossible causes of 50% accuracy:")
        print("  1. Model too simple (try larger model)")
        print("  2. Need more training epochs")
        print("  3. Learning rate too high/low")
        print("  4. Batch size too small")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    
    print("\nNext steps:")
    print("  1. Check diagnostic_visualization.png")
    print("  2. Verify residuals show visible differences")
    print("  3. Try the fixes below if issues detected")


if __name__ == "__main__":
    diagnostic_check()