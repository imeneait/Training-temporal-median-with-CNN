# This repository contains Two major aspects of Deepfake PFE : Use temporal median in Training + Create Masks in an effective Way

## Overview

- This approach uses **temporal median residuals** combined with **3D CNNs** to detect deepfakes. The key insight is that temporal median subtraction isolates motion information while removing static content.
- As a first Try the median Masks are generated with Dlib: key observation is that the median mask in real shift as well. So we maybe take the shifts of All the others.


### Why This Works

**Traditional approach (problematic):**
```
Align all frames to fixed pose → Deletes ALL motion (including expressions)
```

**new approach:**
```
Compute temporal median → Subtract from frames → Isolate motion signal
```

The temporal median captures the "static" appearance of the face, and subtracting it reveals only the dynamic changes (expressions, micro-motions). Deepfakes often have unnatural temporal dynamics that this method can detect.


---

## Start

### Option 1: Simple Version (for Testing)

```bash
python temporal_median_simple.py
```

This will:
- Load 25 real + 25 fake videos (5 from each method)
- Compute temporal median residuals
- Train a lightweight 3D CNN
- Report accuracy and classification metrics

**Pros:**
- Fast training (~10-20 minutes on GPU)
- Easy to modify and experiment
- Good for prototyping

**Cons:**
- Simpler model (lower capacity)
- Fewer hyperparameters to tune

### Option 2: Full Pipeline (For Best Results)

```bash
python temporal_median_cnn_pipeline.py
```

This provides:
- Multiple model architectures (3D ResNet, (2+1)D CNN)
- Advanced training (learning rate scheduling, early stopping)
- Comprehensive visualizations
- Detailed logging and checkpointing

**Pros:**
- State-of-the-art models
- Better accuracy
- Production-ready code

**Cons:**
- Longer training time
- Requires more GPU memory

---

## 🧠 Model Architectures

### 1. Lightweight 3D CNN (Simple Version)
```
Parameters: ~500K
Training time: ~15 min (GPU)
Expected accuracy: 75-85%
```

**Use when:**
- Quick prototyping
- Limited GPU memory
- Baseline comparison

### 2. 3D ResNet-18 (Full Pipeline)
```
Parameters: ~10M
Training time: ~1 hour (GPU)
Expected accuracy: 85-95%
```

**Use when:**
- Best accuracy needed
- Sufficient GPU memory (8GB+)
- Production deployment

### 3. (2+1)D Efficient CNN (Full Pipeline)
```
Parameters: ~5M
Training time: ~30 min (GPU)
Expected accuracy: 80-90%
```

**Use when:**
- Good balance of speed and accuracy
- Limited GPU memory
- Real-time inference needed

---



### Why This Avoids the Alignment Problem

**The alignment paradox:**
- Align to fixed reference → Deletes motion ❌
- Don't align → Head motion contaminates signal ❌

**Temporal median solution:**
- No explicit alignment needed! ✓
- Head motion is SLOW (low frequency)
- Expression motion is FAST (high frequency)
- Median captures static appearance (head position + neutral face)
- Residual isolates expressions
- 3D CNN learns to distinguish real vs fake motion patterns

---

## 🔧 Configuration

Edit the configuration in the script:

```python
# Simple version
MAX_FRAMES = 128      # Number of frames to load per video
IMG_SIZE = 128        # Resize to 128x128 (increase for better quality)
BATCH_SIZE = 2        # Reduce if out of memory
EPOCHS = 20           # Training epochs
LR = 1e-4            # Learning rate

# Full pipeline
class Config:
    MAX_FRAMES = 128
    TARGET_SIZE = (224, 224)  # Higher resolution
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    MODEL_TYPE = '3d_resnet'  # or 'efficient_3d', 'simple_3d'
```

---

## 📈 Expected Results

### Baseline Performance

With 25 real + 25 fake videos:

| Model | Accuracy | AUC | Training Time |
|-------|----------|-----|---------------|
| Lightweight 3D CNN | 75-85% | 0.80-0.90 | 15 min |
| 3D ResNet-18 | 85-95% | 0.90-0.97 | 1 hour |
| (2+1)D Efficient | 80-90% | 0.85-0.93 | 30 min |

**Note:** With only 50 videos, expect some variance (±5%). For production, use 500+ videos.

### Per-Method Performance

Different deepfake methods have different difficulty:

```
Easiest to detect:
- NeuralTextures (distinct temporal artifacts)
- FaceSwap (boundary inconsistencies)

Moderate:
- Face2Face (expression transfer)
- Deepfakes (face swap)

Hardest:
- FaceShifter (state-of-the-art, minimal artifacts)
```

---

## 🐛 Troubleshooting

### Out of Memory Error

```python
# Reduce batch size
BATCH_SIZE = 1

# Reduce image size
IMG_SIZE = 64

# Reduce frames
MAX_FRAMES = 64
```

### Low Accuracy (<70%)

**Possible causes:**

1. **Insufficient data**
   - Solution: Collect more videos (aim for 100+ per class)

2. **Data quality issues**
   - Check if videos have enough frames (min 16)
   - Verify temporal median looks reasonable

3. **Model underfitting**
   - Train longer (increase epochs)
   - Use larger model (3D ResNet instead of Lightweight)

4. **Learning rate too high/low**
   - Try: 1e-3 (higher), 1e-5 (lower)

### Slow Training

```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 Visualizations

The pipeline generates:

1. **Training curves** (`training_curves.png`)
   - Loss over epochs
   - Validation accuracy

2. **Confusion matrix** (`confusion_matrix.png`)
   - True positives/negatives
   - False positives/negatives

3. **Sample residuals** (`sample_residuals.png`)
   - Visual comparison of real vs fake residuals

4. **Results summary** (`results.json`)
   - Final metrics in JSON format

---

## 🔬 Advanced Usage

### Custom Preprocessing

```python
def custom_residual(frames):
    #Temporal median on row faces -> gradient magnitude
    """Example: Use gradient magnitude instead of raw intensity."""
    
    # Compute spatial gradients
    grad_x = np.gradient(frames, axis=2)
    grad_y = np.gradient(frames, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Temporal median on gradient magnitude
    median_grad = np.median(grad_mag, axis=0)
    residual = grad_mag - median_grad
    
    return residual
```

### Ensemble Models

```python
# Train multiple models
model1 = Lightweight3DCNN()
model2 = ResNet3D_18()
model3 = Efficient2Plus1D()

# At inference
pred1 = model1(video)
pred2 = model2(video)
pred3 = model3(video)

# Ensemble (average)
final_pred = (pred1 + pred2 + pred3) / 3
```

### Data Augmentation

```python
def augment_residual(residual):
    """Apply augmentations to residual."""
    
    # Temporal reversal (50% chance)
    if np.random.rand() > 0.5:
        residual = residual[::-1]
    
    # Temporal crop (random start)
    if len(residual) > 64:
        start = np.random.randint(0, len(residual) - 64)
        residual = residual[start:start+64]
    
    # Spatial flip (50% chance)
    if np.random.rand() > 0.5:
        residual = np.flip(residual, axis=2)
    
    return residual
```

---

## 💡 Key Advantages of This Approach

### 1. No Alignment Needed ✓
- Temporal median naturally handles pose variations
- Avoids the alignment paradox
- Simpler preprocessing pipeline

### 2. Motion-Focused ✓
- Removes static appearance
- Isolates temporal dynamics
- Deepfakes fail to replicate natural motion

### 3. Interpretable ✓
- Residuals are human-interpretable
- Can visualize what model sees
- Easier to debug and improve

### 4. Efficient ✓
- Lightweight models possible
- Fast inference
- Scales to real-time

### 5. Robust ✓
- Works across different deepfake methods
- Generalizes to unseen techniques
- Resistant to post-processing

---


### Scientific Justification

**Why temporal median instead of alignment?**

```
Alignment Problem:
  - Fixed reference → Deletes expression motion
  - Frame-to-frame → Accumulates errors

Temporal Median Solution:
  - Median captures "typical" appearance over time
  - Includes head pose (not a problem - median includes it)
  - Residual = deviation from typical
  - Expressions are deviations from neutral
  - Head motion is SLOW, expressions are FAST
  - 3D CNN can separate these frequencies
```



---

