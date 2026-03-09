# ============================================================
# Notebook 02: Preprocessing Pipeline Demo
# ============================================================
# Demonstrates the 5-step preprocessing pipeline:
# 1. Four-channel tensor construction
# 2. Bilateral filter (edge-preserving denoising)
# 3. CLAHE normalization
# 4. Letterbox resize
# 5. Segmentation masks
# ============================================================

# === Cell 1: Setup ===
import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.pipeline import PreprocessingPipeline, preprocess_xray
from src.preprocessing.segmentation import SegmentationRefinement, segment_objects
from src.preprocessing.augmentation import get_train_augmentation, get_val_augmentation
from src.utils.config import load_config

config = load_config("configs/default.yaml")

# === Cell 2: Load a Sample Image ===
# For demo, create a synthetic X-ray-like image
def create_synthetic_xray(h=480, w=640):
    """Create a synthetic X-ray image for demo purposes."""
    # Background: noisy gray (simulates baggage)
    bg = np.random.normal(180, 20, (h, w)).clip(0, 255).astype(np.uint8)

    # Add some rectangular "objects"
    # Object 1: Dense metallic object (dark)
    cv2.rectangle(bg, (100, 100), (250, 180), 40, -1)
    # Object 2: Organic object (light)
    cv2.ellipse(bg, (400, 200), (80, 50), 30, 0, 360, 200, -1)
    # Object 3: Long thin object (knife-like)
    pts = np.array([[300, 300], [500, 310], [505, 320], [300, 320]])
    cv2.fillPoly(bg, [pts], 60)
    # Object 4: Small dense object
    cv2.circle(bg, (150, 350), 25, 30, -1)

    # Add noise
    noise = np.random.normal(0, 5, (h, w)).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return bg

# Try loading a real image, fallback to synthetic
sample_path = None  # Set this to a real X-ray image path if available
if sample_path and os.path.exists(sample_path):
    image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
else:
    print("Using synthetic X-ray image for demo")
    image = create_synthetic_xray()

plt.figure(figsize=(10, 6))
plt.imshow(image, cmap='gray')
plt.title("Input X-Ray Image", fontsize=14)
plt.colorbar(label="Pixel Intensity")
plt.axis('off')
plt.show()

import os  # needed later

# === Cell 3: Step-by-Step Preprocessing ===
pipeline = PreprocessingPipeline(target_size=640)

# Step 1: 4-channel tensor
tensor_4ch = pipeline.construct_four_channel(image)
print(f"Step 1 - Four-Channel Tensor: {tensor_4ch.shape}")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
channel_names = ['-log(HE)', '-log(LE)', 'HE+LE (Density)', 'HE-LE (Material)']
for i, (ax, name) in enumerate(zip(axes, channel_names)):
    ax.imshow(tensor_4ch[:, :, i], cmap='viridis')
    ax.set_title(name, fontsize=11)
    ax.axis('off')
plt.suptitle("Step 1: Four-Channel Tensor Construction", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 2: Bilateral filter
tensor_filtered = pipeline.bilateral_filter(tensor_4ch)
print(f"Step 2 - Bilateral Filter: applied")

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i in range(4):
    axes[0, i].imshow(tensor_4ch[:, :, i], cmap='viridis')
    axes[0, i].set_title(f"Before - Ch {i}")
    axes[0, i].axis('off')
    axes[1, i].imshow(tensor_filtered[:, :, i], cmap='viridis')
    axes[1, i].set_title(f"After - Ch {i}")
    axes[1, i].axis('off')
plt.suptitle("Step 2: Bilateral Filter (Edge-Preserving Denoising)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 3: CLAHE
tensor_clahe = pipeline.clahe_normalize(tensor_filtered)
print(f"Step 3 - CLAHE Normalization: applied")

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(tensor_clahe[:, :, i], cmap='viridis')
    ax.set_title(f"CLAHE Ch {i}: {channel_names[i]}")
    ax.axis('off')
plt.suptitle("Step 3: CLAHE Contrast Enhancement", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 4: Letterbox resize
tensor_resized, scale, pad = pipeline.letterbox_resize(tensor_clahe, 640)
print(f"Step 4 - Letterbox Resize: {tensor_clahe.shape} → {tensor_resized.shape}")
print(f"  Scale: {scale:.3f}, Padding: {pad}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(tensor_clahe[:, :, 0], cmap='viridis')
axes[0].set_title(f"Before Resize {tensor_clahe.shape[:2]}")
axes[0].axis('off')
axes[1].imshow(tensor_resized[:, :, 0], cmap='viridis')
axes[1].set_title(f"After Letterbox {tensor_resized.shape[:2]}")
axes[1].axis('off')
plt.suptitle("Step 4: Letterbox Resize (Aspect-Ratio Preserving)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# === Cell 4: Full Pipeline in One Call ===
result = preprocess_xray(image)
print(f"\nFull pipeline: {image.shape} → {result.shape} (CHW format)")
assert result.shape == (4, 640, 640)
print("✓ Full pipeline test PASSED")

# === Cell 5: Segmentation Demo ===
print("\n🔬 Segmentation Refinement Demo")
segmentor = SegmentationRefinement(use_detectron2=False)

# Define bounding boxes for our synthetic objects
demo_bboxes = [
    [100, 100, 250, 180],   # Metallic rectangle
    [320, 150, 480, 250],   # Organic ellipse
    [300, 300, 505, 320],   # Knife-like object
    [125, 325, 175, 375],   # Small dense object
]

masks = segmentor.segment(image, demo_bboxes)

fig, axes = plt.subplots(1, len(demo_bboxes) + 1, figsize=(20, 4))
axes[0].imshow(image, cmap='gray')
for bbox in demo_bboxes:
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
    axes[0].add_patch(rect)
axes[0].set_title("Input + BBoxes")
axes[0].axis('off')

for i, (mask, bbox) in enumerate(zip(masks, demo_bboxes)):
    axes[i+1].imshow(mask, cmap='gray')
    axes[i+1].set_title(f"Mask {i+1}")
    axes[i+1].axis('off')

plt.suptitle("Step 5: Segmentation Refinement (Saliency + GrabCut)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# === Cell 6: Augmentation Demo ===
print("\n🎨 Augmentation Pipeline Demo")

# Create a 3-channel image for augmentation
image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
aug_pipeline = get_train_augmentation(image_size=640)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes[0, 0].imshow(image_3ch[:, :, ::-1])
axes[0, 0].set_title("Original")
axes[0, 0].axis('off')

for idx, ax in enumerate(axes.flatten()[1:], 1):
    augmented = aug_pipeline(
        image=image_3ch,
        bboxes=[[100, 100, 250, 180, 0]],  # xyxy + class
        class_labels=[0],
    )
    ax.imshow(augmented['image'][:, :, ::-1])
    ax.set_title(f"Augmented #{idx}")
    ax.axis('off')

plt.suptitle("Training Augmentations", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# === Cell 7: Preprocessing Benchmark ===
import time

print("\n⏱ Preprocessing Benchmark")
num_iterations = 50
start = time.time()
for _ in range(num_iterations):
    _ = preprocess_xray(image)
elapsed = time.time() - start

print(f"  Average time per image: {elapsed/num_iterations*1000:.1f} ms")
print(f"  Target (training): < 500 ms  {'✓' if elapsed/num_iterations < 0.5 else '⚠'}")
print(f"  Target (inference): < 100 ms  {'✓' if elapsed/num_iterations < 0.1 else '⚠'}")

print(f"\n{'='*60}")
print(f"  ✓ Preprocessing pipeline demo complete!")
print(f"{'='*60}")
print(f"\nNext step: Run notebook 03_model1_training.py")
