# Offroad Semantic Segmentation - Solution

## Overview
Semantic segmentation solution for Duality AI's Offroad Autonomy Segmentation Challenge. Uses a **DINOv2-Small (ViT-S/14)** backbone with a lightweight ConvNeXt‑style decoder, trained on synthetic desert terrain imagery. The model was trained on the provided dataset and evaluated on unseen test images.

**Best Validation IoU:** 0.5039
**Pixel Accuracy:** 0.89  
**Inference speed:** ~88ms per image (unoptimized, T4 GPU)

---

## Architecture
- **Backbone:** DINOv2‑Small (`dinov2_vits14`, frozen). Extracts patch tokens of dimension 384.
- **Decoder:** A custom segmentation head with:
  - Reshape of patch tokens to a 2D feature map.
  - Two depthwise separable convolution blocks (7×7, groups=128) with GELU activation.
  - Final 1×1 convolution to produce 10 class logits.
- **Output:** Logits upsampled to original input resolution using bilinear interpolation.

*No multi‑scale feature fusion or LoRA fine‑tuning was used in this baseline.*

---

## Training Details

### Data
- **Train:** 2,857 RGB images with pixel‑wise masks.
- **Validation:** 317 images.
- **Classes:** 10 (Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky).

### Augmentations
Applied on the fly using `albumentations`:
- Random horizontal flip (p=0.5)
- Rotation ±10° (p=0.5)
- Color jitter (brightness, contrast, saturation, hue; p=0.5)
- Resize to 266×476 (compatible with DINOv2’s patch size of 14)
- Normalization using ImageNet statistics

Validation images were only resized and normalized.

### Loss & Optimizer
- **Loss:** Cross‑entropy (weighted by class frequencies to mitigate imbalance).
- **Optimizer:** SGD with momentum 0.9, learning rate 1e‑4.
- **Batch size:** 2 (later increased to 16 with mixed precision on GPU).
- **Epochs:** 20 (best model selected based on validation IoU).

### Hardware & Environment
- **Hardware:** Google Colab with Tesla T4 GPU (15 GB VRAM).
- **Software:** Python 3.10, PyTorch 2.x, Albumentations, OpenCV, etc.
- **Mixed precision:** Used `torch.cuda.amp` to accelerate training.

---

## Results

### Validation Metrics
After 20 epochs, the model achieved:

| Metric            | Value  |
|-------------------|--------|
| Mean IoU          | 0.5039 |
| Pixel Accuracy    | 0.89   |

**Loss and IoU curves** are shown below (plots from `train_stats/`):

Check out the Metrics from My Drive : https://drive.google.com/drive/folders/1lCkmOzgRlpPypQFTQWvrNH-91KBUBVBz?usp=sharing

### Per‑class IoU (Validation)| 
Class	IoU
Trees	 - 0.61
Lush Bushes - 	0.55
Dry Grass	 - 0.52
Dry Bushes - 	0.48
Ground Clutter	 - 0.42
Flowers - 	0.46
Logs	 - 0.39
Rocks - 	0.44
Landscape	 - 0.63
Sky - 	0.74





---

## Environment Setup

bash
# Python 3.10+ with PyTorch 2.x and CUDA
pip install torch torchvision
pip install albumentations opencv-python matplotlib tqdm pillow


How to use it?
from train import SegmentationHeadConvNeXt
import torch

model = SegmentationHeadConvNeXt(in_channels=384, out_channels=10, tokenW=476//14, tokenH=266//14)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))


