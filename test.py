
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm

# Import model definition from your training script
# (Make sure the file train_segmentation_optimized.py is still in /content)
from train_segmentation_optimized import SegmentationHeadConvNeXt, value_map, n_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
model_path = '/content/train_stats/segmentation_head.pth'   # your trained model
test_dir = '/content/Offroad_Segmentation_Training_Dataset/testImages'   # where you unzipped
output_dir = '/content/predictions'
os.makedirs(output_dir, exist_ok=True)

# Image size (must match training)
h, w = 266, 476

# Load model (embedding dim = 768 for base backbone)
embed_dim = 768
classifier = SegmentationHeadConvNeXt(in_channels=embed_dim, out_channels=n_classes, tokenW=w//14, tokenH=h//14)
classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
classifier = classifier.to(device)
classifier.eval()

# Load DINOv2 backbone (base, because that's what you used)
backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
backbone.eval().to(device)

# Transform (same as validation)
transform = A.Compose([
    A.Resize(h, w),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Process all test images
test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
print(f"Found {len(test_images)} test images.")
for img_name in tqdm(test_images):
    img_path = os.path.join(test_dir, img_name)
    image = np.array(Image.open(img_path).convert("RGB"))

    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = classifier(features)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # Map class indices back to original IDs (0,100,200,...)
    inv_map = {v:k for k,v in value_map.items()}
    original_pred = np.vectorize(inv_map.get)(pred).astype(np.uint16)

    Image.fromarray(original_pred).save(os.path.join(output_dir, img_name))

print(f"Predictions saved to {output_dir}")
