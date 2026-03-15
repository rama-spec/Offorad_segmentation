
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Enable cudnn benchmark for fixed input sizes
torch.backends.cudnn.benchmark = True

plt.switch_backend('Agg')

# ========== Utility Functions ==========
def save_image(img, filename):
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])

# ========== Mask Conversion ==========
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

# ========== Dataset ==========
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_pil = Image.open(mask_path)
        mask = convert_mask(mask_pil)
        mask = np.array(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        return image, mask

# ========== Model ==========
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ========== Metrics ==========
def compute_iou(pred, target, num_classes=10, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)

def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()

def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    iou_scores = []; dice_scores = []; pixel_accuracies = []
    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            iou = compute_iou(outputs, labels, num_classes=num_classes)
            dice = compute_dice(outputs, labels, num_classes=num_classes)
            pixel_acc = compute_pixel_accuracy(outputs, labels)
            iou_scores.append(iou); dice_scores.append(dice); pixel_accuracies.append(pixel_acc)
    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)

# ========== Plotting Functions ==========
def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val'); plt.title('Loss'); plt.legend(); plt.grid()
    plt.subplot(1,2,2); plt.plot(history['train_pixel_acc'], label='train'); plt.plot(history['val_pixel_acc'], label='val'); plt.title('Pixel Acc'); plt.legend(); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'training_curves.png')); plt.close()
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.plot(history['train_iou'], label='Train IoU'); plt.title('Train IoU'); plt.grid()
    plt.subplot(1,2,2); plt.plot(history['val_iou'], label='Val IoU'); plt.title('Val IoU'); plt.grid()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'iou_curves.png')); plt.close()
    print("Plots saved.")

def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write(f"Final Val IoU: {history['val_iou'][-1]:.4f}\nBest Val IoU: {max(history['val_iou']):.4f}\n")
    print(f"Metrics saved to {filepath}")

# ========== Main ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # INCREASE BATCH SIZE (try 8, then 16 if memory allows)
    batch_size = 8
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)
    lr = 1e-4
    n_epochs = 20

    # Local dataset
    base_path = '/content'
    data_dir = os.path.join(base_path, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(base_path, 'Offroad_Segmentation_Training_Dataset', 'val')
    output_dir = os.path.join(base_path, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    train_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    trainset = MaskDataset(data_dir=data_dir, transform=train_transform)
    # INCREASE NUM_WORKERS (try 4, 8) and use persistent_workers for faster restarts
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valset = MaskDataset(data_dir=val_dir, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Train samples: {len(trainset)}, Val samples: {len(valset)}")
    print(f"Batch size: {batch_size}, Workers: 4")

    # Backbone (you can try 'base' if you have enough GPU memory)
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.eval()
    backbone_model.to(device)

    # Get embedding dimension
    imgs, _ = next(iter(train_loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
    n_embedding = output.shape[2]

    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)

    history = {k:[] for k in ['train_loss','val_loss','train_iou','val_iou','train_dice','val_dice','train_pixel_acc','val_pixel_acc']}

    for epoch in range(n_epochs):
        classifier.train()
        train_losses = []
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            loss = loss_fct(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                labels = labels.squeeze(dim=1).long()
                loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())

        train_iou, train_dice, train_acc = evaluate_metrics(classifier, backbone_model, train_loader, device, num_classes=n_classes, show_progress=False)
        val_iou, val_dice, val_acc = evaluate_metrics(classifier, backbone_model, val_loader, device, num_classes=n_classes, show_progress=False)

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_acc)
        history['val_pixel_acc'].append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={history['train_loss'][-1]:.4f}, Val Loss={history['val_loss'][-1]:.4f}, Val IoU={val_iou:.4f}")

    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    torch.save(classifier.state_dict(), os.path.join(output_dir, 'segmentation_head.pth'))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
