import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from model import MultiTaskScoliosisResNet

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
SAVE_MODEL_PATH = "best_scoliosis_model_balanced.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 2.5e-4

INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
STAT_MAP = {"0-derece": 0, "1-derece": 1, "2-derece": 2, "3-derece": 3}
SHAPE_MAP = {"Normal": 0, "C": 1, "S": 2, "C-shape": 1, "S-shape": 2}
LOCATION_MAP = {"None": 0, "Upper Thoracic": 1, "Thoracic": 2, "Lumbar": 3, "Upper": 1}

# تعزيز بيانات محسّن للقضاء على Overfitting وتعميم التعلم
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_image_path_map(data_raw_dir):
    image_map = {}
    for root, _, files in os.walk(data_raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if f not in image_map:
                    image_map[f] = os.path.join(root, f)
    return image_map

class UnifiedScoliosisDataset(Dataset):
    def __init__(self, df, path_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])

        severity = INTENSITY_MAP.get(str(row.get('intensity', 'Healthy')), 0)
        stat_val = STAT_MAP.get(str(row.get('stat', '0-derece')), 0)
        shape_val = SHAPE_MAP.get(str(row.get('shape', 'Normal')), 0)
        loc_val = LOCATION_MAP.get(str(row.get('location', 'None')), 0)

        has_cobb = bool(row.get('has_cobb_gt', False))
        u_th = float(row.get('cobb_upper_thoracic', 0.0))
        m_th = float(row.get('cobb_main_thoracic', 0.0))
        lum_th = float(row.get('cobb_lumbar', 0.0))
        max_cobb = float(row.get('max_cobb_angle', 0.0))

        cobb_target = torch.tensor([u_th / 100.0, m_th / 100.0, lum_th / 100.0, max_cobb / 100.0], dtype=torch.float32)
        has_cobb_mask = torch.tensor(1.0 if has_cobb else 0.0, dtype=torch.float32)

        img_path = self.path_map.get(img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, severity, stat_val, shape_val, loc_val, cobb_target, has_cobb_mask
        except Exception:
            return torch.zeros((3, 224, 224)), severity, stat_val, shape_val, loc_val, cobb_target, torch.tensor(0.0)

def train_pipeline():
    print("=" * 75)
    print(f"[+] بدء عملية التدريب المحسّنة (Regularized High-Performance Training): {DEVICE}")
    print("=" * 75)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] ملف الميتاداتا النظيف غير موجود: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    df['computed_severity'] = df['intensity'].map(INTENSITY_MAP)
    path_map = build_image_path_map(DATA_RAW_DIR)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['computed_severity'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['computed_severity'])

    class_counts = np.bincount(train_df['computed_severity'], minlength=4)
    total_samples = len(train_df)
    weights = total_samples / (4.0 * class_counts)
    class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    train_dataset = UnifiedScoliosisDataset(train_df, path_map, transform=train_transforms)
    val_dataset = UnifiedScoliosisDataset(val_df, path_map, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskScoliosisResNet(pretrained=True).to(DEVICE)

    ce_intensity = nn.CrossEntropyLoss(weight=class_weights_tensor)
    ce_stat = nn.CrossEntropyLoss()
    ce_shape = nn.CrossEntropyLoss()
    ce_location = nn.CrossEntropyLoss()
    cobb_criterion = nn.SmoothL1Loss(reduction='none')

    # زيادة weight_decay لتقليل Overfitting
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=4e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for imgs, severities, stats, shapes, locs, cobb_targets, cobb_masks in train_loader:
            imgs = imgs.to(DEVICE)
            severities = severities.to(DEVICE)
            stats = stats.to(DEVICE)
            shapes = shapes.to(DEVICE)
            locs = locs.to(DEVICE)
            cobb_targets = cobb_targets.to(DEVICE)
            cobb_masks = cobb_masks.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)

            l_intensity = ce_intensity(outputs['intensity'], severities)
            l_stat = ce_stat(outputs['stat'], stats)
            l_shape = ce_shape(outputs['shape'], shapes)
            l_loc = ce_location(outputs['location'], locs)

            raw_cobb_loss = cobb_criterion(outputs['cobb_angles'], cobb_targets)
            masked_cobb_loss = (raw_cobb_loss * cobb_masks).sum() / (cobb_masks.sum() * 4 + 1e-8)

            total_loss = l_intensity + (0.2 * l_stat) + (0.2 * l_shape) + (0.2 * l_loc) + (0.4 * masked_cobb_loss)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * imgs.size(0)
            preds = torch.argmax(outputs['intensity'], dim=1)
            correct_train += (preds == severities).sum().item()
            total_train += severities.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, severities, _, _, _, _, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs['intensity'], dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(severities.numpy())

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean() * 100

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"   [---> تم حفظ النموذج الأفضل بمسار {SAVE_MODEL_PATH} مع Val F1: {best_val_f1:.4f}]")

    print("\n[+] اكتمل التدريب المحسّن النهائي وحُفظت الأوزان بملف:", SAVE_MODEL_PATH)

if __name__ == "__main__":
    train_pipeline()