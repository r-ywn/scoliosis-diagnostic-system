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
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
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
    """فهرسة مسارات جميع الصور ديناميكياً بداخل data_raw عبر المجموعات الثلاث."""
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
        severity = INTENSITY_MAP.get(str(row['intensity']), 0)

        has_cobb = bool(row.get('has_cobb_gt', False))
        max_cobb = float(row.get('max_cobb_angle', 0.0))

        cobb_target = torch.tensor([0.0, 0.0, 0.0, max_cobb], dtype=torch.float32)
        has_cobb_mask = torch.tensor(1.0 if has_cobb else 0.0, dtype=torch.float32)

        img_path = self.path_map.get(img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, severity, cobb_target, has_cobb_mask
        except Exception:
            return torch.zeros((3, 224, 224)), severity, cobb_target, torch.tensor(0.0)


def train_pipeline():
    print("=" * 75)
    print(f"[+] بدء عملية التدريب الموحدة على المجموعات المعتمدة الثلاث: {DEVICE}")
    print("=" * 75)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] ملف الميتاداتا النظيف غير موجود: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    df['computed_severity'] = df['intensity'].map(INTENSITY_MAP)
    path_map = build_image_path_map(DATA_RAW_DIR)

    print(f"[+] إجمالي البيانات المعتمدة للتدريب والتقييم: {len(df):,} صورة من المصادر الثلاثة.")

    # تقسيم البيانات مع الحفاظ على التوازن الطبقي للشدة السريرية
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['computed_severity']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['computed_severity']
    )

    class_counts = np.bincount(train_df['computed_severity'], minlength=4)
    total_samples = len(train_df)
    weights = total_samples / (4.0 * class_counts)
    class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    print(f"[+] أوزان الفئات الموازنة: {weights.round(2)}")
    print(f"[+] صور التدريب: {len(train_df):,} | صور التحقق: {len(val_df):,} | صور الاختبار: {len(test_df):,}")

    train_dataset = UnifiedScoliosisDataset(train_df, path_map, transform=train_transforms)
    val_dataset = UnifiedScoliosisDataset(val_df, path_map, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiTaskScoliosisResNet(pretrained=True).to(DEVICE)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    cobb_criterion = nn.MSELoss(reduction='none')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, severities, cobb_targets, cobb_masks in train_loader:
            imgs = imgs.to(DEVICE)
            severities = severities.to(DEVICE)
            cobb_targets = cobb_targets.to(DEVICE)
            cobb_masks = cobb_masks.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss_ce = ce_criterion(outputs['intensity'], severities)

            # حساب خسارة زوايا كوب الموجهة (Masked Cobb Loss)
            raw_cobb_loss = cobb_criterion(outputs['cobb_angles'], cobb_targets)
            masked_cobb_loss = (raw_cobb_loss * cobb_masks).sum() / (cobb_masks.sum() * 4 + 1e-8)

            total_loss = loss_ce + (1.5 * masked_cobb_loss)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * imgs.size(0)
            preds = torch.argmax(outputs['intensity'], dim=1)
            correct_train += (preds == severities).sum().item()
            total_train += severities.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100

        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for imgs, severities, _, _ in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs['intensity'], dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(severities.numpy())

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean() * 100

        scheduler.step(val_f1)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"   [---> تم حفظ النموذج الأفضل بمسار {SAVE_MODEL_PATH} مع Val F1: {best_val_f1:.4f}]")

    print("\n[+] اكتمل التدريب الشامل الموحد بنجاح وحُفظت أوزان النموذج بملف:", SAVE_MODEL_PATH)


if __name__ == "__main__":
    train_pipeline()