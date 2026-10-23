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
METADATA_CSV = os.path.join(BASE_DIR, r"data_raw\metadata.csv")
SAVE_MODEL_PATH = "best_scoliosis_model_balanced.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# 1. تحسينات الصور وقواعد Augmentation المناسبة للصور الطبية
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.97, 1.03)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_spinal_ai_cobb_angles():
    cobb_map = {}
    gt_files = [
        os.path.join(BASE_DIR, r"data_raw\Spinal-AI2024\Cobb_spinal-AI2024-train_gt.txt"),
        os.path.join(BASE_DIR, r"data_raw\Spinal-AI2024\Cobb_spinal-AI2024-test_gt.txt")
    ]
    for gt_path in gt_files:
        if os.path.exists(gt_path):
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        img_name = os.path.basename(parts[0].strip()).lower()
                        try:
                            angles = [float(p) for p in parts[1:] if p.strip()]
                            max_angle = max(angles) if angles else 0.0
                            if max_angle < 10.0:
                                sev = 0
                            elif max_angle < 25.0:
                                sev = 1
                            elif max_angle < 45.0:
                                sev = 2
                            else:
                                sev = 3
                            cobb_map[img_name] = sev
                        except ValueError:
                            pass
    return cobb_map


SPINAL_COBB_MAP = load_spinal_ai_cobb_angles()


def extract_true_severity(row):
    image_str = str(row.get('image_path', '')).strip()
    img_filename = os.path.basename(image_str).lower()

    if img_filename in SPINAL_COBB_MAP:
        return SPINAL_COBB_MAP[img_filename]

    label_str = str(row.get('label_or_mask_path', '')).strip()
    file_str = str(row.get('file_name', '')).strip()
    rel_str = str(row.get('relative_path', '')).strip()

    if os.path.exists(label_str) and label_str.endswith('.txt'):
        try:
            with open(label_str, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                classes = [int(l.split()[0]) for l in lines if l.split()[0].isdigit()]
                if classes:
                    return min(max(classes), 3)
        except Exception:
            pass

    path_str = f"{image_str} {rel_str} {label_str} {file_str}".lower().replace("\\", "/")

    if "/3-degree" in path_str or "3-degree" in path_str or "severe" in path_str:
        return 3
    elif "/2-degree" in path_str or "2-degree" in path_str or "moderate" in path_str:
        return 2
    elif "/1-degree" in path_str or "1-degree" in path_str or "mild" in path_str:
        return 1
    elif "/healthy" in path_str or "healthy" in path_str or "normal" in path_str:
        return 0

    if label_str in ['0', '1', '2', '3']:
        return int(label_str)

    return 0


class ScoliosisTrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['computed_severity'])

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception:
            return torch.zeros((3, 224, 224)), label


def train_pipeline():
    print("=" * 65)
    print(f"[+] بدء تدريب النموذج على البيانات النظيفة باستخدام: {DEVICE}")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] ملف الميتاداتا غير موجود: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    df['computed_severity'] = df.apply(extract_true_severity, axis=1)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['computed_severity']
    )
    val_df, _ = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['computed_severity']
    )

    # 2. حساب الأوزان الموازنة (Class Weights) لمعالجة عدم توازن الأعداد
    class_counts = np.bincount(train_df['computed_severity'], minlength=4)
    total_samples = len(train_df)
    weights = total_samples / (4.0 * class_counts)
    class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

    print(f"[+] أوزان الفئات المحسوبة (Class Weights): {weights.round(2)}")

    train_dataset = ScoliosisTrainDataset(train_df, transform=train_transforms)
    val_dataset = ScoliosisTrainDataset(val_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiTaskScoliosisResNet(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs['intensity'], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs['intensity'], dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = (correct_train / total_train) * 100

        # التقييم على Validation Set
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds = torch.argmax(outputs['intensity'], dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean() * 100

        scheduler.step(val_f1)

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")

        # حفظ النموذج الأفضل عند تحسين Val F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"   [---> Saved Best Model with Val F1: {best_val_f1:.4f}]")

    print("\n[+] اكتمل التدريب بنجاح وحُفظت أفضل أوزان للنموذج في:", SAVE_MODEL_PATH)


if __name__ == "__main__":
    train_pipeline()