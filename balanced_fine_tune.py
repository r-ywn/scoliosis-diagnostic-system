import os
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import AdamW
from model import MultiTaskScoliosisResNet

# 1. إعدادات الجلسة والأجهزة
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL_PATH = "best_scoliosis_model.pth"
FINE_TUNED_MODEL_PATH = "best_scoliosis_model_balanced.pth"
METADATA_CSV = r"D:\Scoliosis_Project\data_processed\new_dataset_metadata.csv"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# 2. التحويلات البصرية المتقدمة المقاومة لاختلاف أجهزة الأشعة (Domain Adaptation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAutocontrast(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. بناء الفئة المخصصة لتحميل البيانات الجاهزة للضبط
class BalancedScoliosisDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        intensity_label = int(row['intensity_label'])

        # تحديد قيم كوب مرجعية متناسبة مع الدرجة الطبية لمعايرة رأس التراجع الحسابي
        if intensity_label == 0:  # Healthy
            target_max_cobb = float(np.random.uniform(0.0, 5.0))
        elif intensity_label == 1:  # 1-degree (Mild)
            target_max_cobb = float(np.random.uniform(11.0, 22.0))
        elif intensity_label == 2:  # 2-degree (Moderate)
            target_max_cobb = float(np.random.uniform(26.0, 42.0))
        else:  # 3-degree (Severe)
            target_max_cobb = float(np.random.uniform(46.0, 75.0))

        # افتراض زوايا فرعية متناسبة
        cobb_targets = torch.tensor([
            target_max_cobb * 0.3,  # Upper Thoracic
            target_max_cobb * 0.8,  # Main Thoracic
            target_max_cobb * 0.5,  # Lumbar
            target_max_cobb  # Max Cobb
        ], dtype=torch.float32)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        # استنباط التسميات الطبية المباشرة للشكل والموقع
        shape_label = 0 if intensity_label == 0 else 1
        location_label = 0 if intensity_label == 0 else 2

        return {
            "image": image,
            "cobb_angles": cobb_targets,
            "intensity": torch.tensor(intensity_label, dtype=torch.long),
            "shape": torch.tensor(shape_label, dtype=torch.long),
            "location": torch.tensor(location_label, dtype=torch.long)
        }


# 4. دالة إعادة التدريب والضبط المتوازن
def run_balanced_fine_tuning():
    print(f" [1/4] جاري قراءة الـ Metadata والتحضير للتدريب...")
    df = pd.read_csv(METADATA_CSV)

    train_df = df[df['split'].isin(['train', 'valid'])].copy()
    test_df = df[df['split'] == 'test'].copy()

    # إذا كانت عينة الاختبار صغيرة جداً نقتطع جزءاً من التدريب
    if len(test_df) < 100:
        test_df = df.sample(frac=0.2, random_state=42)
        train_df = df.drop(test_df.index)

    train_dataset = BalancedScoliosisDataset(train_df, transform=train_transforms)
    test_dataset = BalancedScoliosisDataset(test_df, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f" عينات التدريب والضبط: {len(train_dataset)} | عينات التحقق: {len(test_dataset)}")

    # تحميل النموذج السابق
    print(f" [2/4] تحميل النموذج المدرب سابقاً: {PRETRAINED_MODEL_PATH}")
    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        print(" تم تحميل الأوزان بنجاح.")
    else:
        print("⚠️ لم يتم العثور على النموذج السابق، سيتم البدء بأوزان ResNet18 القياسية.")

    # إعداد دوال الخسارة
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')

    print("\n [3/4] بدء عملية الضبط الدقيق المتوازن (Balanced Fine-Tuning)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            cobb_gt = batch["cobb_angles"].to(DEVICE)
            intensity_gt = batch["intensity"].to(DEVICE)
            shape_gt = batch["shape"].to(DEVICE)
            location_gt = batch["location"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)

            # حساب الخسارة الموزونة للمخرجات الخمسة
            loss_cobb = mse_loss_fn(outputs["cobb_angles"], cobb_gt)
            loss_intensity = ce_loss_fn(outputs["intensity"], intensity_gt)
            loss_shape = ce_loss_fn(outputs["shape"], shape_gt)
            loss_location = ce_loss_fn(outputs["location"], location_gt)

            total_loss = (loss_cobb * 1.5) + (loss_intensity * 2.0) + loss_shape + loss_location

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        train_loss /= len(train_loader)

        # التقييم على داتا التحقق
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(DEVICE)
                cobb_gt = batch["cobb_angles"].to(DEVICE)
                intensity_gt = batch["intensity"].to(DEVICE)
                shape_gt = batch["shape"].to(DEVICE)
                location_gt = batch["location"].to(DEVICE)

                outputs = model(images)
                loss_cobb = mse_loss_fn(outputs["cobb_angles"], cobb_gt)
                loss_intensity = ce_loss_fn(outputs["intensity"], intensity_gt)
                loss_shape = ce_loss_fn(outputs["shape"], shape_gt)
                loss_location = ce_loss_fn(outputs["location"], location_gt)

                v_loss = (loss_cobb * 1.5) + (loss_intensity * 2.0) + loss_shape + loss_location
                val_loss += v_loss.item()

        val_loss /= len(test_loader)

        print(f" Epoch [{epoch}/{EPOCHS}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), FINE_TUNED_MODEL_PATH)
            print(f"   تم حفظ أفضل نموذج متوازن بمسار: {FINE_TUNED_MODEL_PATH}")

    print("\n🎉 [4/4] اكتملت عملية الضبط الدقيق المتوازن بنجاح!")


if __name__ == "__main__":
    run_balanced_fine_tuning()