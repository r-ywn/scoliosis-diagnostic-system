import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. تحديد القواميس لتحويل التصنيفات النصية إلى أرقام
INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
STAT_MAP = {"0-derece": 0, "1-derece": 1, "2-derece": 2, "3-derece": 3}
SHAPE_MAP = {"Normal": 0, "C": 1, "S": 2}
LOCATION_MAP = {"None": 0, "Upper Thoracic": 1, "Thoracic": 2, "Lumbar": 3}


class ScoliosisDataset(Dataset):
    def __init__(self, csv_file, raw_data_dir, transform=None):
        # keep_default_na=False تمنع تحويل كلمة "None" إلى NaN
        self.df = pd.read_csv(csv_file, keep_default_na=False)
        self.raw_data_dir = raw_data_dir
        self.transform = transform

        print("جاري فحص وتحديد مسارات الصور...")
        self.image_path_map = {}
        all_image_paths = glob.glob(os.path.join(raw_data_dir, "Spinal-AI2024-subset*", "*.jpg")) + \
                          glob.glob(os.path.join(raw_data_dir, "Spinal-AI2024-subset*", "*.png"))

        for path in all_image_paths:
            filename = os.path.basename(path)
            self.image_path_map[filename] = path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])

        img_path = self.image_path_map.get(img_name)
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"الصورة غير موجودة: {img_name}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        cobb_angles = torch.tensor([
            float(row['cobb_upper_thoracic']),
            float(row['cobb_main_thoracic']),
            float(row['cobb_lumbar']),
            float(row['max_cobb_angle'])
        ], dtype=torch.float32)

        # معالجة حذرة للتأكد من المطبقة مع القواميس
        loc_str = str(row['location']) if row['location'] else "None"
        shape_str = str(row['shape']) if row['shape'] else "Normal"
        intensity_str = str(row['intensity']) if row['intensity'] else "Healthy"
        stat_str = str(row['stat']) if row['stat'] else "0-derece"

        intensity = torch.tensor(INTENSITY_MAP.get(intensity_str, 0), dtype=torch.long)
        stat = torch.tensor(STAT_MAP.get(stat_str, 0), dtype=torch.long)
        shape = torch.tensor(SHAPE_MAP.get(shape_str, 0), dtype=torch.long)
        location = torch.tensor(LOCATION_MAP.get(loc_str, 0), dtype=torch.long)

        return {
            "image": image,
            "cobb_angles": cobb_angles,
            "intensity": intensity,
            "stat": stat,
            "shape": shape,
            "location": location
        }


# 2. تحديد تحويلات الصور (Resizing & Normalization)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 3. دالة إنشاء الـ DataLoaders
def get_dataloaders(batch_size=32, num_workers=2):
    processed_dir = r"D:\Scoliosis_Project\data_processed"
    raw_dir = r"D:\Scoliosis_Project\data_raw\Spinal-AI2024"

    train_dataset = ScoliosisDataset(
        csv_file=os.path.join(processed_dir, "metadata_train.csv"),
        raw_data_dir=raw_dir,
        transform=train_transforms
    )

    test_dataset = ScoliosisDataset(
        csv_file=os.path.join(processed_dir, "metadata_test.csv"),
        raw_data_dir=raw_dir,
        transform=test_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    print("جاري فحص وتحميل الـ DataLoader...")
    train_loader, test_loader = get_dataloaders(batch_size=16, num_workers=0)

    # سحب أول دفعة (Batch) للتأكد
    batch = next(iter(train_loader))

    print("\n تم فحص الـ DataLoader بنجاح!")
    print(f" أبعاد دفعة الصور (Image Batch): {batch['image'].shape}")
    print(f" أبعاد زوايا كوب (Cobb Angles Batch): {batch['cobb_angles'].shape}")
    print(f"️ أبعاد الشدة (Intensity Batch): {batch['intensity'].shape}")
    print(f"️ أبعاد الشكل (Shape Batch): {batch['shape'].shape}")