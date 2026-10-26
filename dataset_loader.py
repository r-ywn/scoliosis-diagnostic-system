import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
STAT_MAP = {"0-derece": 0, "1-derece": 1, "2-derece": 2, "3-derece": 3}
SHAPE_MAP = {"Normal": 0, "C": 1, "S": 2}
LOCATION_MAP = {"None": 0, "Upper Thoracic": 1, "Thoracic": 2, "Lumbar": 3}

def build_image_path_map(data_raw_dir):
    """فهرسة مكان جميع الصور في data_raw ديناميكياً."""
    image_map = {}
    for root, _, files in os.walk(data_raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if f not in image_map:
                    image_map[f] = os.path.join(root, f)
    return image_map

class ScoliosisDataset(Dataset):
    def __init__(self, df, data_raw_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_path_map = build_image_path_map(data_raw_dir)

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

        # التعامل الآمن مع زاوية كوب القصوى
        max_cobb = float(row.get('max_cobb_angle', 0.0))
        cobb_angles = torch.tensor([0.0, 0.0, 0.0, max_cobb], dtype=torch.float32)

        intensity_str = str(row.get('intensity', 'Healthy'))
        stat_str = str(row.get('stat', '0-derece'))
        shape_str = str(row.get('shape', 'Normal'))
        loc_str = str(row.get('location', 'None'))

        return {
            "image": image,
            "cobb_angles": cobb_angles,
            "intensity": torch.tensor(INTENSITY_MAP.get(intensity_str, 0), dtype=torch.long),
            "stat": torch.tensor(STAT_MAP.get(stat_str, 0), dtype=torch.long),
            "shape": torch.tensor(SHAPE_MAP.get(shape_str, 0), dtype=torch.long),
            "location": torch.tensor(LOCATION_MAP.get(loc_str, 0), dtype=torch.long)
        }

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

def get_dataloaders(batch_size=32, num_workers=0):
    base_dir = r"D:\Scoliosis_Project"
    csv_file = os.path.join(base_dir, r"data_processed\clean_unified_metadata.csv")
    raw_dir = os.path.join(base_dir, "data_raw")

    df = pd.read_csv(csv_file)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    train_dataset = ScoliosisDataset(df=train_df, data_raw_dir=raw_dir, transform=train_transforms)
    test_dataset = ScoliosisDataset(df=test_df, data_raw_dir=raw_dir, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader