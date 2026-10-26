import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import MultiTaskScoliosisResNet

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
RESULTS_DIR = os.path.join(BASE_DIR, "results_external_validation")

os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ["Healthy", "1-degree (Mild)", "2-degree (Moderate)", "3-degree (Severe)"]
INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

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

class ExternalEvalDataset(Dataset):
    def __init__(self, df, path_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.path_map = path_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_name'])
        label = INTENSITY_MAP.get(str(row['intensity']), 0)
        img_path = self.path_map.get(img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, True
        except Exception:
            return torch.zeros((3, 224, 224)), label, False

def run_external_validation():
    print("=" * 75)
    print(f" بدء عملية الاختبار الخارجي (External Validation): {DEVICE}")
    print("=" * 75)

    if not os.path.exists(METADATA_CSV) or not os.path.exists(MODEL_PATH):
        print(f" الملفات المطلوبة غير متوفرة: {METADATA_CSV} أو {MODEL_PATH}")
        return

    full_df = pd.read_csv(METADATA_CSV)
    ext_df = full_df[full_df['split'] == 'external'].copy()

    if len(ext_df) == 0:
        print("⚠️ لم يتم العثور على صور خارجية في الميتاداتا الموحدة.")
        return

    print(f" إجمالي عدد صور التقييم الخارجي: {len(ext_df):,} صورة.")
    path_map = build_image_path_map(DATA_RAW_DIR)

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = ExternalEvalDataset(ext_df, path_map, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels, valids in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs['intensity'], dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred) * 100
    print("\n" + "=" * 70)
    print(f" الدقة في التقييم الخارجي (External Accuracy): {acc:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    run_external_validation()