import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, mean_absolute_error
from model import MultiTaskScoliosisResNet

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INTENSITY_MAP = {"Healthy": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
CLASSES = ["Healthy", "Mild", "Moderate", "Severe"]

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

class UnifiedDataset(Dataset):
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

        has_cobb = bool(row.get('has_cobb_gt', False))
        u_th = float(row.get('cobb_upper_thoracic', 0.0))
        m_th = float(row.get('cobb_main_thoracic', 0.0))
        lum_th = float(row.get('cobb_lumbar', 0.0))
        max_cobb = float(row.get('max_cobb_angle', 0.0))

        cobb_target = torch.tensor([u_th, m_th, lum_th, max_cobb], dtype=torch.float32)
        has_cobb_mask = torch.tensor(1.0 if has_cobb else 0.0, dtype=torch.float32)

        img_path = self.path_map.get(img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, severity, cobb_target, has_cobb_mask
        except Exception:
            return torch.zeros((3, 224, 224)), severity, cobb_target, torch.tensor(0.0)

def plot_and_save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 11})
    plt.title(title, fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

def evaluate_and_plot():
    print("=" * 70)
    print(f"[+] بدء التقييم الموحد الشامل لكافة الأقسام (Train, Val, Test): {DEVICE}")
    print("=" * 70)

    if not os.path.exists(METADATA_CSV) or not os.path.exists(MODEL_PATH):
        print("[ERROR] تعذر العثور على الميتاداتا النظيفة أو ملف الأوزان!")
        return

    df = pd.read_csv(METADATA_CSV)
    df['computed_severity'] = df['intensity'].map(INTENSITY_MAP)
    path_map = build_image_path_map(DATA_RAW_DIR)

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['computed_severity'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['computed_severity'])

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    def get_predictions(subset_df):
        dataset = UnifiedDataset(subset_df, path_map, transform=val_transforms)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds, targets, probs = [], [], []
        cobb_preds_list, cobb_targets_list = [], []

        with torch.no_grad():
            for imgs, labels, cobb_gt, cobb_masks in loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)

                logits = outputs['intensity']
                pr = torch.softmax(logits, dim=1)
                p = torch.argmax(pr, dim=1)

                preds.extend(p.cpu().numpy())
                targets.extend(labels.numpy())
                probs.extend(pr.cpu().numpy())

                # إلغاء التنميط لاستعادة درجات زوايا كوب الحقيقية
                cobb_pred_deg = outputs['cobb_angles'].cpu().numpy() * 100.0
                mask_indices = (cobb_masks == 1.0).numpy()

                if np.sum(mask_indices) > 0:
                    cobb_preds_list.append(cobb_pred_deg[mask_indices])
                    cobb_targets_list.append(cobb_gt.numpy()[mask_indices])

        cobb_preds_arr = np.vstack(cobb_preds_list) if cobb_preds_list else None
        cobb_targets_arr = np.vstack(cobb_targets_list) if cobb_targets_list else None

        return np.array(targets), np.array(preds), np.array(probs), cobb_targets_arr, cobb_preds_arr

    # 1. حساب وحفظ مصفوفات الارتباك الموحدة
    print("[1/4] حساب مصفوفة ارتباك التدريب...")
    y_tr_true, y_tr_pred, _, _, _ = get_predictions(train_df)
    plot_and_save_cm(y_tr_true, y_tr_pred, "Confusion Matrix (Training Set)", "confusion_matrix_train.png")

    print("[2/4] حساب مصفوفة ارتباك التحقق...")
    y_val_true, y_val_pred, _, _, _ = get_predictions(val_df)
    plot_and_save_cm(y_val_true, y_val_pred, "Confusion Matrix (Validation Set)", "confusion_matrix_val.png")

    print("[3/4] حساب مصفوفة ارتباك الاختبار وخسارة زوايا كوب...")
    y_ts_true, y_ts_pred, probs_ts, cobb_gt_ts, cobb_pred_ts = get_predictions(test_df)
    plot_and_save_cm(y_ts_true, y_ts_pred, "Confusion Matrix (Clean Test Set)", "confusion_matrix_test.png")

    if cobb_pred_ts is not None:
        mae_max = mean_absolute_error(cobb_gt_ts[:, 3], cobb_pred_ts[:, 3])
        print(f"   [+] متوسط الخطأ المطلق لزاوية كوب القصوى (Test Max Cobb MAE): {mae_max:.2f}°")

    # 2. رسم مقاييس الأداء المحدّثة (Precision, Recall, F1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_ts_true, y_ts_pred, average=None)
    x = np.arange(len(CLASSES))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, precision, width, label='Precision', color='#1f4e78')
    plt.bar(x, recall, width, label='Recall', color='#2e75b6')
    plt.bar(x + width, f1, width, label='F1-Score', color='#5b9bd5')
    plt.xticks(x, CLASSES, fontsize=10)
    plt.ylabel("Score")
    plt.title("Real Performance Metrics by Severity Category (Clean Test Split)")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "performance_metrics.png"), dpi=300)
    plt.close()

    # 3. رسم منحنيات ROC الخام المباشرة
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(CLASSES):
        y_true_binary = (y_ts_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, probs_ts[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves (Clean Evaluation)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.jpg"), dpi=300)
    plt.close()

    print("[+] تم حساب وحفظ جميع المخططات المباشرة والدقيقة بنجاح!")

if __name__ == "__main__":
    evaluate_and_plot()