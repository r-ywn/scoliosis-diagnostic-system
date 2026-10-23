import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from model import MultiTaskScoliosisResNet

# 1. المسارات والإعدادات الرئيسية
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
METADATA_CSV = r"D:\Scoliosis_Project\data_raw\metadata.csv"
OUTPUT_DIR = r"D:\Scoliosis_Project\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# التحويلات البصرية المعيارية للصورة
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TARGET_NAMES = ["Healthy", "1-degree (Mild)", "2-degree (Moderate)", "3-degree (Severe)"]


def extract_gt_label(row):
    """استخراج الفئة الحقيقية الموثوقة بناءً على التوسيم والمصدر."""
    ann_type = str(row['annotation_type'])
    label_info = str(row['label_or_mask_path'])

    if ann_type == "folder_degree_classification":
        mapping = {'healthy': 0, '1-degree': 1, '2-degree': 2, '3-degree': 3}
        return mapping.get(label_info, None)

    elif ann_type == "folder_binary_classification":
        return 0 if label_info == "Normal" else 2  # Moderate كتمثيل عام للجنف

    elif "yolo" in ann_type:
        if label_info and os.path.exists(label_info):
            # فحص إذا كان ملف YOLO نصياً وغير فارغ
            if os.path.getsize(label_info) > 0:
                return 1  # الجنف الخفيف/العام
        return 0

    return None


def calculate_custom_metrics(y_true, y_pred, target_names):
    """حساب مصفوفة الارتباك الحقيقية ومقاييس Precision, Recall, F1-Score."""
    n_classes = len(target_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    report_lines = []
    header = f"{'الفئة (Category)':<24} | {'الدقة (Precision)':<18} | {'الاستدعاء (Recall)':<18} | {'F1-Score':<10} | {'العدد (Support)':<10}"
    report_lines.append(header)
    report_lines.append("-" * 90)

    metrics_dict = {"precision": [], "recall": [], "f1": []}

    for i, name in enumerate(target_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics_dict["precision"].append(precision)
        metrics_dict["recall"].append(recall)
        metrics_dict["f1"].append(f1)

        line = f"{name:<24} | {precision:<18.2f} | {recall:<18.2f} | {f1:<10.2f} | {support:<10}"
        report_lines.append(line)

    return cm, "\n".join(report_lines), metrics_dict


def plot_and_save_charts(cm, metrics_dict, y_true_all, y_probs_all):
    """رسم وتصدير المخططات الرسمية الحقيقية إلى مجلد results."""
    # 1. رسم مصفوفة الارتباك (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title("Real Confusion Matrix (35,683 Dataset Evaluation)")
    plt.xlabel("Predicted Diagnosis")
    plt.ylabel("True Diagnosis")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. رسم مخطط مقاييس الأداء (Precision, Recall, F1-Score)
    x = np.arange(len(TARGET_NAMES))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, metrics_dict["precision"], width, label='Precision', color='#1E3A8A')
    plt.bar(x, metrics_dict["recall"], width, label='Recall', color='#2563EB')
    plt.bar(x + width, metrics_dict["f1"], width, label='F1-Score', color='#60A5FA')

    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.title('Real Performance Metrics by Severity Category')
    plt.xticks(x, TARGET_NAMES, rotation=15)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_metrics.png"), dpi=300)
    plt.close()

    # 3. رسم منحنيات ROC متعددة الفئات
    if len(y_probs_all) > 0:
        plt.figure(figsize=(8, 6))
        y_true_oh = pd.get_dummies(y_true_all).values
        for i in range(len(TARGET_NAMES)):
            if i < y_true_oh.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_probs_all[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{TARGET_NAMES[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Real Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300)
        plt.close()

    print(f"\n[+] تم تصدير كافة المخططات البيانية الحقيقية بنجاح إلى: {OUTPUT_DIR}")


def evaluate_all_categories():
    print("=" * 65)
    print(f"[+] بدء التقييم الحقيقي على جهاز: {DEVICE}")
    print("=" * 65)

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"[+] تم تحميل أوزان النموذج المتوازن: {MODEL_PATH}")
    else:
        print(f"[ERROR] لم يتم العثور على أوزان النموذج: {MODEL_PATH}")
        return

    model.eval()

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] لم يتم العثور على ملف الفهرس: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    print(f"[+] إجمالي الصور المسجلة بالفهرس: {len(df):,} صورة عبر المجموعات الـ 7")

    # اختيار عينات الاختبار الحقيقية الممثلة للمجموعات ذات الوسوم
    test_df = df[df['annotation_type'].isin(['folder_degree_classification', 'folder_binary_classification', 'yolo_bbox', 'yolo_spine_vertebrae'])].copy()

    if len(test_df) == 0:
        test_df = df.sample(n=min(2000, len(df)), random_state=42)

    print(f"[+] جاري تقييم {len(test_df):,} صورة حقيقية مصنفة...")

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_path = row['image_path']
            gt_label = extract_gt_label(row)

            if gt_label is None or not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                tensor_img = val_transforms(img).unsqueeze(0).to(DEVICE)
                outputs = model(tensor_img)

                # استخراج احتمالات الشدة
                intensity_logits = outputs["intensity"]
                probs = torch.softmax(intensity_logits, dim=1).cpu().numpy()[0]

                # استخراج زاوية كوب لمعايرة التشخيص
                pred_cobb = outputs["cobb_angles"].cpu().numpy()[0]
                pred_max_cobb = float(pred_cobb[3])

                if pred_max_cobb < 10.0:
                    calibrated_label = 0
                elif pred_max_cobb < 25.0:
                    calibrated_label = 1
                elif pred_max_cobb < 45.0:
                    calibrated_label = 2
                else:
                    calibrated_label = 3

                y_true.append(gt_label)
                y_pred.append(calibrated_label)
                y_probs.append(probs)

            except Exception:
                continue

    if len(y_true) == 0:
        print("[ERROR] لم يتم العثور على عينات صالحة للمعالجة.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    cm, report_text, metrics_dict = calculate_custom_metrics(y_true, y_pred, TARGET_NAMES)

    print("\n==================== التقرير الطبي الشامل للتقييم الفعلي ====================")
    print("\n1. دقة التشخيص الحقيقية لكل فئة:")
    print(report_text)

    print("\n2. مصفوفة الارتباك الفعليه (Real Confusion Matrix):")
    cm_df = pd.DataFrame(cm, index=[f"حقيقي: {n}" for n in TARGET_NAMES],
                         columns=[f"متوقع: {n}" for n in TARGET_NAMES])
    print(cm_df.to_string())

    # تصدير المخططات الرسمية
    plot_and_save_charts(cm, metrics_dict, y_true, y_probs)


if __name__ == "__main__":
    evaluate_all_categories()