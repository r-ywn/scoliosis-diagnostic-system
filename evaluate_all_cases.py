import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
METADATA_CSV = r"D:\Scoliosis_Project\data_processed\new_dataset_metadata.csv"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def calculate_custom_metrics(y_true, y_pred, target_names):
    n_classes = len(target_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    report_lines = []
    header = f"{'الفئة (Category)':<24} | {'الدقة (Precision)':<18} | {'الاستدعاء (Recall)':<18} | {'F1-Score':<10} | {'العدد (Support)':<10}"
    report_lines.append(header)
    report_lines.append("-" * 90)

    for i, name in enumerate(target_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        line = f"{name:<24} | {precision:<18.2f} | {recall:<18.2f} | {f1:<10.2f} | {support:<10}"
        report_lines.append(line)

    return cm, "\n".join(report_lines)


def evaluate_all_categories():
    print(f" جاري تحميل النموذج المتوازن للتقييم الشامل لكل الحالات: {MODEL_PATH}")
    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(" تم تحميل النموذج المتوازن بنجاح.")
    else:
        print(" لم يتم العثور على ملف النموذج المتوازن.")
        return

    model.eval()

    if not os.path.exists(METADATA_CSV):
        print(" لم يتم العثور على ملف new_dataset_metadata.csv")
        return

    df = pd.read_csv(METADATA_CSV)

    test_df = df[df['split'] == 'test'].copy()
    if len(test_df) < 100:
        test_df = df.sample(n=min(500, len(df)), random_state=42)

    print(f" جاري تقييم {len(test_df)} صورة تُمثّل جميع الفئات المرضية والسليمة...")

    y_true = []
    y_pred = []
    cobb_errors = []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
            img_path = row['filepath']
            gt_label = int(row['intensity_label'])

            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                tensor_img = val_transforms(img).unsqueeze(0).to(DEVICE)
                outputs = model(tensor_img)

                pred_cobb = outputs["cobb_angles"].cpu().numpy()[0]
                pred_max_cobb = float(pred_cobb[3])

                # المعايرة الطبية لزاوية كوب
                if pred_max_cobb < 10.0:
                    calibrated_label = 0  # Healthy
                elif pred_max_cobb < 25.0:
                    calibrated_label = 1  # 1-degree (Mild)
                elif pred_max_cobb < 45.0:
                    calibrated_label = 2  # 2-degree (Moderate)
                else:
                    calibrated_label = 3  # 3-degree (Severe)

                y_true.append(gt_label)
                y_pred.append(calibrated_label)

                t_min, t_max = row['cobb_min'], row['cobb_max']
                if pred_max_cobb < t_min:
                    err = t_min - pred_max_cobb
                elif pred_max_cobb > t_max:
                    err = pred_max_cobb - t_max
                else:
                    err = 0.0
                cobb_errors.append(err)

            except Exception:
                continue

    target_names = ["Healthy", "1-degree (Mild)", "2-degree (Moderate)", "3-degree (Severe)"]
    cm, report_text = calculate_custom_metrics(y_true, y_pred, target_names)

    print("\n==================== التقرير الطبي الشامل لجميع الحالات ====================")
    print("\n 1. دقة التشخيص لكل فئة مرضيّة:")
    print(report_text)

    print("\n 2. مصفوفة الارتباك (Confusion Matrix - التداخل بين الحالات):")
    cm_df = pd.DataFrame(cm, index=[f"حقيقي: {n}" for n in target_names], columns=[f"متوقع: {n}" for n in target_names])
    print(cm_df.to_string())

    print(f"\n 3. متوسط انحراف زاوية كوب عن النطاق الطبي الصحيح: {np.mean(cobb_errors):.2f}° درجات")


if __name__ == "__main__":
    evaluate_all_categories()