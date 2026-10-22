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


def hybrid_predict(pred_max_cobb, intensity_probs, t_healthy=8.8, t_mild=22.5, t_severe=43.0):
    """
    محرك القرار الهجين: يدمج احتمالية التصنيف المباشر مع زاوية كوب لتعديل الحدود الحرجة
    """
    cls_pred = int(np.argmax(intensity_probs))
    cls_conf = float(intensity_probs[cls_pred])

    # إذا كان رأس التصنيف متأكداً بنسبة عالية جداً (> 70%) نعتمد توقعه
    if cls_conf > 0.70:
        return cls_pred

    # وإلا نعتمد الحدود المعايرة لزاوية كوب
    if pred_max_cobb < t_healthy:
        return 0  # Healthy
    elif pred_max_cobb < t_mild:
        return 1  # 1-degree (Mild)
    elif pred_max_cobb < t_severe:
        return 2  # 2-degree (Moderate)
    else:
        return 3  # 3-degree (Severe)


def run_calibration():
    print(f" جاري تحميل النموذج المتوازن للمعايرة: {MODEL_PATH}")
    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(" تم تحميل النموذج بنجاح.")
    else:
        print(" لم يتم العثور على ملف النموذج المتوازن.")
        return

    model.eval()
    df = pd.read_csv(METADATA_CSV)
    test_df = df[df['split'] == 'test'].copy()
    if len(test_df) < 100:
        test_df = df.sample(n=min(500, len(df)), random_state=42)

    print(f" تقييم المعايرة الهجينة على {len(test_df)} صورة...")

    y_true = []
    y_pred_calibrated = []

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

                # 1. استخراج زاوية كوب القصوى
                pred_cobb = outputs["cobb_angles"].cpu().numpy()[0]
                pred_max_cobb = float(pred_cobb[3])

                # 2. استخراج الاحتمالات الرياضية لرأس التصنيف (Softmax)
                probs = torch.softmax(outputs["intensity"], dim=1).cpu().numpy()[0]

                # 3. القرار الهجين المعاير
                final_label = hybrid_predict(pred_max_cobb, probs)

                y_true.append(gt_label)
                y_pred_calibrated.append(final_label)

            except Exception:
                continue

    target_names = ["Healthy", "1-degree (Mild)", "2-degree (Moderate)", "3-degree (Severe)"]
    cm, report_text = calculate_custom_metrics(y_true, y_pred_calibrated, target_names)

    print("\n==================== التقرير الطبي المعدل بعد المعايرة الهجينة ====================")
    print("\n 1. دقة التشخيص لكل فئة مرضيّة:")
    print(report_text)

    print("\n 2. مصفوفة الارتباك الجديدة (Confusion Matrix):")
    cm_df = pd.DataFrame(cm, index=[f"حقيقي: {n}" for n in target_names], columns=[f"متوقع: {n}" for n in target_names])
    print(cm_df.to_string())


if __name__ == "__main__":
    run_calibration()