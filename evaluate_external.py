import os
import glob
import torch
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

# 1. إعداد الجهاز وتحميل التحويلات
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model.pth"

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# قواميس تحويل الأرقام التنبؤية إلى نصوص طبية
INTENSITY_LABELS = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}
SHAPE_LABELS = {0: "Normal", 1: "C", 2: "S"}
LOCATION_LABELS = {0: "None", 1: "Upper Thoracic", 2: "Thoracic", 3: "Lumbar"}


def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_trained_model():
    print(f" جاري تحميل النموذج المدرب من: {MODEL_PATH}")
    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def run_external_evaluation(external_dir):
    model = load_trained_model()

    print(f"\n جاري تجميع الصور وتقييمها من المسار الخارجي: {external_dir}")

    image_paths = glob.glob(os.path.join(external_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(external_dir, "**", "*.png"), recursive=True)

    # استبعاد صور داتا التدريب الأساسية Spinal-AI2024 للتركيز على البيانات الخارجية فقط
    external_images = [p for p in image_paths if "Spinal-AI2024" not in p]

    print(f"📷 إجمالي الصور الخارجية المكتشفة للتقييم: {len(external_images)}")

    seen_hashes = set()
    results = []

    with torch.no_grad():
        for idx, img_path in enumerate(external_images, 1):
            # فحص التكرار وتجاوز الصورة إذا كانت مكررة
            img_hash = get_file_hash(img_path)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            try:
                raw_img = Image.open(img_path).convert("RGB")
                input_tensor = eval_transforms(raw_img).unsqueeze(0).to(DEVICE)

                # التنبؤ بواسطة النموذج
                outputs = model(input_tensor)

                # استخراج زوايا كوب المتوقعة
                cobb_preds = outputs["cobb_angles"].squeeze(0).cpu().numpy()
                upper_th = round(float(cobb_preds[0]), 2)
                main_th = round(float(cobb_preds[1]), 2)
                lumbar = round(float(cobb_preds[2]), 2)
                max_cobb = round(float(cobb_preds[3]), 2)

                # استخراج التصنيفات المتوقعة
                intensity_pred = INTENSITY_LABELS[int(torch.argmax(outputs["intensity"], dim=1).item())]
                shape_pred = SHAPE_LABELS[int(torch.argmax(outputs["shape"], dim=1).item())]
                location_pred = LOCATION_LABELS[int(torch.argmax(outputs["location"], dim=1).item())]

                results.append({
                    "file_name": os.path.basename(img_path),
                    "pred_max_cobb": max_cobb,
                    "pred_upper_thoracic": upper_th,
                    "pred_main_thoracic": main_th,
                    "pred_lumbar": lumbar,
                    "pred_intensity": intensity_pred,
                    "pred_shape": shape_pred,
                    "pred_location": location_pred
                })

            except Exception as e:
                continue

            if idx % 1000 == 0 or idx == len(external_images):
                print(f"  تم التنبؤ وتقييم [{idx}/{len(external_images)}] صورة...")

    # حفظ نتائج التنبؤات الخارجية في CSV
    output_csv = r"D:\Scoliosis_Project\data_processed\external_predictions.csv"
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_csv, index=False)

    print("\n==================== تقرير التقييم الخارجي ====================")
    print(f" تم التنبؤ بنجاح لـ {len(df_res)} صورة فريدة من البيانات الخارجية!")
    print(f" تم حفظ تقرير التنبؤات الشامل في: {output_csv}")
    print("\n توزيع الشدة المتوقعة على البيانات الخارجية:")
    print(df_res['pred_intensity'].value_counts().to_string())
    print("\n متوسط زاوية كوب القصوى المتوقعة (Mean Max Cobb):", round(df_res['pred_max_cobb'].mean(), 2))


if __name__ == "__main__":
    RAW_DATA_DIR = r"D:\Scoliosis_Project\data_raw"
    run_external_evaluation(RAW_DATA_DIR)