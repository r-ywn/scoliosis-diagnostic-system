import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

# 1. إعداد الجهاز والنموذج
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model_balanced.pth"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ClinicalScoliosisAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        self.device = DEVICE
        self.model = MultiTaskScoliosisResNet(pretrained=False).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f" تم تحميل النموذج المتوازن بنجاح من: {model_path}")
        else:
            raise FileNotFoundError(f" لم يتم العثور على أوزان النموذج بمسار: {model_path}")

    def analyze_xray(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"الملف غير موجود: {image_path}"}

        try:
            img = Image.open(image_path).convert("RGB")
            tensor_img = val_transforms(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)

            # 1. استخراج زوايا كوب ومعايرتها
            raw_cobb = outputs["cobb_angles"].cpu().numpy()[0]

            # ضمان قيم موجبة ومنطقية
            ut_angle = max(0.0, float(raw_cobb[0]))  # Upper Thoracic
            mt_angle = max(0.0, float(raw_cobb[1]))  # Main Thoracic
            lum_angle = max(0.0, float(raw_cobb[2]))  # Lumbar
            max_cobb = max(ut_angle, mt_angle, lum_angle, float(raw_cobb[3]))

            # 2. التوافق المنطقي السريري للمخرجات (Clinical Harmonization Engine)
            if max_cobb < 10.0:
                intensity = "Healthy"
                shape = "Normal"
                location = "None"
                stat = f"Healthy Spine (Max Cobb: {max_cobb:.1f}° - Below clinical scoliosis threshold)"
            else:
                # تحديد الشدة وفق المعيار الطبي الدولي SRS
                if max_cobb < 25.0:
                    intensity = "Mild"
                elif max_cobb < 45.0:
                    intensity = "Moderate"
                else:
                    intensity = "Severe"

                # تحديد موقع الانحناء الرئيسي (Location)
                angles_dict = {
                    "Upper Thoracic": ut_angle,
                    "Main Thoracic": mt_angle,
                    "Lumbar": lum_angle
                }
                location = max(angles_dict, key=angles_dict.get)

                # تحديد شكل الانحناء (Shape)
                active_curves = sum(1 for a in [ut_angle, mt_angle, lum_angle] if a >= 10.0)
                if active_curves >= 2:
                    shape = "S-shape"
                else:
                    shape = "C-shape"

                stat = f"Scoliosis Detected: {intensity} {location} {shape} Curve (Max Cobb: {max_cobb:.1f}°)"

            # 3. بناء الهيكل الخماسي النهائي المعتمد
            result = {
                "stat": stat,
                "shape": shape,
                "estimated_cobb_angles": {
                    "upper_thoracic_angle": round(ut_angle, 2),
                    "main_thoracic_angle": round(mt_angle, 2),
                    "lumbar_angle": round(lum_angle, 2),
                    "max_cobb_angle": round(max_cobb, 2)
                },
                "intensity": intensity,
                "location": location
            }
            return result

        except Exception as e:
            return {"error": f"حدث خطأ أثناء معالجة الصورة: {str(e)}"}


if __name__ == "__main__":
    analyzer = ClinicalScoliosisAnalyzer()

    # ️ ضع مسار الصورة المطلوبة بدقة هنا
    test_image = r"D:\Scoliosis_Project\data_raw\scoliosis-dataset\valid\3-degree\01-July-2019-91_jpg.rf.b8097406e61e9aa835df22764d4a1546.jpg"

    # التحقق المباشر بدون جلب تلقائي بديل
    if not os.path.exists(test_image):
        print(f" لم يتم العثور على الصورة! تأكد من كتابة المسار واسم الملف بشكل صحيح:\n{test_image}")
    else:
        print(f"\n جاري تشخيص الصورة المختارة: {test_image}")
        prediction = analyzer.analyze_xray(test_image)

        print("\n التقرير الطبي النهائي (JSON):")
        print(json.dumps(prediction, indent=2, ensure_ascii=False))