import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# تحديد مسار النموذج الذكي (البحث عن أوزان التدريب الجديد الموازن أولاً)
MODEL_BALANCED = "best_scoliosis_model_balanced.pth"
MODEL_LEGACY = "best_scoliosis_model.pth"
DEFAULT_MODEL_PATH = MODEL_BALANCED if os.path.exists(MODEL_BALANCED) else MODEL_LEGACY

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

INTENSITY_MAP = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}


class ClinicalScoliosisAnalyzer:
    def __init__(self, model_path=None):
        self.device = DEVICE

        # اختيار المسار الافتراضي الذكي إذا لم يُحدد مسار خاص
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model = MultiTaskScoliosisResNet(pretrained=False).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[+] تم تحميل أوزان النموذج بنجاح من: {model_path}")
        else:
            raise FileNotFoundError(f"تعذر العثور على أوزان النموذج: {model_path}")

    def analyze_xray(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"الملف غير موجود: {image_path}"}

        try:
            raw_bgr = cv2.imread(image_path)
            if raw_bgr is None:
                return {"error": "تعذر قراءة صورة الأشعة"}

            img_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
            tensor_img = val_transforms(img_pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_img)

            # 1. الاستجابة المباشرة لرأس التصنيف
            intensity_logits = outputs["intensity"]
            probs = F.softmax(intensity_logits, dim=1).squeeze(0).cpu().numpy()
            pred_class_idx = int(np.argmax(probs))
            intensity = INTENSITY_MAP.get(pred_class_idx, "Healthy")

            # 2. استخراج زوايا Cobb الحسابية
            raw_cobb = outputs["cobb_angles"].squeeze(0).cpu().numpy()
            raw_ut = max(0.0, float(raw_cobb[0]))
            raw_mt = max(0.0, float(raw_cobb[1]))
            raw_lum = max(0.0, float(raw_cobb[2]))
            raw_max = max(0.0, float(raw_cobb[3]))

            # 3. المطابقة الدقيقة لقواعد الميتاداتا الموحدة
            if intensity == "Healthy":
                max_cobb = round(min(raw_max if raw_max > 0 else 5.0, 9.5), 2)
                ut_angle = round(min(raw_ut, 4.5), 2)
                mt_angle = round(min(raw_mt, 6.8), 2)
                lum_angle = round(min(raw_lum, 4.0), 2)

                location = "None"
                shape = "Normal"
                stat = f"Healthy Spine (Max Cobb: {max_cobb}° - Normal)"

            else:
                if intensity == "Mild":
                    min_b, max_b = 10.0, 24.9
                elif intensity == "Moderate":
                    min_b, max_b = 25.0, 44.9
                else:  # Severe
                    min_b, max_b = 45.0, 95.0

                if raw_max < min_b or raw_max > max_b:
                    max_cobb = round(float(np.clip(raw_max, min_b, max_b)), 2)
                else:
                    max_cobb = round(raw_max, 2)

                raw_sum = max(0.1, raw_ut + raw_mt + raw_lum)
                ut_angle = round(max(1.0, (raw_ut / raw_sum) * max_cobb), 2)
                mt_angle = round(max(1.0, (raw_mt / raw_sum) * max_cobb), 2)
                lum_angle = round(max(1.0, (raw_lum / raw_sum) * max_cobb), 2)

                angles_dict = {"Upper Thoracic": ut_angle, "Main Thoracic": mt_angle, "Lumbar": lum_angle}
                location = max(angles_dict, key=angles_dict.get)

                sorted_angles = sorted([ut_angle, mt_angle, lum_angle], reverse=True)
                if sorted_angles[0] >= 14.0 and sorted_angles[1] >= 14.0:
                    shape = "S-shape"
                else:
                    shape = "C-shape"

                stat = f"Scoliosis Detected: {intensity} {location} {shape} (Max Cobb: {max_cobb}°)"

            return {
                "stat": stat,
                "intensity": intensity,
                "shape": shape,
                "location": location,
                "estimated_cobb_angles": {
                    "upper_thoracic_angle": ut_angle,
                    "main_thoracic_angle": mt_angle,
                    "lumbar_angle": lum_angle,
                    "max_cobb_angle": max_cobb
                }
            }

        except Exception as e:
            return {"error": f"حدث خطأ أثناء معالجة الصورة: {str(e)}"}


if __name__ == "__main__":
    analyzer = ClinicalScoliosisAnalyzer()
    print(" تم تفعيل التشخيص بالاعتماد على قواعد الميتاداتا الموحدة وحذف الضوضاء!")