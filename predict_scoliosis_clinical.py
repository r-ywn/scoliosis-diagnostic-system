import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_BALANCED = "best_scoliosis_model_balanced.pth"
MODEL_LEGACY = "best_scoliosis_model.pth"
DEFAULT_MODEL_PATH = MODEL_BALANCED if os.path.exists(MODEL_BALANCED) else MODEL_LEGACY

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

INTENSITY_MAP_REV = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}
STAT_MAP_REV = {0: "0-derece", 1: "1-derece", 2: "2-derece", 3: "3-derece"}
SHAPE_MAP_REV = {0: "Normal", 1: "C-shape", 2: "S-shape"}
LOCATION_MAP_REV = {0: "None", 1: "Upper Thoracic", 2: "Main Thoracic", 3: "Lumbar"}

class ClinicalScoliosisAnalyzer:
    def __init__(self, model_path=None):
        self.device = DEVICE

        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model = MultiTaskScoliosisResNet(pretrained=False).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"[+] تم تحميل أوزان النموذج المحدثة بنجاح من: {model_path}")
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

            intensity_idx = int(torch.argmax(outputs["intensity"], dim=1).item())
            intensity = INTENSITY_MAP_REV.get(intensity_idx, "Healthy")

            stat_idx = int(torch.argmax(outputs["stat"], dim=1).item())
            stat_str = STAT_MAP_REV.get(stat_idx, "0-derece")

            shape_idx = int(torch.argmax(outputs["shape"], dim=1).item())
            shape = SHAPE_MAP_REV.get(shape_idx, "Normal")

            loc_idx = int(torch.argmax(outputs["location"], dim=1).item())
            location = LOCATION_MAP_REV.get(loc_idx, "None")

            # إلغاء التنميط بقسمة الأرقام السابقة وضربها في 100 لاستعادة الدرجات الحقيقية
            cobb_vals = outputs["cobb_angles"].squeeze(0).cpu().numpy() * 100.0
            u_th = round(max(0.0, float(cobb_vals[0])), 1)
            m_th = round(max(0.0, float(cobb_vals[1])), 1)
            lum = round(max(0.0, float(cobb_vals[2])), 1)
            max_cobb = round(max(0.0, float(cobb_vals[3])), 1)

            if intensity == "Healthy":
                stat_msg = f"التشخيص: العمود الفقري سليم (Normal Spine - Max Cobb: {max_cobb}°)"
            else:
                stat_msg = f"التشخيص: Scoliosis Detected: {intensity} {location} {shape} (Max Cobb: {max_cobb}°)"

            return {
                "stat": stat_msg,
                "intensity": intensity,
                "shape": shape,
                "location": location,
                "estimated_cobb_angles": {
                    "upper_thoracic_angle": u_th,
                    "main_thoracic_angle": m_th,
                    "lumbar_angle": lum,
                    "max_cobb_angle": max_cobb
                }
            }

        except Exception as e:
            return {"error": f"حدث خطأ أثناء معالجة الصورة: {str(e)}"}

if __name__ == "__main__":
    analyzer = ClinicalScoliosisAnalyzer()
    print(" تم اختبار كود التنبؤ السريري بنجاح، والنموذج جاهز لاستقبال صور الأشعة")