import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.interpolate import UnivariateSpline
from model import MultiTaskScoliosisResNet

# 1. الإعدادات والمعايير
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model_balanced.pth"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RefinedGradCAMPlusPlus:
    """استخراج خريطة التركيز الحرارية الحقيقية مع فلترة التشتت المباشر لحواف الحوض."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        logits = outputs["intensity"]
        target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        grad_sq = gradients ** 2
        grad_cube = gradients ** 3
        sum_activations = np.sum(activations, axis=(1, 2), keepdims=True)

        aij = grad_sq / (2 * grad_sq + sum_activations * grad_cube + 1e-8)
        weights = np.sum(aij * np.maximum(gradients, 0), axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        # فلترة الضوضاء والتشتت السفلي لضمان التركيز على محور العمود الفقري
        cam[cam < 0.20] = 0.0
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cv2.resize(cam, (224, 224))


def analyze_spine_spline_geometry(landmarks_xy):
    """التحليل الهندسي المرن باستخدام UnivariateSpline لمنع خروج الخط عن الفقرات."""
    if len(landmarks_xy) < 4:
        return "C-shape", 0

    pts = np.array(landmarks_xy)
    sort_idx = np.argsort(pts[:, 1])
    x = pts[sort_idx, 0]
    y = pts[sort_idx, 1]

    y_unique, unique_indices = np.unique(y, return_index=True)
    x_unique = x[unique_indices]

    if len(y_unique) < 4:
        return "C-shape", 0

    try:
        spline = UnivariateSpline(y_unique, x_unique, k=3, s=len(y_unique) * 2)
        y_dense = np.linspace(np.min(y_unique), np.max(y_unique), 150)
        d2 = spline.derivative(n=2)(y_dense)

        sign_changes = np.where(np.diff(np.sign(d2)))[0]
        num_inflections = len(sign_changes)

        classified_shape = "S-shape" if num_inflections >= 2 else "C-shape"
        return classified_shape, num_inflections
    except Exception:
        return "C-shape", 0


class ClinicalScoliosisAnalyzer:
    def __init__(self, model_path=MODEL_PATH):
        self.device = DEVICE
        self.model = MultiTaskScoliosisResNet(pretrained=False).to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"تعذر العثور على أوزان النموذج: {model_path}")

        target_layer = self.model.backbone.layer4[1].conv2
        self.grad_cam = RefinedGradCAMPlusPlus(self.model, target_layer)

    def analyze_xray(self, image_path, sample_landmarks=None):
        if not os.path.exists(image_path):
            return {"error": f"الملف غير موجود: {image_path}"}

        try:
            raw_bgr = cv2.imread(image_path)
            h, w, _ = raw_bgr.shape

            img_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
            tensor_img = val_transforms(img_pil).unsqueeze(0).to(self.device)

            tensor_img.requires_grad_()
            outputs = self.model(tensor_img)
            heatmap = self.grad_cam.generate_heatmap(tensor_img)

            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap_colored, (w, h))
            overlay_img = cv2.addWeighted(raw_bgr, 0.65, heatmap_resized, 0.35, 0)

            overlay_path = image_path.replace(".", "_gradcam.")
            cv2.imwrite(overlay_path, overlay_img)

            raw_cobb = outputs["cobb_angles"].detach().cpu().numpy()[0]
            ut_angle = max(0.0, float(raw_cobb[0]))
            mt_angle = max(0.0, float(raw_cobb[1]))
            lum_angle = max(0.0, float(raw_cobb[2]))
            max_cobb = max(ut_angle, mt_angle, lum_angle, float(raw_cobb[3]))

            if max_cobb < 10.0:
                intensity = "Healthy"
                shape = "Normal"
                location = "None"
                stat = f"Healthy Spine (Max Cobb: {max_cobb:.1f}° - Normal)"
            else:
                if max_cobb < 25.0:
                    intensity = "Mild"
                elif max_cobb < 45.0:
                    intensity = "Moderate"
                else:
                    intensity = "Severe"

                angles_dict = {"Upper Thoracic": ut_angle, "Main Thoracic": mt_angle, "Lumbar": lum_angle}
                location = max(angles_dict, key=angles_dict.get)

                if sample_landmarks is not None and len(sample_landmarks) >= 4:
                    shape, _ = analyze_spine_spline_geometry(sample_landmarks)
                else:
                    active_curves = sum(1 for a in [ut_angle, mt_angle, lum_angle] if a >= 10.0)
                    shape = "S-shape" if active_curves >= 2 else "C-shape"

                stat = f"Scoliosis Detected: {intensity} {location} {shape} (Max Cobb: {max_cobb:.1f}°)"

            return {
                "stat": stat,
                "shape": shape,
                "estimated_cobb_angles": {
                    "upper_thoracic_angle": round(ut_angle, 2),
                    "main_thoracic_angle": round(mt_angle, 2),
                    "lumbar_angle": round(lum_angle, 2),
                    "max_cobb_angle": round(max_cobb, 2)
                },
                "intensity": intensity,
                "location": location,
                "gradcam_path": overlay_path
            }

        except Exception as e:
            return {"error": f"حدث خطأ أثناء معالجة الصورة: {str(e)}"}