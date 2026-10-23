import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

# 1. إعداد المسارات والجهاز
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
METADATA_CSV = r"D:\Scoliosis_Project\data_raw\metadata.csv"
OUTPUT_DIR = r"D:\Scoliosis_Project\results\gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RealGradCAMPlusPlus:
    """استخراج الخرائط الحرارية الحقيقية برمجياً عبر التدرجات الحسابية للطبقة الالتفافية الأخيرة."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # تسجيل الهوك (Hooks) لالتقاط الإنقاطات والتدرجات
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)

        # الاعتماد على مخرج الشدة (Intensity Head)
        logits = outputs["intensity"]
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        # خوارزمية Grad-CAM++ الحسابية
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

        cam = cv2.resize(cam, (224, 224))
        return cam, target_class


def run_gradcam_extraction():
    print("=" * 65)
    print(f"[+] بدء توليد خرائط Grad-CAM++ الحقيقية لعينات الأشعة")
    print("=" * 65)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_CSV):
        print("[ERROR] تعذر العثور على أوزان النموذج أو ملف الفهرس.")
        return

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # تحديد الطبقة الالتفافية الأخيرة في ResNet-18
    target_layer = model.backbone.layer4[1].conv2
    grad_cam = RealGradCAMPlusPlus(model, target_layer)

    df = pd.read_csv(METADATA_CSV)

    # اختيار عينة حقيقية واحدة لكل فئة مرضيّة من الصور الموجودة
    categories = ['healthy', '1-degree', '2-degree', '3-degree']
    sample_rows = []

    for cat in categories:
        sub_df = df[df['label_or_mask_path'].str.contains(cat, case=False, na=False)]
        if len(sub_df) > 0:
            sample_rows.append((cat, sub_df.iloc[0]))

    if len(sample_rows) == 0:
        # اختيار عشوائي بحال تعذر المطابقة اللفظية
        sample_rows = [('sample', df.iloc[i]) for i in range(min(4, len(df)))]

    grid_images = []

    for cat_name, row in sample_rows:
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            continue

        h, w, _ = raw_bgr.shape
        img_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        tensor_img = val_transforms(img_pil).unsqueeze(0).to(DEVICE)

        tensor_img.requires_grad_()
        heatmap, pred_class = grad_cam.generate_heatmap(tensor_img)

        # تحويل الخريطة الحرارية وتطبيق ألوان JET
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_colored, (w, h))

        # دمج الخريطة مع صورة الأشعة الأصلية بنسبة 60% إلى 40%
        overlay = cv2.addWeighted(raw_bgr, 0.6, heatmap_resized, 0.4, 0)

        # حفظ الصورة الفردية
        save_path = os.path.join(OUTPUT_DIR, f"gradcam_{cat_name}.png")
        cv2.imwrite(save_path, overlay)

        grid_images.append({
            "category": cat_name,
            "original": cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (224, 224)),
            "heatmap": cv2.resize(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), (224, 224)),
            "overlay": cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), (224, 224))
        })

    # رسم وإخراج لوحة التجميع الشاملة لخرائط التركيز الحقيقية (Grid View)
    if len(grid_images) > 0:
        fig, axes = plt.subplots(len(grid_images), 3, figsize=(10, 3 * len(grid_images)))
        if len(grid_images) == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, item in enumerate(grid_images):
            axes[i, 0].imshow(item["original"])
            axes[i, 0].set_title(f"Original X-Ray ({item['category']})")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(item["heatmap"])
            axes[i, 1].set_title("Grad-CAM++ Heatmap")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(item["overlay"])
            axes[i, 2].set_title("Focused Region Overlay")
            axes[i, 2].axis('off')

        plt.tight_layout()
        grid_save_path = os.path.join(r"D:\Scoliosis_Project\results", "gradcam_summary_grid.png")
        plt.savefig(grid_save_path, dpi=300)
        plt.close()

        print(f"[+] تم تصدير خرائط التركيز الحرارية الحقيقية بنجاح إلى:")
        print(f"    - المجلد: {OUTPUT_DIR}")
        print(f"    - المخطط المجمع: {grid_save_path}")


if __name__ == "__main__":
    run_gradcam_extraction()