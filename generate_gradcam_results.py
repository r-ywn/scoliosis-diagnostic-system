import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
OUTPUT_DIR = os.path.join(BASE_DIR, r"results\gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class PureGradCAMPlusPlus:
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

    def generate_heatmap(self, input_image, target_class=None):
        self.model.eval()
        outputs = self.model(input_image)
        logits = outputs['intensity']

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.sum(weights[:, None, None] * activations, axis=0)

        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        return heatmap


def generate_refined_gradcam_grid():
    print("=" * 65)
    print("[+] توليد خرائط Grad-CAM++ المنقاة والمتركزة على العمود الفقري")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV) or not os.path.exists(MODEL_PATH):
        print(f"[ERROR] تعذر العثور على الميتاداتا أو أوزان النموذج!")
        return

    df = pd.read_csv(METADATA_CSV)
    path_map = build_image_path_map(DATA_RAW_DIR)

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    target_layer = model.backbone.layer4[1].conv2
    grad_cam = PureGradCAMPlusPlus(model, target_layer)

    sample_rows = []
    labels_text = ["Healthy", "Mild", "Moderate", "Severe"]

    for label_name in labels_text:
        sub_df = df[df['intensity'] == label_name]
        if len(sub_df) > 0:
            sample_rows.append((sub_df.iloc[0], label_name))

    fig, axes = plt.subplots(len(sample_rows), 3, figsize=(11, 3.2 * len(sample_rows)))

    for idx, (row, label_name) in enumerate(sample_rows):
        img_name = str(row['image_name'])
        img_path = path_map.get(img_name)
        if not img_path or not os.path.exists(img_path):
            continue

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            continue
        h, w, _ = raw_bgr.shape
        img_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))

        tensor_img = val_transforms(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor_img)
            pred_class = torch.argmax(outputs['intensity'], dim=1).item()

        raw_heatmap = grad_cam.generate_heatmap(tensor_img, target_class=pred_class)

        # تصفية رياضية لعزل القيم الضعيفة على الحواف وبقاء التنبيه على الفقرات حصراً
        raw_heatmap[raw_heatmap < 0.25] = 0.0
        max_v = np.max(raw_heatmap)
        if max_v > 0:
            raw_heatmap = raw_heatmap / max_v

        heatmap_resized = cv2.resize(raw_heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # إزالة الألوان من المناطق الصفريّة الخالية من التنشيط
        heatmap_colored[heatmap_resized == 0] = 0

        overlay = cv2.addWeighted(raw_bgr, 0.65, heatmap_colored, 0.35, 0)

        axes[idx, 0].imshow(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Original X-Ray ({label_name})", fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title("Spine Grad-CAM++ Heatmap", fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title("Neural Focus Overlay", fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    summary_path = os.path.join(os.path.dirname(OUTPUT_DIR), "gradcam_summary_grid.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[+] تم إعادة توليد وتنقية الخرائط الحرارية بنجاح في: {summary_path}")


if __name__ == "__main__":
    generate_refined_gradcam_grid()