import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import MultiTaskScoliosisResNet
from predict_scoliosis_clinical import RefinedGradCAMPlusPlus

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_raw\metadata.csv")
MODEL_PATH = "best_scoliosis_model_balanced.pth"
OUTPUT_DIR = os.path.join(BASE_DIR, r"results\gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_severity_from_row(row):
    path_str = (str(row.get('image_path', '')) + " " + str(row.get('label_or_mask_path', '')) + " " + str(
        row.get('file_name', ''))).lower()
    if "healthy" in path_str or "normal" in path_str:
        return 0
    elif "severe" in path_str or "3-degree" in path_str or "3_degree" in path_str:
        return 3
    elif "moderate" in path_str or "2-degree" in path_str or "2_degree" in path_str:
        return 2
    elif "mild" in path_str or "1-degree" in path_str or "1_degree" in path_str:
        return 1
    return 0


def apply_spine_roi_mask(heatmap, img_shape):
    """تعديل أبعاد الخريطة الحرارية وتطبيق القناع الموجه للعمود الفقري."""
    h, w = img_shape

    # 1. تغيير حجم الخريطة الحرارية لتتطابق تماماً مع أبعاد الصورة الأصلية (w, h)
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    # 2. إنشاء القناع الموجه لحجم الصورة الأصلي
    mask = np.ones((h, w), dtype=np.float32)

    # تضعيف الحواف الجانبية (خارج نطاق العمود الفقري)
    x_margin = int(w * 0.15)
    mask[:, :x_margin] *= 0.2
    mask[:, w - x_margin:] *= 0.2

    # تضعيف المنطقة السفلية (منطقة الحوض)
    y_pelvis = int(h * 0.82)
    mask[y_pelvis:, :] *= 0.15

    # تنعيم حواف القناع
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    # دمج القناع مع الخريطة الحرارية
    refined_heatmap = heatmap * mask

    # 3. قص الضوضاء الضعيفة لتركيز اللون الأحمر على قمة الانحناء (Curve Apex)
    threshold = np.percentile(refined_heatmap, 60)
    refined_heatmap[refined_heatmap < threshold] = 0

    max_val = np.max(refined_heatmap)
    if max_val > 0:
        refined_heatmap = refined_heatmap / max_val

    return refined_heatmap


def generate_refined_gradcam_grid():
    print("=" * 65)
    print("[+] توليد شبكة Grad-CAM++ الموجهة حيوياً لقمة انحراف الفقرات")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV) or not os.path.exists(MODEL_PATH):
        return

    df = pd.read_csv(METADATA_CSV)
    df['computed_severity'] = df.apply(extract_severity_from_row, axis=1)

    model = MultiTaskScoliosisResNet(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    target_layer = model.backbone.layer4[1].conv2
    grad_cam = RefinedGradCAMPlusPlus(model, target_layer)

    sample_rows = []
    for label_id in [0, 1, 2, 3]:
        sub_df = df[df['computed_severity'] == label_id]
        if len(sub_df) > 0:
            sample_rows.append(sub_df.iloc[0])
        else:
            sample_rows.append(df.iloc[label_id % len(df)])

    fig, axes = plt.subplots(len(sample_rows), 3, figsize=(10, 3 * len(sample_rows)))
    labels_text = ["Healthy", "1-degree (Mild)", "2-degree (Moderate)", "3-degree (Severe)"]

    for idx, row in enumerate(sample_rows):
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

        # 1. استخراج الخريطة الحرارية الخام
        raw_heatmap = grad_cam.generate_heatmap(tensor_img)

        # 2. مطابقة الأبعاد وتطبيق قناع الفقرات وقمع ضوضاء الحوض
        spine_heatmap = apply_spine_roi_mask(raw_heatmap, (h, w))

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * spine_heatmap), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(raw_bgr, 0.65, heatmap_colored, 0.35, 0)

        label_name = labels_text[idx]

        axes[idx, 0].imshow(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Original X-Ray ({label_name})", fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title("Spine-Focused Heatmap", fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[idx, 2].set_title("Focused Region Overlay", fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    summary_path = os.path.join(os.path.dirname(OUTPUT_DIR), "gradcam_summary_grid.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"[+] تم حفظ الخريطة الحرارية المحدثة بدقة عالية في: {summary_path}")


if __name__ == "__main__":
    generate_refined_gradcam_grid()