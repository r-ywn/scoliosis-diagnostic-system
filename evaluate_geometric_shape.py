import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, r"results\shape_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_image_path_map(data_raw_dir):
    image_map = {}
    for root, _, files in os.walk(data_raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                if f not in image_map:
                    image_map[f] = os.path.join(root, f)
    return image_map


def fit_spline_shape(landmarks_xy):
    pts = np.array(landmarks_xy)
    if len(pts) < 4:
        return "C-shape", None, 0

    sort_idx = np.argsort(pts[:, 1])
    x = pts[sort_idx, 0]
    y = pts[sort_idx, 1]

    y_u, idx = np.unique(y, return_index=True)
    x_u = x[idx]

    if len(y_u) < 4:
        return "C-shape", None, 0

    try:
        spline = UnivariateSpline(y_u, x_u, k=3, s=len(y_u) * 50)
        y_dense = np.linspace(np.min(y_u), np.max(y_u), 200)
        x_dense = spline(y_dense)

        d2 = spline.derivative(n=2)(y_dense)
        sign_changes = np.where(np.diff(np.sign(d2)))[0]
        num_inflections = len(sign_changes)

        classified_shape = "S-shape" if num_inflections >= 2 else "C-shape"
        return classified_shape, (x_dense, y_dense), num_inflections
    except Exception:
        return "C-shape", None, 0


def parse_yolo_landmarks(label_path, img_width, img_height):
    landmarks = []
    if not os.path.exists(label_path):
        return landmarks
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            x_c = float(parts[1]) * img_width
            y_c = float(parts[2]) * img_height
            landmarks.append([x_c, y_c])
    return landmarks


def find_label_path_for_image(img_path):
    base_no_ext = os.path.splitext(img_path)[0]
    possible_paths = [
        base_no_ext + ".txt",
        img_path.replace("images", "labels").rsplit('.', 1)[0] + ".txt",
        img_path.replace("image", "label").rsplit('.', 1)[0] + ".txt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None


def run_shape_analysis_evaluation():
    print("=" * 65)
    print("[+] بدء التحليل الهندسي المطور باستخدام B-Splines على الميتاداتا الموحدة")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] تعذر العثور على الميتاداتا الموحدة: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)
    path_map = build_image_path_map(DATA_RAW_DIR)
    results_summary = []
    sample_plots = 0

    for idx, row in df.iterrows():
        img_name = str(row.get('image_name', ''))
        img_path = path_map.get(img_name)

        if not img_path or not os.path.exists(img_path):
            continue

        label_path = find_label_path_for_image(img_path)
        if label_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        landmarks = parse_yolo_landmarks(label_path, w, h)
        if len(landmarks) < 4:
            continue

        shape_class, fit_data, inflections = fit_spline_shape(landmarks)

        results_summary.append({
            "image_name": img_name,
            "detected_shape": shape_class,
            "inflection_points": inflections
        })

        if sample_plots < 4 and fit_data is not None:
            sample_plots += 1
            x_dense, y_dense = fit_data
            pts = np.array(landmarks)

            plt.figure(figsize=(6, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.plot(x_dense, y_dense, 'r-', lw=3, label=f'Smooth Spline ({shape_class})')
            plt.scatter(pts[:, 0], pts[:, 1], c='yellow', s=45, zorder=5, label='Vertebrae Centroids')
            plt.title(f"Spine Spline Fit: {shape_class} (Inflections: {inflections})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shape_sample_{sample_plots}.png"), dpi=300)
            plt.close()

    summary_df = pd.DataFrame(results_summary)
    if len(summary_df) > 0:
        shape_counts = summary_df['detected_shape'].value_counts()
        print("\n==================== نتائج التحليل الهندسي المطور ====================")
        for shape_name, count in shape_counts.items():
            print(f"- {shape_name}: {count:,} حالة ({count/len(summary_df)*100:.1f}%)")
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "geometric_shape_report.csv"), index=False)
    else:
        print("[+] اكتمل الفحص: لا توجد صور مرتبطة بملفات ترقيم .txt في الهيكل الحالي.")


if __name__ == "__main__":
    run_shape_analysis_evaluation()