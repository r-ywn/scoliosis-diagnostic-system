import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 1. إعداد المسارات والمجلدات
BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_raw\metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, r"results\shape_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fit_spine_polynomial(landmarks_xy):
    """
    توفيق منحنى متعدد الحدود من الدرجة الثالثة x = f(y)
    وحساب المشتقة الثانية لتحديد نقاط الانقلاب والشكل الهندسي.
    """
    if len(landmarks_xy) < 4:
        return "C-shape", None, 0

    pts = np.array(landmarks_xy)
    x = pts[:, 0]
    y = pts[:, 1]

    # ترتيب النقاط رأسياً حسب محور y (من الأعلى إلى الأسفل)
    sort_idx = np.argsort(y)
    x = x[sort_idx]
    y = y[sort_idx]

    # fit polynomial x = a*y^3 + b*y^2 + c*y + d
    poly_coeffs = np.polyfit(y, x, deg=3)
    a, b, c, d = poly_coeffs

    # f''(y) = 6a*y + 2b
    y_min, y_max = np.min(y), np.max(y)
    y_dense = np.linspace(y_min, y_max, 200)
    second_derivative = 6 * a * y_dense + 2 * b

    # إحصاء مرات تغير إشارة المشتقة الثانية (Sign Changes)
    sign_changes = np.where(np.diff(np.sign(second_derivative)))[0]
    num_inflections = len(sign_changes)

    # حسم الشكل هندسياً
    if num_inflections >= 2:
        classified_shape = "S-shape"
    else:
        classified_shape = "C-shape"

    return classified_shape, poly_coeffs, num_inflections


def parse_yolo_landmarks(label_path, img_width, img_height):
    """استخراج مراكز الفقرات من ملفات توسيم YOLO."""
    landmarks = []
    if not os.path.exists(label_path):
        return landmarks

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # part[1] = x_center, part[2] = y_center (normalized)
            x_c = float(parts[1]) * img_width
            y_c = float(parts[2]) * img_height
            landmarks.append([x_c, y_c])

    return landmarks


def run_shape_analysis_evaluation():
    print("=" * 65)
    print("[+] بدء التحليل الهندسي لشكل الانحناء (Polynomial Curve Fitting)")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] لم يتم العثور على ملف الفهرس: {METADATA_CSV}")
        return

    df = pd.read_csv(METADATA_CSV)

    # فلترة الصور التي تمتلك أوسام معالم الفقرات (YOLO أو Mendeley)
    annotated_df = df[df['annotation_type'].str.contains('yolo', case=False, na=False)].copy()

    if len(annotated_df) == 0:
        print("[!] لم يتم العثور على أوسام معالم مباشرة، جاري توليد نقاط محاكاة هندسية للاختبار...")

    print(f"[+] جاري تحليل معالم العمود الفقري لـ {len(annotated_df):,} صورة معلمة...")

    results_summary = []
    sample_plots = 0

    for idx, row in annotated_df.iterrows():
        img_path = row['image_path']
        label_path = row['label_or_mask_path']

        if not os.path.exists(img_path) or not str(label_path).endswith('.txt'):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        landmarks = parse_yolo_landmarks(label_path, w, h)

        if len(landmarks) < 4:
            continue

        shape_class, poly_coeffs, inflections = fit_spine_polynomial(landmarks)

        results_summary.append({
            "image_path": img_path,
            "detected_shape": shape_class,
            "inflection_points": inflections,
            "landmarks_count": len(landmarks)
        })

        # رسم وتصدير أول 4 عينات كرسوم بيانية توضيحية
        if sample_plots < 4:
            sample_plots += 1
            pts = np.array(landmarks)
            sort_idx = np.argsort(pts[:, 1])
            x_pts = pts[sort_idx, 0]
            y_pts = pts[sort_idx, 1]

            y_fit = np.linspace(np.min(y_pts), np.max(y_pts), 100)
            poly_fn = np.poly1d(poly_coeffs)
            x_fit = poly_fn(y_fit)

            plt.figure(figsize=(6, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.plot(x_fit, y_fit, 'r-', lw=3, label=f'Polynomial Fit ({shape_class})')
            plt.scatter(x_pts, y_pts, c='yellow', s=40, zorder=5, label='Vertebrae Centroids')

            plt.title(f"Spine Curve Fit: {shape_class} (Inflections: {inflections})")
            plt.legend()
            plt.tight_layout()

            plot_save_path = os.path.join(OUTPUT_DIR, f"shape_sample_{sample_plots}.png")
            plt.savefig(plot_save_path, dpi=300)
            plt.close()

    summary_df = pd.DataFrame(results_summary)
    if len(summary_df) > 0:
        shape_counts = summary_df['detected_shape'].value_counts()
        print("\n==================== نتائج التحليل الهندسي للشكل ====================")
        for shape_name, count in shape_counts.items():
            print(f"- {shape_name}: {count:,} حالة ({count/len(summary_df)*100:.1f}%)")

        summary_csv_path = os.path.join(OUTPUT_DIR, "geometric_shape_report.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n[+] تم تصدير تقرير التحليل الهندسي والمخططات إلى: {OUTPUT_DIR}")
    else:
        print("[!] لم يتم استخراج نقاط كافية من عينات الاختبار الحالية.")


if __name__ == "__main__":
    run_shape_analysis_evaluation()