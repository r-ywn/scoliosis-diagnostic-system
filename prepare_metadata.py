import os
import glob
import pandas as pd

BASE_DIR = r"D:\Scoliosis_Project"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data_raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data_processed")
OUTPUT_CSV = os.path.join(DATA_PROCESSED_DIR, "clean_unified_metadata.csv")

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def parse_all_datasets():
    print("=" * 70)
    print(" بدء بناء الميتاداتا الموحدة والنظيفة للمجموعات المعتمدة...")
    print("=" * 70)

    records = []

    # 1. معالجة Spinal-AI2024 (المرجع الذهبي - 20,000 صورة كاملة)
    spinal_dir = os.path.join(DATA_RAW_DIR, "Spinal-AI2024")
    if os.path.exists(spinal_dir):
        print("  Parsing Spinal-AI2024 (Gold Standard)...")
        train_csv = os.path.join(DATA_PROCESSED_DIR, "metadata_train.csv")
        test_csv = os.path.join(DATA_PROCESSED_DIR, "metadata_test.csv")

        for csv_p, split_name in [(train_csv, "train"), (test_csv, "test")]:
            if os.path.exists(csv_p):
                df_sp = pd.read_csv(csv_p)
                for _, row in df_sp.iterrows():
                    records.append({
                        "image_name": row["image_name"],
                        "dataset_source": "Spinal-AI2024",
                        "split": split_name,
                        "intensity": row["intensity"],
                        "stat": row["stat"],
                        "shape": row["shape"],
                        "location": row.get("location", "None"),
                        "max_cobb_angle": row["max_cobb_angle"],
                        "has_cobb_gt": True
                    })

    # 2. معالجة scoliosis-dataset (المجلدات الصريحة)
    scol_dir = os.path.join(DATA_RAW_DIR, "scoliosis-dataset")
    if os.path.exists(scol_dir):
        print("  Parsing scoliosis-dataset...")
        for root, _, files in os.walk(scol_dir):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    path_lower = full_path.lower().replace('\\', '/')

                    if 'healthy' in path_lower:
                        intensity, stat = 'Healthy', '0-derece'
                    elif '1-degree' in path_lower:
                        intensity, stat = 'Mild', '1-derece'
                    elif '2-degree' in path_lower:
                        intensity, stat = 'Moderate', '2-derece'
                    elif '3-degree' in path_lower:
                        intensity, stat = 'Severe', '3-derece'
                    else:
                        continue

                    records.append({
                        "image_name": file,
                        "dataset_source": "scoliosis-dataset",
                        "split": "external",
                        "intensity": intensity,
                        "stat": stat,
                        "shape": "Normal" if intensity == "Healthy" else "C",
                        "location": "None" if intensity == "Healthy" else "Thoracic",
                        "max_cobb_angle": 5.0 if intensity == "Healthy" else (
                            18.0 if intensity == "Mild" else (32.0 if intensity == "Moderate" else 55.0)),
                        "has_cobb_gt": False
                    })

    # 3. معالجة xkt857dsxk-1 mendeley (الأحجام الأصلية فقط لمنع التكرار)
    mendeley_dir = os.path.join(DATA_RAW_DIR, "xkt857dsxk-1 mendeley", "xkt857dsxk-1", "ImagesOriginalSize")
    if os.path.exists(mendeley_dir):
        print("  Parsing xkt857dsxk-1 mendeley (Original Size only)...")
        for root, _, files in os.walk(mendeley_dir):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    path_lower = root.lower().replace('\\', '/')
                    if 'normalfinal' in path_lower or 'normal' in path_lower:
                        intensity, stat = 'Healthy', '0-derece'
                    elif 'scolfinal' in path_lower or 'scol' in path_lower:
                        intensity, stat = 'Mild', '1-derece'
                    else:
                        continue

                    records.append({
                        "image_name": file,
                        "dataset_source": "xkt857dsxk-1",
                        "split": "external",
                        "intensity": intensity,
                        "stat": stat,
                        "shape": "Normal" if intensity == "Healthy" else "C",
                        "location": "None" if intensity == "Healthy" else "Thoracic",
                        "max_cobb_angle": 5.0 if intensity == "Healthy" else 18.0,
                        "has_cobb_gt": False
                    })

    df_clean = pd.DataFrame(records)
    initial_count = len(df_clean)

    # -------------------------------------------------------------
    #  الفلترة وحذف التكرار الآمن (بناءً على اسم الصورة والمصدر)
    # -------------------------------------------------------------
    df_clean.drop_duplicates(subset=['image_name', 'dataset_source'], keep='first', inplace=True)

    dropped_count = initial_count - len(df_clean)

    # حفظ النتائج
    df_clean.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f" تم إنشاء الميتاداتا الموحدة والنظيفة: {OUTPUT_CSV}")
    print(f" عدد الصور المكررة المحذوفة: {dropped_count:,}")
    print(f" إجمالي الصور الفريدة والنظيفة: {len(df_clean):,}")
    print("\nتوزيع مصادر البيانات المعتمدة:")
    print(df_clean['dataset_source'].value_counts().to_string())
    print("\nتوزيع الفئات التشخيصية الموحدة:")
    print(df_clean['intensity'].value_counts().to_string())
    print("=" * 70)


if __name__ == "__main__":
    parse_all_datasets()