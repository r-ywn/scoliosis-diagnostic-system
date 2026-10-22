import os
import glob
import hashlib
import pandas as pd

# المسارات
NEW_DATASET_DIR = r"D:\Scoliosis_Project\data_raw\scoliosis-dataset"
RAW_DATA_DIR = r"D:\Scoliosis_Project\data_raw"
OUTPUT_CSV = r"D:\Scoliosis_Project\data_processed\new_dataset_metadata.csv"


def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_and_deduplicate():
    print(f" [1/3] جاري فحص كافة الصور السابقة للحصول على البصمات الرقمية (Hashes)...")
    existing_images = glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.jpg"), recursive=True) + \
                      glob.glob(os.path.join(RAW_DATA_DIR, "**", "*.png"), recursive=True)

    old_images = [p for p in existing_images if "scoliosis-dataset" not in p]

    seen_hashes = {}
    for p in old_images:
        seen_hashes[get_file_hash(p)] = p

    print(f" تم تسجيل بصمات {len(seen_hashes)} صورة قديمة بنجاح.")

    print(f"\n [2/3] جاري فحص وتنظيف الداتا الجديدة: {NEW_DATASET_DIR}")
    new_images = glob.glob(os.path.join(NEW_DATASET_DIR, "**", "*.jpg"), recursive=True) + \
                 glob.glob(os.path.join(NEW_DATASET_DIR, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(NEW_DATASET_DIR, "**", "*.jpeg"), recursive=True)

    print(f"إجمالي الصور المكتشفة في الداتا الجديدة: {len(new_images)}")

    internal_duplicates = 0
    cross_duplicates = 0
    clean_records = []

    degree_map = {
        "healthy": {"intensity": 0, "cobb_min": 0.0, "cobb_max": 9.9},
        "1-degree": {"intensity": 1, "cobb_min": 10.0, "cobb_max": 24.9},
        "2-degree": {"intensity": 2, "cobb_min": 25.0, "cobb_max": 44.9},
        "3-degree": {"intensity": 3, "cobb_min": 45.0, "cobb_max": 80.0}
    }

    for img_path in new_images:
        img_hash = get_file_hash(img_path)

        if img_hash in seen_hashes:
            if "scoliosis-dataset" in seen_hashes[img_hash]:
                internal_duplicates += 1
            else:
                cross_duplicates += 1
            continue

        seen_hashes[img_hash] = img_path

        path_parts = os.path.normpath(img_path).split(os.sep)
        split_type = "train"
        category = "healthy"

        for part in path_parts:
            part_lower = part.lower()
            if part_lower in ["train", "valid", "test"]:
                split_type = part_lower
            if part_lower in degree_map:
                category = part_lower

        cat_info = degree_map[category]

        clean_records.append({
            "filepath": img_path,
            "filename": os.path.basename(img_path),
            "split": split_type,
            "category": category,
            "intensity_label": cat_info["intensity"],
            "cobb_min": cat_info["cobb_min"],
            "cobb_max": cat_info["cobb_max"]
        })

    df = pd.DataFrame(clean_records)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n==================== تقرير معالجة البيانات الجديدة ====================")
    print(f" عدد الصور الفريدة الصالحة للاستخدام: {len(df)}")
    print(f"⚠️ الصور المكررة داخلياً: {internal_duplicates}")
    print(f" الصور المكررة مع البيانات القديمة (Cross Duplicates): {cross_duplicates}")
    print("\n توزيع الحالات حسب الفئة:")
    print(df['category'].value_counts().to_string())
    print(f"\n تم حفظ الـ Metadata المنظفة في: {OUTPUT_CSV}")


if __name__ == "__main__":
    process_and_deduplicate()