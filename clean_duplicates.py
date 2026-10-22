import os
import glob
import hashlib


def get_file_hash(filepath):
    """حساب الـ Hash الرقمي لمحتوى الملف للتأكد من التطابق التام"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # قراءة الملف على أجزاء لتفادي استهلاك الذاكرة
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_clean_duplicates(data_dir):
    print(f" جاري فحص كافة الصور داخل المسار: {data_dir}\n")

    # البحث عن كافة الصور بصيغ jpg و png
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)

    total_images = len(image_paths)
    print(f"📷 إجمالي الصور المكتشفة للفحص: {total_images}")

    seen_hashes = {}
    duplicates = []

    for idx, path in enumerate(image_paths, 1):
        file_hash = get_file_hash(path)

        if file_hash in seen_hashes:
            duplicates.append((path, seen_hashes[file_hash]))
        else:
            seen_hashes[file_hash] = path

        if idx % 5000 == 0 or idx == total_images:
            print(f"   تم فحص [{idx}/{total_images}] صورة...")

    print("\n==================== تقرير فحص التكرار ====================")
    print(f" عدد الصور الفريدة (Unique Images): {len(seen_hashes)}")
    print(f" عدد الصور المكررة (Duplicates): {len(duplicates)}")

    if duplicates:
        print("\n عينة من الصور المكررة المكتشفة:")
        for dup, orig in duplicates[:5]:
            print(f"  مكررة: {os.path.basename(dup)}")
            print(f"   ️ الأصلية: {os.path.basename(orig)}\n")
    else:
        print("\n  لم يتم العثور على أي صور مكررة، البيانات سليمة ")

    return duplicates


if __name__ == "__main__":
    RAW_DATA_DIR = r"D:\Scoliosis_Project\data_raw"
    scan_and_clean_duplicates(RAW_DATA_DIR)