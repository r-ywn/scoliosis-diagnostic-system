import os
import pandas as pd

# تحديد مسارات البيانات المباشرة على القرص D
RAW_DATA_DIR = r"D:\Scoliosis_Project\data_raw\Spinal-AI2024"
OUTPUT_DIR = r"D:\Scoliosis_Project\data_processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_cobb_file(txt_filename):
    txt_path = os.path.join(RAW_DATA_DIR, txt_filename)

    if not os.path.exists(txt_path):
        print(f" تنبيه: الملف غير موجود في المسار: {txt_path}")
        print(" تأكد من تغيير اسم المجلد داخل data_raw إلى Spinal-AI2024 تماماً.")
        return pd.DataFrame()

    data = []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    print(f" تم قراءة الملف {txt_filename} بنجاح، عدد الأسطر: {len(lines)}")

    for line in lines:
        # دعم الفصل سواء بالمسافات أو الفواصل
        clean_line = line.strip().replace(',', ' ')
        parts = clean_line.split()

        if len(parts) < 4:
            continue

        img_name = parts[0]
        try:
            cobb1 = float(parts[1])  # Upper Thoracic
            cobb2 = float(parts[2])  # Main Thoracic
            cobb3 = float(parts[3])  # Lumbar
        except ValueError:
            continue  # تجاهل أسطر العناوين إن وجدت

        max_cobb = max(cobb1, cobb2, cobb3)

        # 1. الشدة والدرجة
        if max_cobb < 10.0:
            intensity, stat = "Healthy", "0-derece"
        elif 10.0 <= max_cobb < 25.0:
            intensity, stat = "Mild", "1-derece"
        elif 25.0 <= max_cobb < 45.0:
            intensity, stat = "Moderate", "2-derece"
        else:
            intensity, stat = "Severe", "3-derece"

        # 2. شكل الانحناء (C أم S)
        significant_curves = sum([1 for angle in [cobb2, cobb3] if angle >= 10.0])
        if significant_curves >= 2:
            shape = "S"
        elif significant_curves == 1:
            shape = "C"
        else:
            shape = "Normal"

        # 3. الموقع
        if max_cobb < 10.0:
            location = "None"
        elif max_cobb == cobb2:
            location = "Thoracic"
        elif max_cobb == cobb3:
            location = "Lumbar"
        else:
            location = "Upper Thoracic"

        data.append({
            "image_name": img_name,
            "cobb_upper_thoracic": cobb1,
            "cobb_main_thoracic": cobb2,
            "cobb_lumbar": cobb3,
            "max_cobb_angle": max_cobb,
            "intensity": intensity,
            "stat": stat,
            "shape": shape,
            "location": location
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("--- معالجة بيانات التدريب ---")
    df_train = process_cobb_file("Cobb_spinal-AI2024-train_gt.txt")
    if not df_train.empty:
        df_train.to_csv(os.path.join(OUTPUT_DIR, "metadata_train.csv"), index=False)
        print(f" تم إنشاء metadata_train.csv بنجاح! عدد الصور: {len(df_train)}\n")

    print("--- معالجة بيانات الاختبار ---")
    df_test = process_cobb_file("Cobb_spinal-AI2024-test_gt.txt")
    if not df_test.empty:
        df_test.to_csv(os.path.join(OUTPUT_DIR, "metadata_test.csv"), index=False)
        print(f" تم إنشاء metadata_test.csv بنجاح! عدد الصور: {len(df_test)}\n")