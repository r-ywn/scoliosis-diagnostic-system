import os
import sys
import pandas as pd

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_raw\metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print("   التشغيل الشامل لمخرجات مشروع الجنف (Scoliosis Pipeline)")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] لم يتم العثور على ملف الفهرس الموحد: {METADATA_CSV}")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    print(f"[+] إجمالي الصور الحقيقي المسجل بالفهرس: {len(df):,} صورة.")

    # 1. تشغيل التقييم الفعلي ومصفوفات الارتباك (الخطوة 2)
    print("\n[1/3] جاري تشغيل التقييم الشامل وتقارير الأداء...")
    os.system(f"{sys.executable} evaluate_all_cases.py")

    # 2. تشغيل استخراج خرائط التركيز الحرارية Grad-CAM++ (الخطوة 3)
    print("\n[2/3] جاري توليد خرائط Grad-CAM++ الحقيقية...")
    os.system(f"{sys.executable} generate_gradcam_results.py")

    # 3. تشغيل التحليل الهندسي للأشكال C/S (الخطوة 4)
    print("\n[3/3] جاري تشغيل التحليل الهندسي لمعادلة المنحنى...")
    os.system(f"{sys.executable} evaluate_geometric_shape.py")

    print("\n" + "=" * 65)
    print(f"[+] اكتملت جميع الخطوات بنجاح! تم حفظ كافة النتائج في: {RESULTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()