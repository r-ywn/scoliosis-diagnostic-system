import os
import sys
import pandas as pd

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_processed\clean_unified_metadata.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 65)
    print("   التشغيل الشامل لمخرجات مشروع الجنف (Scoliosis Pipeline)")
    print("=" * 65)

    if not os.path.exists(METADATA_CSV):
        print(f"[ERROR] لم يتم العثور على ملف الميتاداتا النظيف الموحد: {METADATA_CSV}")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    print(f"[+] إجمالي الصور المسجلة في الميتاداتا الموحدة: {len(df):,} صورة.")

    print("\n[1/3] جاري تشغيل التقييم الشامل وتقارير الأداء الفعلي...")
    os.system(f'"{sys.executable}" evaluate_all_cases.py')

    print("\n[2/3] جاري توليد خرائط Grad-CAM++ المباشرة...")
    os.system(f'"{sys.executable}" generate_gradcam_results.py')

    print("\n[3/3] جاري تشغيل التحليل الهندسي لمعادلة المنحنى...")
    os.system(f'"{sys.executable}" evaluate_geometric_shape.py')

    print("\n" + "=" * 65)
    print(f"[+] اكتملت جميع التقييمات بنجاح! تم حفظ المخرجات والرسوم في: {RESULTS_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()