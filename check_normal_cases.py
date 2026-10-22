import os
import pandas as pd

PRED_CSV = r"D:\Scoliosis_Project\data_processed\external_predictions.csv"

if os.path.exists(PRED_CSV):
    df = pd.read_csv(PRED_CSV)

    # تصفية صور مستشفى الملك عبد الله الحالات السليمة (التي تحتوي أسماؤها على KA أو كلمة Normal)
    ka_normal_df = df[df['file_name'].str.contains('KA', case=False, na=False)]

    print(f" تم العثور على {len(ka_normal_df)} صورة تابعة لمستشفى الملك عبد الله.")
    print("\n عينة من زوايا كوب الرقمية المتوقعة لهذه الحالات السليمة:")
    print(ka_normal_df[['file_name', 'pred_max_cobb', 'pred_intensity']].head(15).to_string())

    print("\n متوسط زاوية كوب المتوقعة للحالات السليمة:", round(ka_normal_df['pred_max_cobb'].mean(), 2))

    # تطبيق المعيار الطبي العالمي المبني على الزاوية
    ka_normal_df['medical_intensity'] = ka_normal_df['pred_max_cobb'].apply(
        lambda x: 'Healthy' if x < 10 else ('Mild' if x < 25 else ('Moderate' if x < 45 else 'Severe'))
    )

    print("\n إعادة التصنيف وفق المعيار الطبي العالمي (Cobb < 10° = Healthy):")
    print(ka_normal_df['medical_intensity'].value_counts().to_string())
else:
    print(" الملف غير موجود.")