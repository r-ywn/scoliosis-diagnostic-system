import os
import pandas as pd

BASE_DIR = r"D:\Scoliosis_Project"
METADATA_CSV = os.path.join(BASE_DIR, r"data_raw\metadata.csv")

df = pd.read_csv(METADATA_CSV)
spinal_df = df[df['dataset_source'] == 'Spinal-AI2024']

print("--- عينة من أسطر Spinal-AI2024 (First 5 Rows) ---")
for idx, row in spinal_df.head(5).iterrows():
    print(f"Image Path: {row.get('image_path')}")
    print(f"Label Path: {row.get('label_or_mask_path')}")
    print(f"Annotation: {row.get('annotation_type')}")
    print("-" * 65)