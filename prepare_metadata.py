import os
import glob
import pandas as pd

DATA_RAW_DIR = r"D:\Scoliosis_Project\data_raw"
OUTPUT_CSV = os.path.join(DATA_RAW_DIR, "metadata.csv")

records = []

# البحث عن كافة الصور
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
all_image_paths = []
for ext in image_extensions:
    all_image_paths.extend(glob.glob(os.path.join(DATA_RAW_DIR, '**', ext), recursive=True))

print(f"تم العثور على {len(all_image_paths)} صورة في D:\\Scoliosis_Project\\data_raw")

for img_path in all_image_paths:
    rel_path = os.path.relpath(img_path, DATA_RAW_DIR)
    parts = rel_path.split(os.sep)
    dataset_name = parts[0]
    file_name = os.path.basename(img_path)
    base_name, ext = os.path.splitext(file_name)

    # استخراج التقسيم (train / valid / test)
    split = "train"
    for p in parts:
        if p.lower() in ['train', 'valid', 'test']:
            split = p.lower()
            break

    annotation_type = "unknown"
    label_info = None

    # 1. Balanced Scoliosis (YOLOv5)
    if "Balanced Scoliosis" in dataset_name:
        annotation_type = "yolo_bbox"
        txt_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(ext, ".txt")
        label_info = txt_path if os.path.exists(txt_path) else None

    # 2. Neurroradiology Dataset 2 (YOLO Polygon / OBB)
    elif "Neurroradiology" in dataset_name:
        annotation_type = "yolo_polygon_obb"
        txt_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(ext, ".txt")
        label_info = txt_path if os.path.exists(txt_path) else None

    # 3. Scoliosis YOLOv5 Annotated Spine (Class 0: Vertebrae, Class 1: Full Region)
    elif "Scoliosis YOLOv5 Annotated" in dataset_name:
        annotation_type = "yolo_spine_vertebrae"
        txt_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").replace(ext, ".txt")
        label_info = txt_path if os.path.exists(txt_path) else None

    # 4. scoliosis-dataset (Folder Classification: healthy, 1-degree, 2-degree, 3-degree)
    elif dataset_name == "scoliosis-dataset":
        annotation_type = "folder_degree_classification"
        for degree in ['healthy', '1-degree', '2-degree', '3-degree']:
            if degree in parts:
                label_info = degree
                break

    # 5. Spinal-AI2024 (COCO JSON)
    elif dataset_name == "Spinal-AI2024":
        annotation_type = "coco_json"
        label_info = os.path.join(DATA_RAW_DIR, "Spinal_AI2024_test_annotation.json")

    # 6. Vertebrae for Scoliosis mendeley (Segmentation Masks)
    elif dataset_name == "Vertebrae for Scoliosis mendeley":
        annotation_type = "segmentation_mask"
        if "raw" in parts:
            mask_path = img_path.replace(f"{os.sep}raw{os.sep}", f"{os.sep}masks{os.sep}").replace(ext, ".png")
            label_info = mask_path if os.path.exists(mask_path) else None

    # 7. xkt857dsxk-1 mendeley (Normal vs Scol)
    elif dataset_name == "xkt857dsxk-1 mendeley":
        annotation_type = "folder_binary_classification"
        if "Normal" in parts:
            label_info = "Normal"
        elif "Scol" in parts:
            label_info = "Scoliosis"

    records.append({
        "image_path": img_path,
        "relative_path": rel_path,
        "file_name": file_name,
        "dataset_source": dataset_name,
        "annotation_type": annotation_type,
        "label_or_mask_path": label_info,
        "split": split
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"تم توليد ملف الفهرس الموحد بنجاح: {OUTPUT_CSV}")
print("\nتوزيع الصور على المجموعات الـ 7:")
print(df['dataset_source'].value_counts())