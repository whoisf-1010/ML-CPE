import os
import cv2
import numpy as np

from preprocessing import preprocess_image

# กำหนดนามสกุลไฟล์ภาพที่อนุญาตให้อ่านได้
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

def load_data(data_path, img_size=100, max_per_class=None):
    images = []
    labels = []

    # 1. ค้นหาคลาสอัตโนมัติจากชื่อโฟลเดอร์ย่อยใน data_path
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # 2. วนลูปอ่านภาพจากแต่ละโฟลเดอร์คลาส
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        # ดึงเฉพาะไฟล์ที่มีนามสกุลตรงกับที่กำหนดไว้
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            # ถ้ามีการจำกัดจำนวนภาพต่อคลาสและโหลดครบแล้ว ให้หยุด (break)
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # 3. ย่อขนาดภาพทันทีที่อ่านเข้ามา เพื่อประหยัดหน่วยความจำ (RAM)
            image = preprocess_image(image, img_size)

            # 4. ข้ามภาพที่เสียหายหรือไม่สามารถอ่านได้
            if image is None:
                skipped += 1
                continue

            # เก็บข้อมูลภาพและป้ายกำกับ (Label) ลงในลิสต์
            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # แปลงลิสต์ให้เป็น Numpy Array เพื่อส่งให้โมเดลใช้งานต่อไป
    return np.stack(images), np.array(labels), classes