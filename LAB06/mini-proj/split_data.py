import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.1):
    """
    แบ่งข้อมูลออกเป็น 3 ส่วน: Train (ฝึกสอน), Validation (ตรวจสอบ), Test (ทดสอบ)
    """
    y = np.asarray(y)

    # 1. แบ่งข้อมูลชุด Test (ทดสอบขั้นสุดท้าย) ออกมาก่อน
    # ใช้ stratify=y เพื่อรักษาสัดส่วนของแต่ละคลาสให้สมดุลกัน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )

    # 2. คำนวณสัดส่วนใหม่ แล้วนำข้อมูลส่วนที่เหลือมาแบ่งเป็นชุด Train และ Validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
