import cv2
import numpy as np

def preprocess_image(image, img_size=100):
    """ย่อขนาดภาพและแปลงสีให้ถูกต้อง คืนค่าเป็น None ถ้าภาพใช้ไม่ได้"""

    if image is None or image.size == 0:
        return None

    # cv2 จะอ่านภาพเข้ามาเป็น BGR ต้องแปลงเป็น RGB เพื่อให้สีถูกต้อง
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # กรณีภาพขาวดำ
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # กรณีภาพสีทั่วไป

    # ย่อขนาดภาพให้เป็นสี่เหลี่ยมจัตุรัสตาม img_size (INTER_AREA เหมาะกับการย่อภาพที่สุด)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image

def to_features(images):
    """
    แปลงข้อมูลภาพเป็น array ที่พร้อมเข้าสู่โมเดล
    - เก็บข้อมูลเป็นชนิด uint8 (0-255) แทนที่จะเป็น float32 เพื่อประหยัด RAM ถึง 4 เท่า
    - การหารค่าสีให้อยู่ในช่วง 0-1 (หรือ -1 ถึง 1) จะไปทำในเลเยอร์ของตัวโมเดลแทน
    """
    return np.ascontiguousarray(images, dtype=np.uint8)
