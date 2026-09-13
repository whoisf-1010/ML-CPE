"""
สคริปต์ทดสอบอิสระ: สุ่มหยิบภาพ 4 ภาพมาให้โมเดลทาย 
แล้วนำเสนอผลลัพธ์ออกมาเป็นรูปตารางตารางกริด (Grid 2x2)
(ต้องรันไฟล์ main.py ให้จบก่อน เพื่อให้มีไฟล์โมเดลถูกบันทึกไว้)
"""

import json
import os
import matplotlib

matplotlib.use("Agg") # ปิดหน้าต่างโชว์รูปภาพเพื่อป้องกัน Error ตอนสั่งเซฟ

import matplotlib.subplots as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4 # จำนวนภาพที่จะสุ่มมาเทส

def test_nn(n_samples=N_SAMPLES):
    
    # 1. โหลดโมเดล ข้อมูลชุดทดสอบ และชื่อคลาสที่เคยบันทึกไว้
    model = keras.models.load_model(f"{OUTPUT_DIR}/nn_model.keras")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)

    # 2. สุ่มหมายเลข Index เพื่อดึงภาพมาจำนวน n_samples ภาพ
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # 3. ให้โมเดลทำนายเปอร์เซ็นต์ความน่าจะเป็น
    probabilities = model.predict(X_sample, verbose=0)
    
    if probabilities.shape[-1] == 1:
        # กรณีทายผลแค่ 2 คลาส
        probabilities = probabilities.ravel()
        predictions = (probabilities > 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1 - probabilities)
    else:
        # กรณีทายผลมากกว่า 2 คลาส ดึงคำตอบที่เปอร์เซ็นต์สูงสุดออกมา
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    # 4. สร้างแผงรูปภาพเพื่อโชว์ผลลัพธ์ (แบบกริด 2x2 ถ้า n_samples = 4)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    # 5. วนลูปวาดรูปภาพ ใส่ข้อความคำตอบจริงและคำตอบที่ทายลงบนรูป
    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off") # ซ่อนกรอบที่ว่าง
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = predictions[i] == y_sample[i] # เช็คว่าทายถูกไหม
        
        color = "green" if correct else "red" # ทายถูกขึ้นสีเขียว ทายผิดขึ้นสีแดง

        ax.imshow(X_sample[i])
        ax.set_xticks([]) # ซ่อนขีดสเกลแกน X
        ax.set_yticks([]) # ซ่อนขีดสเกลแกน Y
        
        # ใส่หัวข้อบนรูปเป็น ชื่อที่ทาย (พร้อม % ความมั่นใจ) และ ชื่อที่ถูกต้อง
        ax.set_title(f"Pred: {pred} ({confidence[i] * 100:.0f}%)\n"
                     f"True: {true}", color=color)

        print(f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
              f"conf {confidence[i] * 100:5.1f}%  "
              f"{'OK' if correct else 'WRONG'}")

    # สรุปผลคะแนนรวมว่าทายถูกกี่ข้อ
    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    # บันทึกเป็นรูปภาพ
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    test_nn()