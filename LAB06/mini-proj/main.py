import json
import os
import numpy as np

# นำเข้าฟังก์ชันจากไฟล์ต่างๆ
from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from nn_model import train_model, predict_model
from evaluate import evaluate_model, plot_history

# กำหนดเส้นทางไฟล์ เพื่อให้รันโปรแกรมจากตำแหน่งไหนก็ได้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "emotionsDetect/train")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ตั้งค่า Hyperparameters 
IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 3000   # จำนวนภาพสูงสุดต่อคลาส (None = ใช้ทั้งหมด)
EPOCHS = 150           # จำนวนรอบฝึกสูงสุด
BATCH_SIZE = 64        # จำนวนภาพต่อการสอน 1 ครั้ง

def main():
    print("--" * 30)
    print("Neural Network Image Recognition: emotionsDetect")
    print("--" * 30)

    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ (ถ้ายังไม่มี)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: อ่านไฟล์ภาพ
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: จัดการและแปลงภาพให้อยู่ในฟอร์แมตข้อมูล (Features)
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    np.save(f"{OUTPUT_DIR}/features.npy", X)
    print(f"Feature shape: {X.shape}")

    # Step 3: แบ่งชุดข้อมูล (Train, Validation, Test)
    print("\n[Step 3] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    # บันทึกข้อมูลที่แบ่งแล้วลงไฟล์ เพื่อความสะดวกรวดเร็วในการรันครั้งหน้า
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_val.npy", X_val)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_val.npy", y_val)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: เทรนโมเดล
    print("\n[Step 4] Training model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )
    print("Training completed.")

    # Step 5: นำข้อมูลชุดทดสอบมาทำนายผล
    print("\n[Step 5] Testing model...")
    predictions = predict_model(model, X_test)

    # Step 6: ประเมินความแม่นยำและสร้างกราฟรายงาน
    print("\n[Step 6] Evaluating model...")
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    plot_history(history, f"{OUTPUT_DIR}/training_history.png")

if __name__ == "__main__":
    main()
