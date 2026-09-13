import matplotlib

# ปิดระบบหน้าต่างแสดงผลรูปของ matplotlib ชั่วคราว เพื่อไม่ให้ error เวลาสั่งเซฟภาพรันผ่าน command line
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_test, predictions, classes, save_path=None):
    """คำนวณและแสดงผลการประเมินความแม่นยำของโมเดล"""

    labels = list(range(len(classes)))

    # 1. คำนวณความแม่นยำรวม (Accuracy)
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    # 2. สร้างรายงานประเมินความแม่นยำรายคลาส (Precision, Recall, F1-Score)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    print("Confusion Matrix:")
    # 3. สร้างตารางแสดงความคลาดเคลื่อน (ดูว่าทายผิดเป็นคลาสไหนบ้าง)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy

def plot_confusion_matrix(matrix, classes, save_path):
    """วาดตาราง Confusion Matrix ออกมาเป็นรูปภาพ"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues") # ใช้เฉดสีน้ำเงิน

    # ตั้งค่าชื่อแกนและฉลาก
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted") # ทายว่า
    ax.set_ylabel("True")      # ของจริงคือ
    ax.set_title("Confusion Matrix")

    # ใส่ตัวเลขจำนวนครั้งที่ทายลงไปในแต่ละช่อง
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            # เปลี่ยนสีตัวอักษรให้ตัดกับพื้นหลัง (ถ้าพื้นเข้มใช้สีขาว พื้นอ่อนใช้สีดำ)
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_history(history, save_path):
    """สร้างกราฟเส้นแสดงประวัติความแม่นยำ (Accuracy) และข้อผิดพลาด (Loss) ตลอดการเทรน"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # กราฟความแม่นยำ
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    # กราฟความคลาดเคลื่อน (Loss)
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")