import json
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

def build_model(input_shape, num_classes):
    """สร้างโมเดลโดยใช้ Transfer Learning จาก MobileNetV2"""
    
    # 1. โหลดโมเดล MobileNetV2 (include_top=False คือไม่เอาส่วนหัวที่ใช้ทายผลของเดิม)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    # แช่แข็งความรู้เดิมไว้ ไม่ให้การเทรนไปทำลายโครงสร้างที่มันจำมา
    base_model.trainable = False

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # 2. Data Augmentation: เพิ่มความหลากหลายให้ภาพ (สุ่มพลิก, หมุน, ซูม) 
        # ช่วยให้โมเดลเก่งขึ้นเมื่อเจอภาพในมุมมองใหม่ๆ
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # 3. ปรับสเกลสี: สถาปัตยกรรม MobileNetV2 ต้องการค่าสีช่วง -1 ถึง 1
        layers.Rescaling(1.0 / 127.5, offset=-1),

        # 4. ใส่โมเดลหลัก (สมองส่วนที่เรียนรู้ลักษณะภาพมาแล้ว)
        base_model,

        # 5. ยุบขนาดภาพด้วย GlobalAveragePooling2D (ประหยัดแรมและเร็วกว่า Flatten ปกติ)
        layers.GlobalAveragePooling2D(),

        # 6. สร้างสมองส่วนท้ายสำหรับทายผลลัพธ์ของโปรเจกต์เรา
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5), # ปิดการทำงานเซลล์ประสาทครึ่งนึงแบบสุ่ม ป้องกันการจำข้อสอบ (Overfitting)

        # เลเยอร์สุดท้าย: ถ้ามี 2 คลาสใช้ sigmoid (ทาย 0 หรือ 1), ถ้าหลายคลาสใช้ softmax (ทายความน่าจะเป็นรวมเป็น 1)
        layers.Dense(
            1 if num_classes == 2 else num_classes,
            activation="sigmoid" if num_classes == 2 else "softmax"
        ),
    ])

    # กำหนดค่า Optimizer และ Loss Function ให้เหมาะกับจำนวนคลาส
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if num_classes == 2
             else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32):
    """สร้าง เทรน และบันทึกโมเดล"""

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        # EarlyStopping: สั่งหยุดเทรนถ้าค่า val_loss ไม่ลดลงติดต่อกัน 15 รอบ เพื่อไม่ให้เสียเวลา
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        # ReduceLROnPlateau: ลดอัตราการเรียนรู้ (Learning Rate) ลงครึ่งนึง ถ้าโมเดลเริ่มตันและเรียนรู้ต่อไม่ได้
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # บันทึกโมเดลและประวัติการเทรน
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(os.path.join(output_dir, "nn_model.keras"))
        
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f)
        print(f"Saved: {os.path.join(output_dir, 'nn_model.keras')}")

    return model, history

def predict_model(model, X_test):
    """ใช้โมเดลทำนายผลลัพธ์จากชุดข้อมูลที่ส่งเข้ามา"""
    probabilities = model.predict(X_test, verbose=0)

    # แปลงความน่าจะเป็นให้ออกมาเป็นหมายเลขคลาส
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() > 0.5).astype(int) # กรณีมี 2 คลาส

    return probabilities.argmax(axis=1) # กรณีมีหลายคลาส เลือกคลาสที่มีเปอร์เซ็นต์สูงสุด