import os
import shutil
import random
from pathlib import Path

def split_dataset():

    base_dir = Path("emotionsDetect")
    
    split_ratio = 0.8
    
    classes = ["happy", "sad"]
    
    for cls in classes:
        cls_dir = base_dir / cls
        if not cls_dir.exists():
            print(f"ไม่พบโฟลเดอร์ {cls_dir}")
            continue
            
        images = [f for f in os.listdir(cls_dir) if os.path.isfile(cls_dir / f)]
        
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        train_dir = base_dir / "train" / cls
        val_dir = base_dir / "val" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_images:
            shutil.move(str(cls_dir / img), str(train_dir / img))
            
        for img in val_images:
            shutil.move(str(cls_dir / img), str(val_dir / img))
            
        if not os.listdir(cls_dir):
            os.rmdir(cls_dir)
            
        print(f"คลาส '{cls}': แบ่งเป็น Train {len(train_images)} รูป, Val {len(val_images)} รูป")

    print("\nจัดโครงสร้างโฟลเดอร์เสร็จเรียบร้อย!")

if __name__ == '__main__':
    split_dataset()