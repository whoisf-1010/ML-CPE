from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    weights = ROOT / 'runs' / 'classify' / 'runs' / 'emotion_model' / 'weights' / 'best.pt'
    if not weights.exists():
        weights = ROOT / 'runs' / 'classify' / 'emotion_model' / 'weights' / 'best.pt'

    image_dir = ROOT / 'emotionsDetect' / 'val'

    if not weights.exists():
        raise FileNotFoundError(f'หาไฟล์โมเดลไม่พบ: {weights}')

    model = YOLO(str(weights))

    extensions = {'.jpg', '.jpeg', '.png'}
    images = [p for p in image_dir.rglob('*') if p.suffix.lower() in extensions]

    if not images:
        raise RuntimeError('ไม่พบไฟล์รูปภาพทดสอบ')

    selected_images = random.sample(images, min(5, len(images)))

    for image in selected_images:
        results = model.predict(
            source=str(image),
            save=True,                  
            project=str(ROOT / 'runs'), 
            name='image_test',           
            verbose=False
        )

        for result in results:
            print(f'\nImage: {image.name}')
            top1_id = result.probs.top1
            top1_conf = float(result.probs.top1conf)
            class_name = result.names[top1_id]

            print(f'Prediction: {class_name} ({top1_conf * 100:.2f}%)')

            annotated = result.plot()
            cv2.imshow('Emotion Classification Test', annotated)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nTesting completed!')

if __name__ == '__main__':
    main()