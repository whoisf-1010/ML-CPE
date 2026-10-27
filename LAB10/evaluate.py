from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    # โหลดโมเดลที่เพิ่งเทรนเสร็จ
    weights = ROOT / 'runs' / 'classify' / 'runs' / 'emotion_model' / 'weights' / 'best.pt'
    if not weights.exists():
        weights = ROOT / 'runs' / 'classify' / 'emotion_model' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'หาไฟล์โมเดลไม่พบ: {weights}')

    model = YOLO(str(weights))

    # ประเมินผลกับชุดข้อมูล Validation
    metrics = model.val(
        data=str(ROOT / 'emotionsDetect'),
        split='val',
        device='cpu',
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    # แสดงผลความแม่นยำของ Classification
    print('\n===== Evaluation Results =====')
    print(f'Top-1 Accuracy: {metrics.top1:.4f}')
    print(f'Top-5 Accuracy: {metrics.top5:.4f}')

if __name__ == '__main__':
    main()