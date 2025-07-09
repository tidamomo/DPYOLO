import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/exp17/weights/best.pt')
    model.val(data='E:\shujuji\jiaotongbiaozhi\zhongguo640150\\data.yaml',
              split='test',
              imgsz=640,
              batch=1,
              project='runs/test',
              name='exp',
              )
