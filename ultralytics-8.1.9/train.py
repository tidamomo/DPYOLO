import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/DP-YOLO/yolov8s-dp.yaml')
    model.train(data='E:\shujuji\jiaotongbiaozhi\zhongguo640150\\data.yaml',
                # cache=True,
                imgsz=640,
                epochs=300,
                batch=16,
                patience=0,  # (int) epochs to wait for no observable improvement for early stopping of training
                # close_mosaic=10,
                workers=8,
                device='0',
                # cos_lr= True,
                # momentum=0.937,  #动量参数
                # lr0=0.01,  #学习率
                # optimizer='SGD', # using SGD
                # close_mosaic=0,
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
