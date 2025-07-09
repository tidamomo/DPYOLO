import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import RTDETR

if __name__ == '__main__':
    model = YOLO('runs/train/exp17/weights/best.pt')
    # model = YOLO(r'F:\mgmj\ultralytics-20240601\ultralytics-main\\runs\\train\exp39\weights\\best.pt')
    # model.val(data='E:\\shujuji\\VisDrone2019\\VisDrone.yaml',
    # model.val(data='E:\shujuji\jiaotongbiaozhi\CCTSDB 2021\\data.yaml',
    # model.val(data='E:\\111111mypic\\3051test\\data.yaml',
    # model.val(data='E:\\111firesmoke\\fire smoke.v1i.yolov8\\data.yaml',
    model.val(data='E:\shujuji\jiaotongbiaozhi\zhongguo640150\\data.yaml',
    # model.val(data='E:\shujuji\jiaotongbiaozhi\zhongguo150\\data.yaml',
    # model.val(data='E:\shujuji\jiaotongbiaozhi\deguokeshan\\data.yaml',
              split='test',
              imgsz=640,
              batch=1,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )