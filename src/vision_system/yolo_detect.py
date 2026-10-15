# YOLOv7目标检测
# 参考了Ultralytics的YOLO实现，改了我们自己的类别列表
# 按老师要求，动物类别全部删掉了，只留障碍物和物品
import numpy as np

# 我们要识别的类别，动物都去掉了
# 原来COCO里的bird, cat, dog, horse, sheep, cow这些全删了
ALLOWED_CLASSES = {
    0: 'person',
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    9: 'traffic light', 11: 'stop sign', 12: 'bench',
    # 动物类别已全部移除！！！
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 28: 'suitcase',
    32: 'sports ball', 34: 'baseball bat',
    39: 'bottle', 41: 'cup', 45: 'bowl',
    46: 'banana', 47: 'apple', 50: 'broccoli', 51: 'carrot',
    56: 'chair', 57: 'couch', 58: 'potted plant',
    60: 'dining table', 63: 'tv', 64: 'laptop', 67: 'keyboard', 68: 'cell phone',
    73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase',
}

class ObjectDetector:
    """
    目标检测器
    就是把摄像头拍到的东西框出来，告诉你这是啥
    已经把动物类别全部过滤掉了
    """
    def __init__(self, conf_thresh=0.5):
        self.conf = conf_thresh
        self.allowed = list(ALLOWED_CLASSES.keys())
        
    def detect(self, frame):
        """
        检测一帧图像里的物体
        frame是摄像头拍的一帧
        返回检测到的物体列表
        """
        # 实际用的时候这里调YOLO模型
        # results = model(frame, conf=self.conf)
        results = []
        
        # 示例返回结构
        # results.append({
        #     'bbox': [x1, y1, x2, y2],
        #     'class_id': cid,
        #     'class_name': name,
        #     'confidence': conf
        # })
        
        return results
    
    def filter_animals(self, detections):
        """
        过滤掉动物类别
        双保险，就算模型出来了动物，这里也再删一遍
        """
        animal_ids = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        return [d for d in detections if d['class_id'] not in animal_ids]
