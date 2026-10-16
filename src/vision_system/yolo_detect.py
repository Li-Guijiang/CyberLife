# 视觉检测模块
# 参考了OpenCV的检测实现，改了我们自己的类别过滤
# 按老师要求，动物类别全部删掉了
import cv2
import numpy as np

# 我们要识别的类别，动物都去掉了
ALLOWED_CLASSES = {
    'obstacle': '障碍物',
    'plant': '植被',
    'bag': '背包',
    'bottle': '瓶子',
    'chair': '椅子',
    'book': '书',
    'traffic_light': '红绿灯',
    'stop_sign': '停止标志',
}

class ObjectDetector:
    """
    目标检测器
    用OpenCV颜色空间检测，不依赖大模型，直接能跑
    已经把动物类别全部过滤掉了
    """
    def __init__(self):
        pass
        
    def detect(self, frame):
        """
        检测一帧图像里的物体
        检测红色障碍物（模拟金属目标）和绿色植被
        """
        h, w = frame.shape[:2]
        detections = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. 检测红色障碍物（模拟金属目标）
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append([x, y, x+bw, y+bh, 'obstacle', 0.92])
        
        # 2. 检测绿色植被
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([77, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append([x, y, x+bw, y+bh, 'plant', 0.85])
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        在图像上画检测框
        """
        for det in detections:
            x1, y1, x2, y2, name, conf = det
            color = (0, 0, 255) if name == 'obstacle' else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
