"""
CyberLife 视觉检测演示脚本
[CyberLife Modified] 2026-10 | 移除动物类别，封装为独立函数 | 软著标记
直接运行，会打开你的摄像头，实时检测物体
"""
import cv2
import numpy as np
import ssl
import urllib.request

# 解决SSL证书验证问题（国内网络环境）
ssl._create_default_https_context = ssl._create_unverified_context

from ultralytics import YOLO

# [CyberLife Modified] 2026-10 | 动物类别已全部移除 | 软著标记
# COCO动物类别ID列表（已从检测范围中排除）
ANIMAL_IDS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

# 加载YOLO模型（第一次运行会自动下载，已关闭SSL验证）
print("正在加载模型...")
try:
    model = YOLO("yolov8n.pt")  # 轻量版，跑得快
except Exception as e:
    print("模型加载失败，请检查网络，使用预训练模型离线运行")
    model = None

# 打开摄像头
cap = cv2.VideoCapture(0)
print("摄像头已打开，对着东西晃一晃吧！")
print("按 q 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if model is None:
        # 模型加载失败时显示提示
        cv2.putText(frame, "Model loading failed - check network", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("CyberLife Vision Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # 运行检测
    results = model(frame, verbose=False)
    
    # [CyberLife Modified] 2026-10 | 过滤动物类别，封装为独立函数 | 软著标记
    # 画检测框
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls)
            
            # 跳过动物类别！
            if cls_id in ANIMAL_IDS:
                continue
                
            # 画框
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            cls_name = model.names[cls_id]
            
            # 画绿色框
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 写标签
            cv2.putText(frame, f"{cls_name} {conf:.2f}", 
                        (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 显示画面
    cv2.imshow("CyberLife Vision Demo - Animal Classes Removed", frame)
    
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("演示结束！")
