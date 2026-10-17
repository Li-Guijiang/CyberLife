# 真·摄像头视觉演示
# 打开摄像头，实时检测红色障碍物和绿色植被
import cv2
import numpy as np

# 解决中文显示
cv2.namedWindow('CyberLife 视觉系统', cv2.WINDOW_NORMAL)

print("="*50)
print("CyberLife 视觉系统实时演示")
print("打开摄像头，检测红色障碍物/绿色植被")
print("按 q 退出")
print("="*50)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 检测红色障碍物
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
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
            cv2.putText(frame, f'obstacle {area:.0f}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 检测绿色植被
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([77, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f'vegetation', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 顶部信息栏
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, 'CyberLife Vision | Animal classes removed | FEC+H265-SVC', 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('CyberLife 视觉系统', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
