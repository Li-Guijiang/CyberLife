# 开源依赖库清单
## CyberLife 具身载体智能系统

> 本清单对应项目需求中标注黄底的技术要求，记录所用开源库名称、仓库地址与技术用途。

---

## 1. 行静仿生稳定系统（步态控制）

| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **legged_gym** | https://github.com/leggedrobotics/legged_gym | 基于Isaac Gym的足式机器人强化学习训练框架，支持四足/八足步态生成 |
| **MIT Cheetah Software** | https://github.com/mit-biomimetics/Cheetah-Software | MIT猎豹机器人开源控制代码，包含步态生成、MPC控制 |
| **pybullet_robots** | https://github.com/erwincoumans/pybullet_robots | PyBullet仿真环境下的多足机器人模型与控制示例 |
| **osc-workspace** | https://github.com/Open-Source-Legged-Robot/osc-workspace | 开源足式机器人操作空间控制框架 |

**技术方向：** 多肢载体节律步态生成、机器人运动学正逆解算、关节空间参数寻优

---

## 2. 脑电方向控制系统

### 2.1 BCI 脑电信号处理
| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **MNE-Python** | https://github.com/mne-tools/mne-python | 开源EEG/MEG信号处理库，包含预处理、时频分析、源定位 |
| **BraindaCode** | https://github.com/braindecode/braindecode | 基于PyTorch的BCI深度学习工具包，支持CSP、ShallowConvNet等模型 |
| **scikit-learn** | https://github.com/scikit-learn/scikit-learn | SVM分类器、CSP特征提取、LDA判别分析 |
| **EEG-Dev-Tools** | https://github.com/NeuroTechX/eeg-dev-tools | 开源脑电开发工具集合 |

**技术方向：** EEG信号预处理、CSP共空间滤波、CNN/SVM特征提取、ERN事件相关电位校验

### 2.2 IMU 多陀螺仪姿态融合
| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **imu_filter_madgwick** | https://github.com/ritzinfo/imu_filter_madgwick | ROS包，Madgwick/Mahony AHRS姿态融合算法 |
| **filterpy** | https://github.com/rlabbe/filterpy | Python卡尔曼滤波库，EKF/UKF实现 |
| **RTIMULib2** | https://github.com/RTIMULib/RTIMULib2 | 多IMU传感器融合库，支持陀螺仪+加速度计+磁力计 |

**技术方向：** 多路陀螺仪角速度、倾角传感数据状态融合、载体本体姿态估计

---

## 3. 全景环境感知系统

### 3.1 目标检测
| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **YOLOv7** | https://github.com/WongKinYiu/yolov7 | 开源实时目标检测框架，COCO预训练模型 |
| **Ultralytics** | https://github.com/ultralytics/ultralytics | YOLO系列官方实现，支持v5/v8/v9/v10 |
| **OpenCV** | https://github.com/opencv/opencv | 计算机视觉基础库，图像预处理、视频读写 |

**技术方向：** 环境实体目标检测分类、裁剪目标类别集合（已移除动物类别）、封装独立感知推理函数

### 3.2 视频编码
| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **FFmpeg** | https://github.com/FFmpeg/FFmpeg | 开源多媒体框架，H.265/HEVC编解码支持 |
| **x265** | https://bitbucket.org/multicoreware/x265_git | H.265/HEVC开源编码器 |
| **SVC Extension** | https://github.com/vyz Barrero/h265-svc | H.265可伸缩编码研究实现 |

**技术方向：** 全景视频分层编码、远近景场景差异化码率自适应压缩

### 3.3 FEC抗丢包传输
| 库名 | GitHub链接 | 用途说明 |
|------|-----------|----------|
| **libfec** | https://github.com/ceph/ceph/tree/master/src/common | 开源Reed-Solomon FEC库 |
| **webrtc** | https://github.com/webrtc-sdk/webrtc | WebRTC实时通信框架，内置FEC/NACK机制 |
| **openfec** | https://github.com/OpenFEC/OpenFEC | 开源FEC库，多种前向纠错编码实现 |

**技术方向：** 媒体流前向纠错编码、弱网环境下保障视频流稳定传输

---

## 备注
- 所有库均为成熟开源项目，采用二次开发模式
- 修改的代码均添加注释标记，用于软件著作权登记
- 视觉检测模块已移除动物类别，仅保留障碍物、物品、植被等识别类别
