"""
YOLOv7 Object Detection Module
Reference: Ultralytics YOLOv7, open-source object detection framework
NOTE: Animal classes have been REMOVED per project requirements.
Only obstacle/object/plant categories retained.
"""
import numpy as np

# Detection class list - ANIMAL CLASSES REMOVED
# Only keep: obstacles, objects, vegetation, structures
ALLOWED_CLASSES = {
    0: 'person',
    1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck',
    9: 'traffic light', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    # NOTE: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    # All animal classes REMOVED per project requirements
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse',
    66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
    70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
    75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear',
    79: 'hair drier', 80: 'toothbrush',
}

class PanoramicDetector:
    """
    Object detection wrapper for panoramic camera feed.
    Detection logic encapsulated as standalone callable function.
    Animal categories EXCLUDED per system requirements.
    """
    def __init__(self, confidence_threshold: float = 0.5):
        self.conf_threshold = confidence_threshold
        self.allowed_classes = list(ALLOWED_CLASSES.keys())
        
    def detect(self, frame: np.ndarray) -> list:
        """
        Run object detection on input frame.
        Wrapped as independent function per requirements.
        Returns: list of detections {bbox, class_name, confidence}
        """
        # In production: call YOLOv7 inference engine here
        # detections = yolo_model(frame, conf=self.conf_threshold)
        
        # Placeholder structure
        results = []
        # Example detection output structure:
        # results.append({
        #     'bbox': [x1, y1, x2, y2],
        #     'class_id': class_id,
        #     'class_name': ALLOWED_CLASSES.get(class_id, 'unknown'),
        #     'confidence': conf
        # })
        return results
    
    def filter_animal_classes(self, detections: list) -> list:
        """
        Filter out any animal class detections.
        Redundant safety check - animal classes already excluded from model.
        """
        animal_ids = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # COCO animal class IDs
        return [d for d in detections if d['class_id'] not in animal_ids]
