import numpy as np

def calculate_iou(box1, boxes2):
    x1, y1, x2, y2 = box1
    x1 = np.maximum(x1, boxes2[:, 0])
    y1 = np.maximum(y1, boxes2[:, 1])
    x2 = np.minimum(x2, boxes2[:, 2])
    y2 = np.minimum(y2, boxes2[:, 3])

    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)

    iou = intersection_area / (box1_area + boxes2_area - intersection_area)
    return iou