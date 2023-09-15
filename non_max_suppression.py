import numpy as np
from iou import calculate_iou


def non_max_suppression(detections,  threshold):

    if len(detections) == 0:
        return [], [], []

    boxes= [box[0:4] for box in detections]
    scores= [box[4].item() for box in detections]
    classes= [box[5] for box in detections]

    boxes= np.array(boxes)
    scores= np.array(scores)
    classes= np.array(classes)

    # Initialize an empty list to store the selected boxes.
    selected_boxes = []
    selected_scores= []
    selected_classes= []

    # Sort the boxes by their confidence scores in descending order.
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    classes= classes[sorted_indices]

    while len(boxes) > 0:
        # Select the box with the highest confidence score and add it to the selected list.
        selected_boxes.append(boxes[0])
        selected_scores.append(scores[0])
        selected_classes.append(classes[0])

        # Calculate the IoU (Intersection over Union) between the selected box and the remaining boxes.
        iou = calculate_iou(selected_boxes[-1], boxes[1:])

        # Filter out the boxes with IoU greater than or equal to the threshold.
        mask = iou < threshold
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]

    return selected_boxes, selected_scores, selected_classes