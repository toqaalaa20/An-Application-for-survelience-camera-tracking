import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from tracker import Object_Tracker
from ultralytics import YOLO
import random
from non_max_suppression import non_max_suppression
from get_names import coco_names

# Initialize the tracker
tracker = Object_Tracker()

# Initialize the detection model
model = YOLO('yolov8n.pt')

# Create different colors for the tracks
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
names= coco_names()

# Adjusting the threshold for the detections
detection_threshold = 0.5


parser = argparse.ArgumentParser(description="Process video input from a file or camera.")

# Add a command-line argument for specifying the input source
parser.add_argument("--input", default=None, help="Path to the input video file or 'camera' for camera input")

# Parse the command-line arguments
args = parser.parse_args()

# Determine the input source
if args.input is None:
    print("Error: Please specify an input source using --input 'video.mp4' or --input 'camera'")
    exit(1)
elif args.input.lower() == "camera":
    # Use the camera as input
    vid = cv2.VideoCapture(0)  # 0 corresponds to the default camera
else:
    # Use a video file as input
    input_video_path = args.input
    vid = cv2.VideoCapture(input_video_path)

# Check if the input source was opened successfully
if not vid.isOpened():
    print("Error: Could not open input source.")
    exit(1)


while (True):
    # start_time = time.time()

    # for img in os.listdir(data_path):
    #     frame = cv2.imread(os.path.join(data_path, img))
    _, frame = vid.read()
    # Perform object detection using YOLO on the current frame
    results = model(frame)

    # Update the tracks using the DeepSort tracker
    for result in results:
        detections = []

        # Creating the detections array
        for r in result.boxes.data:
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            class_id = int(class_id)
            class_name = names[class_id]
            if score >= 0.3:
                detections.append([x1, y1, x2, y2, score, class_name])

        boxes, score, classes = non_max_suppression(detections, 0.5)
        detections = []
        for i in range(len(boxes)):
            b = []
            b.append(boxes[i][0])
            b.append(boxes[i][1])
            b.append(boxes[i][2])
            b.append(boxes[i][3])
            b.append(score[i])
            b.append(classes[i])

            detections.append(b)

        print(detections)

        tracker.update(frame, detections)
        # Updating the tracks with the new frame
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

            # Define the text you want to add
            text = track.class_name

            # Define the position where you want to place the text (adjust coordinates as needed)
            text_x = int(x1)  # You can adjust the x-coordinate as needed
            text_y = int(y1) - 10  # You can adjust the y-coordinate as needed

            # Get the color of the rectangle
            rectangle_color = colors[track_id % len(colors)]

            # Get the size of the text to calculate the size of the background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            background_width = text_width + 10
            background_height = text_height + 10

            # Define the position of the background rectangle
            background_x = text_x
            background_y = text_y - text_height

            # Draw the background rectangle
            cv2.rectangle(frame, (background_x, background_y),
                          (background_x + background_width, background_y + background_height), rectangle_color, -1)

            # Define the font and font scale
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5

            # Define the color of the text (you can change this if needed)
            text_color = (255, 255, 255)  # White color in BGR

            # Add the text to the frame
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, 1, cv2.LINE_AA)
        # Save the frame with tracking results
        cv2.imshow('img', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
vid.release()







