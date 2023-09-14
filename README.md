# Real-time object tracking

- This project aims to detect moving objects in a video and keep track of them.
- YOLOv8 is used to detect and localize each object in each frame. 
- The DeepSort library from [nwojke](https://github.com/nwojke/deep_sort) is used and modified to fit with the project and Tensorflow 2.X.

In this project: 
- Objects' detections are obtained using Yolov8
- Non-max suppression is applied to the detections obtained from the YOLOv8 model in order to eliminate the overlapping bounding boxes using intersection over union
- The tracker is updated with the output of the non-max suppression 
- Finally, a new video is produced using the updated tracks.

The test video:




