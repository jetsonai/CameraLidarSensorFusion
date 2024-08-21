"""
Q2. Make inference Video using YOLOv8 and OpenCV
 
0. Detect Object using YOLOv8n pretrained model.
1. Draw Object bounding Box (Person Green, Car Red, Bicicyle Blue, else White, border size 1.5)
2. Write Object Class above bounding box 
3. Inference all images in "/home/ubuntu/Desktop/object_detection/infer_data/images/image_02/data/0000000000.png" 
4. Make inference image to video (FPS 10)
"""

import cv2
import os
from ultralytics import YOLO

# Path to the image directory
image_dir = '/home/ubuntu/Desktop/object_detection/infer_data/images/image_02/data'

# Output video file name
output_video = 'q2_result.mp4'

# Load the YOLOv8 model 
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8m.pt')

# Get a list of all image files in the directory
image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Check if there are any images in the directory
if not image_files:
    print("No images found in the specified directory.")
else:
    # Load the first image to get the size
    first_image_path = os.path.join(image_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is often a safe codec for mp4
    video = cv2.VideoWriter(output_video, fourcc, 10, (width, height))  # 10 means 10fps

    # Dictionary to map class names to colors
    class_colors = {
        'car': (0, 0, 255),     # Red
        'person': (0, 255, 0),  # Green
        'bicycle': (255, 0, 0)  # Blue
    }
    default_color = (255, 255, 255)  # White for other classes

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping file {image_file}, unable to load.")
            continue

        # Perform object detection
        results = model(image)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                # Bounding box coordinates (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                
                # Class name and confidence score
                class_id = int(box.cls)
                class_name = model.names[class_id]
                confidence = float(box.conf)  # Convert the confidence score to a float
                
                # Select color based on class name
                color = class_colors.get(class_name, default_color)

                # Draw the bounding box with specified thickness
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)

                # Put the class name and confidence on top of the bounding box
                label = f'{class_name} {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y_min = max(y_min, label_size[1] + 10)
                cv2.rectangle(image, (x_min, label_y_min - label_size[1] - 10), (x_min + label_size[0], label_y_min), color, -1)
                cv2.putText(image, label, (x_min, label_y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Add the annotated image to the video
        video.write(image)

    # Release the VideoWriter
    video.release()
    print(f"Video saved as {output_video}")
