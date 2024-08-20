from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8 model (using a pre-trained model)
model = YOLO('yolov8n.pt')  # 'n' stands for nano model, can be changed to s, m, l, x as needed

# Load the image
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'  # Set the image path
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Annotate the detected objects on the original image
annotated_image = results[0].plot()

# Display the result image using cv2.imshow
cv2.imshow('Detected Objects', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
