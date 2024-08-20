from ultralytics import YOLO
import cv2

# Load the YOLOv8 model 
model = YOLO('yolov8n.pt')  

# Load the image
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png' 
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Print bounding box, class, and confidence score for each detected object
for result in results:
    for box in result.boxes:
        # Bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        
        # Class name and confidence score
        class_name = model.names[int(box.cls)]
        confidence = float(box.conf)  # Convert the confidence score to a float

        print(f"Detected {class_name} with confidence {confidence:.2f} at [{x_min}, {y_min}, {x_max}, {y_max}]")

# Annotate the detected objects on the original image
annotated_image = results[0].plot()

# Display the result image using cv2.imshow
cv2.imshow('Detected Objects', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
