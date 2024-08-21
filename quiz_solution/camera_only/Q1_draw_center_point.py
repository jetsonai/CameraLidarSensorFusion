import cv2
from ultralytics import YOLO

# Image path
image_path = '/home/ubuntu/Desktop/object_detection/infer_data/images/image_02/data/0000000000.png'

# Load the YOLOv8 model 
model = YOLO('yolov8n.pt')  

# Load the image
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Initialize object count
object_count = 0

# Process detection results
for result in results:
    for box in result.boxes:
        # Bounding box coordinates (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # Draw a green dot at the center
        cv2.circle(image, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)
        
        # Draw a red bounding box around the object
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)
        
        # Increment object count
        object_count += 1

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
text_color = (0, 0, 255)  # BGR: Red
background_color = (50, 50, 50)  # BGR: Dark Gray

# Define the position for the object count text
text_position = (10, 30)

# Get the size of the text box
(text_width, text_height), baseline = cv2.getTextSize(f'Objects: {object_count}', font, font_scale, thickness=2)

# Draw the background rectangle for the text
cv2.rectangle(image, 
              (text_position[0], text_position[1] - text_height - baseline), 
              (text_position[0] + text_width + 10, text_position[1] + baseline), 
              background_color, 
              thickness=cv2.FILLED)

# Draw the object count on the image
cv2.putText(image, f'Objects: {object_count}', text_position, font, font_scale, text_color, thickness=2)

# Save the final image
save_path = 'q1_result.jpg'
cv2.imwrite(save_path, image)

# Display the result image (optional)
# cv2.imshow('Detected Objects', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
