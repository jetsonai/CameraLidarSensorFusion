import cv2

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the image in color
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Unable to load the image.")
else:
    # Define the rectangle's top-left and bottom-right corners
    top_left_corner = (1051, 161)
    bottom_right_corner = (1172,  275)

    # Define the color of the rectangle (blue in BGR)
    rectangle_color = (255, 0, 0)  # BGR: Blue

    # Draw the rectangle
    cv2.rectangle(image, top_left_corner, bottom_right_corner, rectangle_color, thickness=3)

    # Define the text and its properties
    text = "detection box"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (0, 255, 0)  # BGR: Green
    background_color = (50, 50, 50)  # BGR: Dark Gray

    # Get the size of the text box
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness=2)

    # Calculate the position for the text (slightly below the top-left corner of the rectangle)
    text_x = top_left_corner[0] 
    text_y = top_left_corner[1] - text_height 

    # Draw the background rectangle for the text
    cv2.rectangle(image, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + baseline + 5), \
    background_color, thickness=cv2.FILLED)

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness=2)

    # Save the image
    save_path = 'save_image.jpg'
    cv2.imwrite(save_path, image)

    cv2.waitKey(0)
    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()


