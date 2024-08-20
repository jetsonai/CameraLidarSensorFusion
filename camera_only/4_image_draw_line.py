import cv2

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the image in color
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Unable to load the image.")
else:
    # Draw a point (small circle) at (50, 50) with red color
    point_position = (50, 50)
    cv2.circle(image, point_position, radius=3, color=(0, 0, 255), thickness=-1)  # BGR: Red

    # Draw a line from zero point to end point with green color
    # Get the dimensions of the image
    height, width = image.shape[:2]
    print("wideh : ", width, " height : ", height)
    line_start = (0, 0)
    line_end = (width, height)
    cv2.line(image, line_start, line_end, color=(0, 255, 0), thickness=3)  # BGR: Green

    # Draw a circle centered at (300, 300) with a radius of 50 and blue color
    circle_center = (300, 300)
    cv2.circle(image, circle_center, radius=50, color=(255, 0, 0), thickness=3)  # BGR: Blue

    # Draw a rectangle with top-left corner at (200, 2000) and bottom-right corner at (400, 400) with yellow color
    rectangle_start = (200, 200)
    rectangle_end = (400, 400)
    cv2.rectangle(image, rectangle_start, rectangle_end, color=(0, 255, 255), thickness=3)  # BGR: Yellow

    # Display the image with the drawn shapes
    cv2.imshow('Image with Shapes', image)

    cv2.waitKey(0)
    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()
