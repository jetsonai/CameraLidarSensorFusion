import cv2

# Path to the image file
image_path = '/home/ubuntu/Desktop/sensor_fusion/infer_data/images/image_02/data/0000000000.png'

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print("Unable to load the image.")
else:
    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)

    # Display the original grayscale image and the edges
    cv2.imshow('Grayscale Image', image)
    cv2.imshow('Canny Edges', edges)

    cv2.waitKey(0)
    # Terminate pressing 'ESC' key
    cv2.destroyAllWindows()
