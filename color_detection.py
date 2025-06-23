import cv2
import numpy as np

# Define the HSV and BGR ranges for multiple colors
def get_color_bounds_and_bgr():
    color_data = {
        'red': {
            'hsv': (np.array([0, 120, 70], dtype="uint8"), np.array([10, 255, 255], dtype="uint8")),
            'bgr': (0, 0, 255)
        },
        'blue': {
            'hsv': (np.array([94, 80, 2], dtype="uint8"), np.array([126, 255, 255], dtype="uint8")),
            'bgr': (255, 0, 0)
        },
        'green': {
            'hsv': (np.array([40, 40, 40], dtype="uint8"), np.array([70, 255, 255], dtype="uint8")),
            'bgr': (0, 255, 0)
        },
        'orange': {
            'hsv': (np.array([10, 100, 20], dtype="uint8"), np.array([25, 255, 255], dtype="uint8")),
            'bgr': (0, 165, 255)
        },
        'yellow': {
            'hsv': (np.array([25, 100, 100], dtype="uint8"), np.array([35, 255, 255], dtype="uint8")),
            'bgr': (0, 255, 255)
        },
        'black': {
            'hsv': (np.array([0, 0, 0], dtype="uint8"), np.array([180, 255, 30], dtype="uint8")),
            'bgr': (0, 0, 0)
        },
        'white': {
            'hsv': (np.array([0, 0, 231], dtype="uint8"), np.array([180, 18, 255], dtype="uint8")),
            'bgr': (255, 255, 255)
        },
        # Add other colors similarly
    }
    return color_data

def initialize_camera(camera_index=0):
    # Directly open the camera index specified (avoiding the loop to speed up)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {camera_index}")
        return None
    print(f"Camera initialized at index {camera_index}")
    return cap

# Function to initialize the video capture and set parameters for speed
def setup_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Adjust resolution to speed up processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize the webcam
cap = initialize_camera(0)

if not cap:
    exit()

setup_camera(cap)

# Get color bounds and BGR values for all colors
color_data = get_color_bounds_and_bgr()

# Warm-up the camera (capture and discard first few frames)
for _ in range(5):
    cap.read()

# Main loop for color detection
while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read from camera")
        break

    # Convert the frame to HSV for color detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through each color in color_data
    for color_name, color_info in color_data.items():
        lower_color, upper_color = color_info['hsv']
        color_bgr = color_info['bgr']

        # Create a mask for the color detection
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)
        
        # Find contours in the mask (for color)
        color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in color_contours:
            if cv2.contourArea(contour) < 500:  # Skip small areas
                continue

            # Draw a rectangle around the color area with the corresponding color
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

            # Add label text on top of the rectangle with the corresponding color
            label = f"{color_name.capitalize()} detected"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

    # Show the current frame with color tracking
    cv2.imshow("Color Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
