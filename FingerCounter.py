import cv2
import time
import os
import HandTrackingModule as htm  # Assuming HandTrackingModule is a custom module for hand tracking


# Function to load images from a folder
def load_images(folder_path):
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print(f"Error loading image: {path}")
    return images


# Function to determine which overlay image to display based on finger counts
def get_display_image(fingers, overlay_images):
    if fingers is None or len(fingers) != 5:
        return None

    num_fingers = sum(fingers)

    if num_fingers == 1:
        return overlay_images[0]
    elif num_fingers == 2:
        return overlay_images[1]
    elif num_fingers == 3:
        return overlay_images[2]
    elif num_fingers == 4:
        return overlay_images[3]
    elif num_fingers == 5:
        return overlay_images[4]
    else:
        return None


# Camera parameters
wCam, hCam = 640, 480

# Choose the camera to use (0 for default camera, 1 for secondary camera)
camera_index = 0
cap = cv2.VideoCapture(camera_index)
cap.set(3, wCam)
cap.set(4, hCam)

# Folder path containing PNG images
folder_path = "nums"

# Load PNG images
overlay_images = load_images(folder_path)
print(f"Number of overlay images loaded: {len(overlay_images)}")

# Initialize variables
pTime = 0
detector = htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    # Read from the camera
    success, img = cap.read()

    if not success:
        print("Failed to read from camera.")
        break

    # Find hands and landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers = detector.fingersUp()  # Get fingers up status using HandTrackingModule function

    # Ensure fingers is not None and has exactly 5 elements (one for each finger)
    if fingers is None or len(fingers) != 5:
        fingers = [0, 0, 0, 0, 0]  # Assume no fingers are raised if detection fails

    # Determine which overlay image to display based on finger combination
    display_image = get_display_image(fingers, overlay_images)

    # Display the chosen overlay image if available
    if display_image is not None:
        h, w, c = display_image.shape
        if h <= img.shape[0] and w <= img.shape[1]:
            img[0:h, 0:w] = display_image

    # Display the total number of fingers counted
    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(sum(fingers)), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display the image with overlays
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
