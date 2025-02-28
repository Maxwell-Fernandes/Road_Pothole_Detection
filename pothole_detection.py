import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def choose_image():
    """ Open a file dialog to select an image """
    Tk().withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path

def preprocess_image(image):
    """ Convert to grayscale, apply histogram equalization & blur """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_edges(image):
    """ Apply edge detection """
    edges = cv2.Canny(image, 50, 150)
    return edges

def segment_potholes(image):
    """ Use thresholding and morphological operations to detect potholes """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph

def find_potholes(image, original):
    """ Detect potholes using contours and filtering """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potholes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < 10000:  # Filter out very small or large objects
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Ensure the detected region is not too elongated (to remove cracks)
            if 0.3 < aspect_ratio < 2.5:
                potholes.append(contour)

    result = original.copy()
    cv2.drawContours(result, potholes, -1, (0, 255, 0), 2)

    return result, potholes

# Select an image
image_path = choose_image()
if not image_path:
    print("No image selected. Exiting...")
    exit()

# Load image
image = cv2.imread(image_path)

# Apply processing pipeline
preprocessed = preprocess_image(image)
edges = detect_edges(preprocessed)
segmented = segment_potholes(preprocessed)
final_result, potholes = find_potholes(segmented, image)

# Show results using Matplotlib (Fixes cv2.imshow issue)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.axis("off")  # Hide axis
plt.title(f"Detected Potholes ({len(potholes)} found)")
plt.show()
