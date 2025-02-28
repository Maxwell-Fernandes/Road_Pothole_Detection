import cv2
import numpy as np

# Load the image
image = cv2.imread('pothole_image.jpg')
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)

# Apply median filter for noise reduction
denoised_image = cv2.medianBlur(gray_image, 9)
cv2.imshow('Denoised Image', denoised_image)

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('Binary Image', binary_image)

# Apply morphological closing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closed Image', closed_image)

# Find contours
contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract candidate regions
candidate_regions = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Check if area is zero to avoid division by zero
    if area == 0:
        continue

    compactness = (perimeter ** 2) / (4 * np.pi * area)

    if area > 512 and compactness < 0.05:  # Thresholds for size and compactness
        candidate_regions.append(contour)

# Refine candidate regions
refined_regions = []
for contour in candidate_regions:
    hull = cv2.convexHull(contour)
    refined_regions.append(hull)

# Calculate features and classify
def calculate_ohi(roi, background_region):
    hist_roi = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_background = cv2.calcHist([background_region], [0], None, [256], [0, 256])
    ohi = cv2.compareHist(hist_roi, hist_background, cv2.HISTCMP_INTERSECT)
    return ohi

# Assume a background region for comparison
background_region = gray_image[0:100, 0:100]
cv2.imshow('Background Region', background_region)

potholes = []
for contour in refined_regions:
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray_image[y:y+h, x:x+w]

    std_dev = np.std(roi)
    ohi = calculate_ohi(roi, background_region)

    if std_dev < 10 and ohi > 0.8:  # Thresholds for standard deviation and OHI
        potholes.append(contour)

# Draw bounding boxes around detected potholes
output_image = image.copy()
for contour in potholes:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Potholes', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
