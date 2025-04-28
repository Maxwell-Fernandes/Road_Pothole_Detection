import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from matplotlib.patches import Rectangle
import os


def choose_image():
    """ Open a file dialog to select an image """
    Tk().withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path


def preprocess_image(image):
    """
    Convert to grayscale, apply histogram equalization & blur
    IMPROVED: Added CLAHE for better contrast enhancement in varied lighting
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # IMPROVED: Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return blurred, gray, enhanced


def detect_edges(image):
    """
    Apply edge detection
    IMPROVED: Auto-adjusting thresholds based on image mean
    """
    # Dynamically calculate thresholds based on image statistics
    sigma = 0.33
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    edges = cv2.Canny(image, lower, upper)
    return edges


def segment_potholes(image):
    """
    Use advanced thresholding and morphological operations to detect potholes
    IMPROVED: Added adaptive thresholding and more sophisticated morphological operations
    """
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Also apply Otsu's thresholding as a comparison
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine the two thresholding methods (can improve results in varying conditions)
    combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

    # Morphological operations to refine the results
    kernel = np.ones((5, 5), np.uint8)

    # Opening (erosion followed by dilation) to remove small noise
    opening = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing (dilation followed by erosion) to fill small holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closing, adaptive_thresh, otsu_thresh


def find_potholes(image, original):
    """
    Detect potholes using contours with improved filtering
    IMPROVED: Added better filtering criteria and classification confidence
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store pothole data
    potholes = []
    pothole_data = []

    # Get image dimensions for relative size calculations
    img_height, img_width = original.shape[:2]
    img_area = img_height * img_width

    # Result image for visualization
    result = original.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # IMPROVED: Filter using relative area (percentage of image)
        relative_area = area / img_area * 100

        # Only consider objects that are between 0.05% and 5% of the image area
        # (these thresholds can be adjusted based on your specific needs)
        if 0.05 < relative_area < 5:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Calculate perimeter and circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Define confidence score based on multiple features
            # Higher score = more likely to be a pothole
            confidence = 0

            # Aspect ratio check (potholes tend to be somewhat circular)
            if 0.5 < aspect_ratio < 2.0:
                confidence += 30
            elif 0.3 < aspect_ratio < 3.0:
                confidence += 15

            # Circularity check (potholes often have higher circularity)
            if circularity > 0.6:
                confidence += 30
            elif circularity > 0.4:
                confidence += 15

            # Solidity check (potholes tend to be somewhat solid/filled)
            if solidity > 0.8:
                confidence += 20
            elif solidity > 0.6:
                confidence += 10

            # Only include if confidence is sufficient
            if confidence >= 40:
                potholes.append(contour)
                pothole_data.append({
                    'contour': contour,
                    'confidence': confidence,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity
                })

    # Draw contours with color indicating confidence
    for data in pothole_data:
        # Color based on confidence (red->yellow->green from low->high)
        confidence = data['confidence']
        if confidence < 50:
            color = (0, 0, 255)  # Red for lower confidence
        elif confidence < 70:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 255, 0)  # Green for high confidence

        cv2.drawContours(result, [data['contour']], -1, color, 2)

        # Add bounding box and confidence score
        x, y, w, h = data['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)
        cv2.putText(result, f"{confidence}%", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result, pothole_data


def visualize_results(original, steps_data, pothole_data):
    """
    NEW FUNCTION: Create comprehensive visualization of all processing steps
    """
    # Unpack the processing steps
    gray, enhanced, blurred = steps_data['preprocessing']
    edges = steps_data['edges']
    combined_thresh, adaptive_thresh, otsu_thresh = steps_data['segmentation']
    final_result = steps_data['final_result']

    # Create a figure with subplots
    plt.figure(figsize=(15, 12))

    # 1. Original image
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 2. Grayscale
    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    # 3. Enhanced contrast (CLAHE)
    plt.subplot(3, 3, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Contrast (CLAHE)")
    plt.axis('off')

    # 4. Blurred
    plt.subplot(3, 3, 4)
    plt.imshow(blurred, cmap='gray')
    plt.title("Gaussian Blur")
    plt.axis('off')

    # 5. Edge detection
    plt.subplot(3, 3, 5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')

    # 6. Adaptive Threshold
    plt.subplot(3, 3, 6)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title("Adaptive Threshold")
    plt.axis('off')

    # 7. Otsu's Threshold
    plt.subplot(3, 3, 7)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu's Threshold")
    plt.axis('off')

    # 8. Combined Thresholds + Morphology
    plt.subplot(3, 3, 8)
    plt.imshow(combined_thresh, cmap='gray')
    plt.title("Combined Thresholds + Morphology")
    plt.axis('off')

    # 9. Final Result
    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Potholes ({len(pothole_data)} found)")
    plt.axis('off')

    plt.tight_layout()

    # NEW: Create a detailed summary figure with pothole analyses
    if pothole_data:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        plt.title(f"Pothole Analysis (Total: {len(pothole_data)})")
        plt.axis('off')

        # Add a color-coded legend for confidence levels
        plt.figtext(0.15, 0.02, "Red: Low confidence (<50%)", color='red')
        plt.figtext(0.45, 0.02, "Yellow: Medium confidence (50-70%)", color='yellow')
        plt.figtext(0.75, 0.02, "Green: High confidence (>70%)", color='green')

    plt.tight_layout()
    plt.show()


def save_results(original_path, final_result, pothole_data):
    """
    NEW FUNCTION: Save the detection results and generate a simple report
    """
    # Create results directory next to the original image
    base_dir = os.path.dirname(original_path)
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    results_dir = os.path.join(base_dir, f"{base_name}_pothole_results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the annotated image
    result_path = os.path.join(results_dir, f"{base_name}_detected.jpg")
    cv2.imwrite(result_path, final_result)

    # Create a simple report
    report_path = os.path.join(results_dir, f"{base_name}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"Pothole Detection Report for {os.path.basename(original_path)}\n")
        f.write(f"========================================================\n\n")
        f.write(f"Total potholes detected: {len(pothole_data)}\n\n")

        if pothole_data:
            f.write("Pothole details:\n")
            f.write("----------------\n")

            for i, data in enumerate(pothole_data, 1):
                f.write(f"Pothole #{i}:\n")
                f.write(f"  - Confidence: {data['confidence']}%\n")
                f.write(f"  - Area: {data['area']:.2f} pixels\n")
                f.write(f"  - Aspect ratio: {data['aspect_ratio']:.2f}\n")
                f.write(f"  - Circularity: {data['circularity']:.2f}\n")
                f.write(f"  - Location (x,y,w,h): {data['bbox']}\n\n")

        f.write(f"\nAnalysis completed on {os.path.basename(original_path)}")

    print(f"Results saved to {results_dir}")
    return results_dir


def main():
    # Select an image
    image_path = choose_image()
    if not image_path:
        print("No image selected. Exiting...")
        exit()

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()

    # Store original for comparison
    original = image.copy()

    # Apply improved processing pipeline
    print("Processing image...")

    # Step 1: Preprocess the image
    blurred, gray, enhanced = preprocess_image(image)

    # Step 2: Detect edges
    edges = detect_edges(blurred)

    # Step 3: Segment potential potholes
    combined_thresh, adaptive_thresh, otsu_thresh = segment_potholes(blurred)

    # Step 4: Find and classify potholes
    final_result, pothole_data = find_potholes(combined_thresh, original)

    # Organize the processing steps for visualization
    steps_data = {
        'preprocessing': (gray, enhanced, blurred),
        'edges': edges,
        'segmentation': (combined_thresh, adaptive_thresh, otsu_thresh),
        'final_result': final_result
    }

    # Visualize all processing steps
    visualize_results(original, steps_data, pothole_data)

    # Save results to disk
    if len(pothole_data) > 0:
        results_dir = save_results(image_path, final_result, pothole_data)
        print(f"Found {len(pothole_data)} potential potholes.")
        print(f"Detailed report saved to {results_dir}")
    else:
        print("No potholes detected in the image.")


if __name__ == "__main__":
    main()