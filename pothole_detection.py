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

    # Apply Gaussian blur to reduce noise - smaller kernel for detailed images
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return blurred, gray, enhanced


def detect_edges(image, original=None):
    """
    Apply edge detection with improved handling for reflective surfaces
    IMPROVED: Added gradient-based detection to complement Canny edges
    """
    # Dynamically calculate thresholds based on image statistics
    sigma = 0.33
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    # Standard Canny edge detection
    edges = cv2.Canny(image, lower, upper)

    # IMPROVED: Add special handling for reflective surfaces like water
    if original is not None:
        # Convert to HSV to better detect brightness changes in water
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # Get gradients for detecting subtle changes in brightness (common in water)
        sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalize and convert to uint8
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold the gradient magnitude
        _, gradient_edges = cv2.threshold(gradient_mag, 30, 255, cv2.THRESH_BINARY)

        # Combine with Canny edges
        edges = cv2.bitwise_or(edges, gradient_edges)

    return edges


def detect_water(image, gray):
    """
    NEW FUNCTION: Detect potential water surfaces in the image
    """
    # Convert to HSV for better water detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract channels
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Water often has:
    # 1. Low to moderate saturation
    # 2. Specific hue range (often blue/gray)
    # 3. High brightness variance due to reflections

    # Create masks for potential water areas
    low_saturation = (s_channel < 50).astype(np.uint8) * 255

    # Get local variance of brightness (characteristic of reflective surfaces)
    kernel_size = 5
    mean_v = cv2.blur(v_channel, (kernel_size, kernel_size))
    mean_v2 = cv2.blur(v_channel * v_channel, (kernel_size, kernel_size))
    variance = mean_v2 - mean_v * mean_v

    # High variance areas (reflections)
    high_variance = (variance > 100).astype(np.uint8) * 255

    # Combine indicators
    water_mask = cv2.bitwise_and(low_saturation, high_variance)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return water_mask


def segment_potholes(image, original):
    """
    Use advanced thresholding and morphological operations to detect potholes
    IMPROVED: Added water detection and special handling for water-filled potholes
    """
    # Apply adaptive thresholding - adjusted for detailed textures
    adaptive_thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 9, 3
    )

    # Also apply Otsu's thresholding
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Combine the two thresholding methods
    combined = cv2.bitwise_or(adaptive_thresh, otsu_thresh)

    # IMPROVED: Detect and incorporate water areas
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    water_mask = detect_water(original, gray)

    # Water areas are likely part of potholes, enhance these regions
    water_enhanced = cv2.bitwise_or(combined, water_mask)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Opening to remove small noise
    opening = cv2.morphologyEx(water_enhanced, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing to fill small holes
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closing, adaptive_thresh, otsu_thresh, water_mask


def analyze_texture(image, contour):
    """
    NEW FUNCTION: Analyze texture features inside a contour
    Used to distinguish real potholes from other road features
    """
    # Create mask from contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], 0, 255, -1)

    # Extract ROI
    roi = cv2.bitwise_and(image, image, mask=mask)

    # Skip if no pixels are in the mask
    if np.sum(mask) == 0:
        return 0, 0, 0

    # Calculate texture features
    non_zero = roi[mask > 0]
    if len(non_zero) == 0:
        return 0, 0, 0

    # Statistical features
    mean_val = np.mean(non_zero)
    std_dev = np.std(non_zero)
    entropy = 0

    # Calculate entropy (texture complexity)
    if len(non_zero) > 1:
        hist = cv2.calcHist([roi], [0], mask, [32], [0, 256])
        hist = hist / np.sum(hist)  # Normalize
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))

    return mean_val, std_dev, entropy


def find_potholes(binary_image, original, edges=None):
    """
    Detect potholes using contours with improved filtering
    IMPROVED: More flexible shape detection and better handling of water reflections
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store pothole data
    potholes = []
    pothole_data = []

    # Get image dimensions
    img_height, img_width = original.shape[:2]
    img_area = img_height * img_width

    # Convert to grayscale for texture analysis
    if len(original.shape) == 3:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original.copy()

    # Result image for visualization
    result = original.copy()

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter using relative area (percentage of image)
        relative_area = area / img_area * 100

        # More flexible area range - include smaller potholes
        # Adjust the minimum threshold to catch smaller potholes
        if 0.1 < relative_area < 40:  # Wider range to be more inclusive
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h

            # Calculate shape features
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            # Get texture features
            mean_val, std_dev, entropy = analyze_texture(gray, contour)

            # Check if this region contains water (high std_dev often indicates reflections)
            has_water = std_dev > 30

            # Define confidence score - IMPROVED to be more flexible with shape
            confidence = 0

            # IMPROVED: More flexible shape criteria
            # Aspect ratio is less important - potholes can be any shape
            if 0.2 < aspect_ratio < 3.0:
                confidence += 20
            elif 0.1 < aspect_ratio < 5.0:  # Very permissive range
                confidence += 10

            # Circularity is also less strict
            if circularity > 0.4:  # Relaxed circularity requirement
                confidence += 20
            elif circularity > 0.2:
                confidence += 10

            # Solidity still matters - potholes are generally somewhat solid regions
            if solidity > 0.7:
                confidence += 20
            elif solidity > 0.5:
                confidence += 10

            # IMPROVED: Texture features boost confidence
            # Higher entropy/complexity often indicates potholes vs. plain road
            if entropy > 3.0:
                confidence += 15

            # Bonus for water detection (water collects in potholes)
            if has_water:
                confidence += 20

            # IMPROVED: Edge density check - potholes often have strong edges
            if edges is not None:
                # Create mask
                edge_mask = np.zeros_like(edges)
                cv2.drawContours(edge_mask, [contour], 0, 255, -1)

                # Count edge pixels in the contour area
                edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=edge_mask))
                edge_density = edge_pixels / area if area > 0 else 0

                # High edge density is characteristic of potholes
                if edge_density > 0.2:
                    confidence += 15
                elif edge_density > 0.1:
                    confidence += 10

            # More permissive confidence threshold - detect more potholes
            # We'd rather have false positives than miss actual potholes
            if confidence >= 30:  # Lowered from 40
                potholes.append(contour)
                pothole_data.append({
                    'contour': contour,
                    'confidence': confidence,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'has_water': has_water,
                    'entropy': entropy
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

        # Thicker outline for water-containing potholes
        thickness = 3 if data['has_water'] else 2
        cv2.drawContours(result, [data['contour']], -1, color, thickness)

        # Add bounding box and confidence score
        x, y, w, h = data['bbox']
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)

        # Add labels with more information
        label = f"{confidence}%"
        if data['has_water']:
            label += " (water)"

        cv2.putText(result, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return result, pothole_data


def visualize_results(original, steps_data, pothole_data):
    """
    Create comprehensive visualization of all processing steps
    IMPROVED: Added water mask visualization
    """
    # Unpack the processing steps
    gray, enhanced, blurred = steps_data['preprocessing']
    edges = steps_data['edges']
    combined_thresh, adaptive_thresh, otsu_thresh, water_mask = steps_data['segmentation']
    final_result = steps_data['final_result']

    # Create a figure with subplots (3x4 grid now)
    plt.figure(figsize=(16, 12))

    # 1. Original image
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 2. Grayscale
    plt.subplot(3, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')

    # 3. Enhanced contrast (CLAHE)
    plt.subplot(3, 4, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Contrast (CLAHE)")
    plt.axis('off')

    # 4. Blurred
    plt.subplot(3, 4, 4)
    plt.imshow(blurred, cmap='gray')
    plt.title("Gaussian Blur")
    plt.axis('off')

    # 5. Edge detection
    plt.subplot(3, 4, 5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')

    # 6. Water Mask (NEW)
    plt.subplot(3, 4, 6)
    plt.imshow(water_mask, cmap='Blues')
    plt.title("Water Detection")
    plt.axis('off')

    # 7. Adaptive Threshold
    plt.subplot(3, 4, 7)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title("Adaptive Threshold")
    plt.axis('off')

    # 8. Otsu's Threshold
    plt.subplot(3, 4, 8)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu's Threshold")
    plt.axis('off')

    # 9. Combined Thresholds + Morphology
    plt.subplot(3, 4, 9)
    plt.imshow(combined_thresh, cmap='gray')
    plt.title("Combined Thresholds + Morphology")
    plt.axis('off')

    # 10-11. Empty plots to maintain grid alignment
    plt.subplot(3, 4, 10)
    plt.axis('off')
    plt.subplot(3, 4, 11)
    plt.axis('off')

    # 12. Final Result
    plt.subplot(3, 4, 12)
    plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Potholes ({len(pothole_data)} found)")
    plt.axis('off')

    plt.tight_layout()

    # Create a detailed summary figure with pothole analyses
    if pothole_data:
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        plt.title(f"Pothole Analysis (Total: {len(pothole_data)})")
        plt.axis('off')

        # Add water-enhanced combined threshold
        plt.subplot(1, 2, 2)
        overlay = original.copy()
        # Create colored overlay of the water mask
        if len(original.shape) == 3:
            water_overlay = np.zeros_like(original)
            water_overlay[:, :, 0] = water_mask  # Add to blue channel
            overlay = cv2.addWeighted(overlay, 1, water_overlay, 0.5, 0)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Water Detection Overlay")
        plt.axis('off')

        # Add a color-coded legend for confidence levels
        plt.figtext(0.15, 0.02, "Red: Low confidence (<50%)", color='red')
        plt.figtext(0.40, 0.02, "Yellow: Medium confidence (50-70%)", color='yellow')
        plt.figtext(0.72, 0.02, "Green: High confidence (>70%)", color='green')

    plt.tight_layout()
    plt.show()


def save_results(original_path, final_result, pothole_data, water_mask=None):
    """
    Save the detection results and generate a detailed report
    IMPROVED: Added water detection information
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

    # Save water mask if available
    if water_mask is not None:
        water_path = os.path.join(results_dir, f"{base_name}_water_mask.jpg")
        cv2.imwrite(water_path, water_mask)

    # Create a detailed report
    report_path = os.path.join(results_dir, f"{base_name}_report.txt")

    with open(report_path, "w") as f:
        f.write(f"Pothole Detection Report for {os.path.basename(original_path)}\n")
        f.write(f"========================================================\n\n")
        f.write(f"Total potholes detected: {len(pothole_data)}\n")

        # Count water-containing potholes
        water_potholes = sum(1 for data in pothole_data if data.get('has_water', False))
        f.write(f"Potholes containing water: {water_potholes}\n\n")

        if pothole_data:
            f.write("Pothole details:\n")
            f.write("----------------\n")

            for i, data in enumerate(pothole_data, 1):
                f.write(f"Pothole #{i}:\n")
                f.write(f"  - Confidence: {data['confidence']}%\n")
                f.write(f"  - Area: {data['area']:.2f} pixels\n")
                f.write(f"  - Aspect ratio: {data['aspect_ratio']:.2f}\n")
                f.write(f"  - Circularity: {data['circularity']:.2f}\n")
                f.write(f"  - Contains water: {'Yes' if data.get('has_water', False) else 'No'}\n")
                f.write(f"  - Texture complexity: {data.get('entropy', 0):.2f}\n")
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

    # Step 2: Detect edges with improved water handling
    edges = detect_edges(blurred, original)

    # Step 3: Segment potential potholes with water detection
    combined_thresh, adaptive_thresh, otsu_thresh, water_mask = segment_potholes(blurred, original)

    # Step 4: Find and classify potholes with improved shape flexibility
    final_result, pothole_data = find_potholes(combined_thresh, original, edges)

    # Organize the processing steps for visualization
    steps_data = {
        'preprocessing': (gray, enhanced, blurred),
        'edges': edges,
        'segmentation': (combined_thresh, adaptive_thresh, otsu_thresh, water_mask),
        'final_result': final_result
    }

    # Visualize all processing steps
    visualize_results(original, steps_data, pothole_data)

    # Save results to disk
    if len(pothole_data) > 0:
        results_dir = save_results(image_path, final_result, pothole_data, water_mask)
        print(f"Found {len(pothole_data)} potential potholes.")

        # Count water potholes
        water_potholes = sum(1 for data in pothole_data if data.get('has_water', False))
        if water_potholes > 0:
            print(f"  - {water_potholes} potholes contain water.")

        print(f"Detailed report saved to {results_dir}")
    else:
        print("No potholes detected in the image.")


if __name__ == "__main__":
    main()