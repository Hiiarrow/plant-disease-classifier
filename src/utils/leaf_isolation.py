import cv2
import numpy as np
from PIL import Image

def isolate_leaf_roi(pil_img: Image.Image, min_contour_area: int = 500) -> Image.Image:
    """
    Automated Foliar Segmentation & Background Clutter Removal.
    Converts PIL Image to OpenCV BGR, applies HSV/Lab color space masking,
    finds the primary foliar leaf contour, masks out non-leaf background (soil/pots/gravel),
    and crops tightly around the leaf ROI.
    """
    # Convert PIL Image to OpenCV BGR numpy array
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]

    # Convert to HSV color space for foliar green/yellow/diseased brown segmentation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Broad foliar health & lesion color ranges (Greens, Yellows, Browns, Necrotic Oranges)
    # Range 1: Healthy Green Foliage
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Range 2: Yellowish / Chlorotic / Fungal Discolored Foliage
    lower_yellow = np.array([12, 30, 40])
    upper_yellow = np.array([25, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Range 3: Brown / Necrotic / Rot Lesion Foliage
    lower_brown = np.array([0, 20, 20])
    upper_brown = np.array([12, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine foliar color masks
    foliar_mask = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_yellow, mask_brown))

    # Morphological Operations to clean noise & fill leaf gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(foliar_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)

    # Find contours and extract primary leaf ROI
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: Return original image if no foliar contour detected
        return pil_img

    # Select largest foliar contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < min_contour_area:
        return pil_img

    # Create final binary leaf mask
    final_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Mask out background (replace non-foliar pixels with neutral white/gray)
    masked_bgr = np.full_like(img_bgr, 255)
    masked_bgr[final_mask == 255] = img_bgr[final_mask == 255]

    # Crop tight bounding box around the leaf ROI
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    # Add a small padding margin
    margin = 5
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    cropped_leaf_bgr = masked_bgr[y1:y2, x1:x2]

    # Convert back to PIL RGB Image
    cropped_leaf_rgb = cv2.cvtColor(cropped_leaf_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_leaf_rgb)
