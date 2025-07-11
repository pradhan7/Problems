import cv2
import numpy as np

# Known circle diameter (1 inch)
CIRCLE_DIAMETER_INCHES = 1.0

# Load image
image = cv2.imread('1.jpg')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circle_contour = None
rect_contour = None
max_circle_area = 0
max_rect_area = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:  # Skip small contours
        continue
    
    # Circle detection
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    if circle_area > 0:
        circularity = area / circle_area
        if 0.85 <= circularity <= 1.1 and area > max_circle_area:
            max_circle_area = area
            circle_contour = contour
            circle_diameter_px = 2 * radius
    
    # Rectangle detection
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        if area > max_rect_area:
            max_rect_area = area
            rect_contour = approx

if circle_contour is None or rect_contour is None:
    raise ValueError("Reference circle or rectangle not found")

# Pixel-to-inch conversion
pixels_per_inch = circle_diameter_px / CIRCLE_DIAMETER_INCHES

# Extract rectangle vertices
rect_points = rect_contour.reshape(4, 2)

# Draw shapes
cv2.drawContours(image, [circle_contour], -1, (0, 255, 0), 2)
cv2.drawContours(image, [rect_contour], -1, (0, 0, 255), 2)

# Calculate rectangle center
rect_center = np.mean(rect_points, axis=0)

# Calculate and annotate side lengths just outside the rectangle
for i in range(4):
    pt1 = rect_points[i]
    pt2 = rect_points[(i + 1) % 4]
    
    # Calculate side length (inches)
    side_length_px = np.linalg.norm(pt1 - pt2)
    side_length_in = side_length_px / pixels_per_inch
    
    # Midpoint of the side
    midpoint = ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)
    
    # Vector from rectangle center to midpoint
    center_to_mid = np.array([midpoint[0] - rect_center[0], midpoint[1] - rect_center[1]])
    
    # Normalize vector and extend beyond midpoint
    norm = np.linalg.norm(center_to_mid)
    if norm > 0:
        direction = center_to_mid / norm
        # Position text 25 pixels outside the rectangle
        text_pos = (int(midpoint[0] + direction[0] * 25), 
                    int(midpoint[1] + direction[1] * 25))
        
        # Prepare text and get size
        text = f"{side_length_in:.2f}\""
        (text_width, text_height), baseline = cv2.getTextSize(text, 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                                            0.7, 2)
        
        # Draw white background rectangle
        bg_top_left = (text_pos[0] - text_width//2 - 5, 
                       text_pos[1] - text_height//2 - 5)
        bg_bottom_right = (text_pos[0] + text_width//2 + 5, 
                           text_pos[1] + text_height//2 + 5)
        cv2.rectangle(image, bg_top_left, bg_bottom_right, 
                     (255, 255, 255), -1)
        
        # Draw text
        text_org = (text_pos[0] - text_width//2, 
                    text_pos[1] + text_height//2)
        cv2.putText(image, text, text_org, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Display result
cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save result
cv2.imwrite("Rectangle Dimensions.jpg", image)