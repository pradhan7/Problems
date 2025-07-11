import cv2
import numpy as np
import math

def order_points_clockwise(pts):
    """Order triangle vertices clockwise from top-left"""
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
    return pts[np.argsort(angles)]

# Load image
image = cv2.imread('3.jpg')
orig = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
triangle_contour = None
max_area = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area < 100:
        continue
        
    # Approximate contour as polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 3 and area > max_area:
        max_area = area
        triangle_contour = approx

if triangle_contour is None:
    raise ValueError("No triangle detected in the image")

# Extract vertices
vertices = triangle_contour.reshape(3, 2)
vertices = order_points_clockwise(vertices)

# Label vertices A, B, C
A, B, C = vertices

# Calculate side lengths
a = np.linalg.norm(B - C)  # Opposite vertex A
b = np.linalg.norm(A - C)  # Opposite vertex B
c = np.linalg.norm(A - B)  # Opposite vertex C

# Calculate angles using Law of Cosines
def calculate_angle(a, b, c):
    """Calculate angle opposite side a (in radians)"""
    cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
    return math.acos(max(min(cos_angle, 1), -1))  # Clamp for floating point errors

angle_A_rad = calculate_angle(a, b, c)
angle_B_rad = calculate_angle(b, a, c)
angle_C_rad = calculate_angle(c, a, b)

# Convert to degrees
angle_A = math.degrees(angle_A_rad)
angle_B = math.degrees(angle_B_rad)
angle_C = math.degrees(angle_C_rad)

print(f"Angles: A={angle_A:.1f}°, B={angle_B:.1f}°, C={angle_C:.1f}°")