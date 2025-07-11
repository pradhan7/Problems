import cv2
import numpy as np
import math

def circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    area = cv2.contourArea(contour)
    return (4 * math.pi * area) / (perimeter * perimeter)

def pack_circles_in_rectangle(width, height, radius):
    if width < 2 * radius or height < 2 * radius:
        return []
    row_spacing = math.sqrt(3) * radius
    centers = []
    j = 0
    while True:
        y = radius + j * row_spacing
        if y + radius > height:
            break
        offset = (j % 2) * radius
        x = radius + offset
        while x + radius <= width:
            centers.append((x, y))
            x += 2 * radius
        j += 1
    return centers

img = cv2.imread("2.jpg")
if img is None:
    print("Error: Image file not found.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
inverted = cv2.bitwise_not(gray)
_, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = hierarchy[0]

max_area = 0
rect_contour_idx = -1
inner_contour_idx = -1
for i, cnt in enumerate(contours):
    if hierarchy[i][2] != -1:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            rect_contour_idx = i
            inner_contour_idx = hierarchy[i][2]

if rect_contour_idx == -1 or inner_contour_idx == -1:
    print("Rectangle not detected.")
    exit()

x, y, w, h = cv2.boundingRect(contours[inner_contour_idx])

circle_contour_idx = -1
max_circularity_val = 0
for i, cnt in enumerate(contours):
    if i == rect_contour_idx or i == inner_contour_idx:
        continue
    if len(cnt) < 5:
        continue
    circ = circularity(cnt)
    if circ > max_circularity_val and 0.8 < circ < 1.2:
        max_circularity_val = circ
        circle_contour_idx = i

if circle_contour_idx == -1:
    print("Circle not detected.")
    exit()

(center_x, center_y), radius = cv2.minEnclosingCircle(contours[circle_contour_idx])
radius = int(radius)

centers = pack_circles_in_rectangle(w, h, radius)
circle_count = len(centers)
print(f"Number of complete circles: {circle_count}")

output_img = img.copy()
for cx, cy in centers:
    img_cx = int(x + cx)
    img_cy = int(y + cy)
    cv2.circle(output_img, (img_cx, img_cy), radius, (0, 0, 255), 2)

cv2.imshow("Packed Circles", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output_2.jpg", output_img)