import cv2
import numpy as np

# Create a blank black image
image = np.zeros((400, 400), dtype=np.uint8)

# Create some shapes or areas on the image
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)  # White rectangle
cv2.circle(image, (300, 200), 50, 255, -1)  # White circle

cv2.imshow("Original Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Find contours
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a new blank image
output_image = np.zeros_like(image)  # Blank canvas
cv2.drawContours(output_image, contours, -1, 255, thickness=cv2.FILLED)  # Draw all contours in white with thickness 2

# Display the result
cv2.imshow("Contours", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()