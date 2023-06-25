import cv2
import numpy as np

# Load the input image
img = cv2.imread('D:\PROJECTS\HOUSE_F_watermarked.tif', cv2.IMREAD_GRAYSCALE)

# Apply the Gaussian low-pass filter
kernel_size = (3, 3)
sigma = 1
filtered_img = cv2.GaussianBlur(img, kernel_size, sigma)

# Display the original and filtered images
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("D:\PROJECTS\Python Pro\Attacks\House_blur_5.tif", filtered_img)
