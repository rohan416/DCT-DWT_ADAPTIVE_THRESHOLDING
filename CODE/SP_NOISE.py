import cv2
import numpy as np

# Load an image
img = cv2.imread('D:\PROJECTS\HOUSE_F_watermarked.tif')

# Add salt and pepper noise
noise_amount = 0.002
noise_mask = np.random.rand(*img.shape[:2])
img[noise_mask < noise_amount/2] = 0
img[noise_mask > 1 - noise_amount/2] = 255

# Display the noisy image
cv2.imshow('Noisy Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("D:\PROJECTS\Python Pro\Attacks\House_SPN002.tif", img)