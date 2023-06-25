import cv2
import numpy as np

def add_speckle_noise(img, noise_variance=0.001):
    # Calculate the speckle noise
    noise = np.random.randn(*img.shape) * noise_variance * np.mean(img)
    
    # Add the noise to the image
    noisy_img = img + img * noise
    
    # Ensure that the pixel values are within the valid range of [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

# Load an image
img = cv2.imread('D:\PROJECTS\HOUSE_F_watermarked.tif', cv2.IMREAD_GRAYSCALE)

# Add speckle noise to the image
noisy_img = add_speckle_noise(img, noise_variance=0.001)

# Display the original and noisy images
cv2.imshow('Original Image', noisy_img)
cv2.imwrite('D:\PROJECTS\Python Pro\Attacks\House_Speckle03.tif', noisy_img)
