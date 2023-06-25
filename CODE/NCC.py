import cv2
import numpy as np

def ncc(img1, img2):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    num = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    den = np.sqrt(np.sum((img1 - np.mean(img1))**2)) * np.sqrt(np.sum((img2 - np.mean(img2))**2))
    return num / den



# Example usage
img1 = cv2.imread('D:\PROJECTS\standard_test_images\house.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('D:\PROJECTS\Python Pro\Attacks\House_MF3.tif', cv2.IMREAD_GRAYSCALE)

# Calculate NCC
ncc_val = ncc(img1, img2)
print("NCC value:", ncc_val)

