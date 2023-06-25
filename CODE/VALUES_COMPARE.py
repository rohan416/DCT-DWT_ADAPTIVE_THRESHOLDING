import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim





def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Load original and watermarked images
original = cv2.imread('D:\PROJECTS\standard_test_images\lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)
watermarked = cv2.imread('D:\PROJECTS\GUS_LENNA_F_watermarked.tif', cv2.IMREAD_GRAYSCALE)

# Calculate PSNR value
psnr_value = psnr(original, watermarked)

print(f"PSNR value: {psnr_value} dB")

def ncc(img1, img2):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    num = np.sum((img1 - np.mean(img1)) * (img2 - np.mean(img2)))
    den = np.sqrt(np.sum((img1 - np.mean(img1))**2)) * np.sqrt(np.sum((img2 - np.mean(img2))**2))
    return num / den

def beer(img1, img2, block_size):
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    h, w = img1.shape[:2]
    num_blocks = (h // block_size) * (w // block_size)
    errors = np.zeros(num_blocks)
    k = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block1 = img1[i:i+block_size, j:j+block_size]
            block2 = img2[i:i+block_size, j:j+block_size]
            errors[k] = np.sum((block1 - block2)**2)
            k += 1
    return np.mean(errors) 

def calculate_mse(img1, img2):
    # calculate the squared difference between the images
    diff = (img1.astype("float") - img2.astype("float")) ** 2

    # calculate the mean squared error
    mse = np.mean(diff)

    return mse

def calculate_ssim(img1, img2):
    # calculate the SSIM value
    ssim_value= ssim(img1, img2, multichannel=False)

    return ssim_value



# Example usage
img1 = cv2.imread('D:\PROJECTS\standard_test_images\lena_gray_512.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('D:\PROJECTS\GUS_LENNA_F_watermarked.tif', cv2.IMREAD_GRAYSCALE)

# Calculate NCC
ncc_val = ncc(img1, img2)
print("NCC value:", ncc_val)

# Calculate BEER
beer_val = beer(img1, img2, block_size=8)
print("BER value:", beer_val)

mse_val = calculate_mse(img1,img2)
print("MSE vaue is :", mse_val)

ssim_val = calculate_ssim(img1,img2)
print("SSIM value is :", ssim_val)


