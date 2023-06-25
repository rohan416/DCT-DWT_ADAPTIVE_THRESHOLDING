import cv2
import numpy as np

# Load the original image and the watermarked image
original_img = cv2.imread("D:\PROJECTS\standard_test_images\house.tif", cv2.IMREAD_GRAYSCALE)
watermarked_img = cv2.imread("D:\PROJECTS\HOUSE_F_watermarked.tif", cv2.IMREAD_GRAYSCALE)

# Calculate the XOR between the original image and the watermarked image
xor_img = cv2.bitwise_xor(original_img, watermarked_img)

# Calculate the total number of bits in the watermark
watermark_size = watermarked_img.shape[0] * watermarked_img.shape[1] * 8

# Calculate the number of bit errors in the watermark
ber = np.count_nonzero(xor_img) / watermark_size


print("Bit Error Rate (BER): {:.2%}".format(ber))
