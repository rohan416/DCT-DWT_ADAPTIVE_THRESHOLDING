import cv2
import imageio
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from skimage.color import rgb2gray


# Load the original image and watermark



# Apply the watermarking algorithm to embed the watermark into the image
watermarked_img = cv2.imread('D:\PROJECTS\HOUSE_F_watermarked.tif', cv2.IMREAD_GRAYSCALE)

# Compress the watermarked image using JPEG compression
compressed_img = imageio.imwrite('D:\PROJECTS\Python Pro\Attacks\House_JP40.jpg', watermarked_img, quality=40)





