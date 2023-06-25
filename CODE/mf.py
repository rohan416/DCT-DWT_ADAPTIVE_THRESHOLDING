import cv2

# Load the input image
img = cv2.imread("D:\PROJECTS\HOUSE_F_watermarked.tif")

# Apply median filter with kernel size of 3x3
img_median = cv2.medianBlur(img, 3)

# Display the original and median-filtered images side by side
cv2.imshow("Input Image", img)
cv2.imshow("Median Filtered Image", img_median)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('D:\PROJECTS\Python Pro\Attacks\House_MF3.tif',img_median)
