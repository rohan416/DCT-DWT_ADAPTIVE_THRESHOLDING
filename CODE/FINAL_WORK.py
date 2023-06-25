import cv2
import numpy as np
import pywt


def embed_watermark(host_image_path, watermark_path):
    # Load host image and watermark
    host_image = cv2.imread('D:\PROJECTS\standard_test_images\house.tif', cv2.IMREAD_GRAYSCALE)
    watermark = cv2.imread('D:\PROJECTS\makor\LOGO_2.jpg', cv2.IMREAD_GRAYSCALE)
    watermark1 = cv2.resize(watermark, (host_image.shape[1], host_image.shape[0]))

    # Apply 3-level DWT on watermark
    coeffs = pywt.wavedec2(watermark1, 'haar', level=3)

    # Divide the host image into 8x8 blocks
    blocks = [host_image[i:i+8, j:j+8] for i in range(0, host_image.shape[0], 8) for j in range(0, host_image.shape[1], 8)]

    # Calculate threshold as a fraction of the average standard deviation of wavelet coefficients
    avg_std = np.mean([np.std(pywt.wavedec2(block, 'haar', level=3)[0]) for block in blocks])
    alpha = 0.05 
    # adjust the fraction as needed
                                                                                                                                                                    
    # Apply DCT to each block
    dct_blocks = [cv2.dct(np.float32(block)) for block in blocks]
    vr = alpha* 0.001
    # Embed watermark in DCT coefficients using adaptive thresholding
    for i in range(len(blocks)):
        block = dct_blocks[i]
        row = i // host_image.shape[1]
        col = i % host_image.shape[1]
        threshold = (alpha*vr) * avg_std
        if abs(block[0, 0]) > threshold: 
            watermark_block = coeffs[0][row//8][col//8]
            block[0, 0] = (1 + (alpha*vr) * watermark_block) * block[0, 0]
            dct_blocks[i] = block

    # Apply inverse DCT to get watermarked image
    watermarked_image = np.zeros_like(host_image)
    k = 0
    for i in range(0, host_image.shape[0], 8):
        for j in range(0, host_image.shape[1], 8):
            block = cv2.idct(dct_blocks[k])
            watermarked_image[i:i+8, j:j+8] = block
            k += 1
            

    cv2.imshow("Watermarked Image", watermarked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('D:\PROJECTS\HOUSE_F_watermarked.tif', watermarked_image)

    # Apply inverse DWT to get original watermark
    watermark_reconstructed = pywt.waverec2(coeffs, 'haar')
    dct_blocks_orig = [np.zeros_like(block) for block in dct_blocks]
    k = 0
    for i in range(0, host_image.shape[0], 8):
        for j in range(0, host_image.shape[1], 8):
            dct_blocks_orig[k] = cv2.dct(np.float32(host_image[i:i+8, j:j+8]))
            k += 1

    # Apply inverse DCT to get original host image
    host_image_reconstructed = np.zeros_like(host_image)
    k = 0
    for i in range(0, host_image.shape[0], 8):
        for j in range(0, host_image.shape[1], 8):
            block = cv2.idct(dct_blocks_orig[k])
            host_image_reconstructed[i:i+8, j:j+8] = block
            k += 1

    watermark_reconstructed = cv2.resize(watermark_reconstructed, (watermark.shape[1], watermark.shape[0]))

    
    cv2.imshow("extracted cover Image", host_image_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('D:\PROJECTS\HOUSE_F_cover_ex.tif', host_image_reconstructed)

    cv2.imshow("Extracted Watermark", watermark_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('D:\PROJECTS\HOUSE_F_wm_extracted.jpg', watermark_reconstructed)

    return watermarked_image, host_image_reconstructed,  watermark_reconstructed


image = embed_watermark('D:\PROJECTS\standard_test_images\house.tif','D:\PROJECTS\makor\LOGO_2.jpg')

