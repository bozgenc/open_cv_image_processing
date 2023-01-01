import cv2 as cv
import numpy as np
import warnings

def main():
    warnings.filterwarnings("ignore")
    
    img = cv.imread('noisyImage_Gaussian.jpg', 0)
    img2 = cv.imread('noisyImage_Gaussian_01.jpg', 0)
    clean_img = cv.imread('lena_grayscale_hq.jpg', 0)
    cv.imshow("Original", img)

    question1(img, clean_img)  # Her seferinde tek bir fonksiyonu çalıştırınca output daha okunabilir oluyor 
    #question2(img2, clean_img) # yorumdan çıkarıp 2 ve 3. soruyu da çalıştırabilirsiniz
    #question3(img2)
    cv.waitKey(0)


def question1(img, clean_img):  # img okunur normalize edilir opencv filtreleri uygulanır psnr hesaplanır # 
    img = cv.normalize(img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    cv_3 = cv.blur(img, (3,3), borderType=cv.BORDER_REPLICATE)
    cv_5 = cv.blur(img, (5,5), borderType=cv.BORDER_REPLICATE)
    gaussian_3 = cv.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0, borderType=cv.BORDER_REPLICATE)
    gaussian_5 = cv.GaussianBlur(img, (5,5), sigmaX=0, sigmaY=0, borderType=cv.BORDER_REPLICATE)
    adaptive_mean = adaptive_mean_filter(img, 5)
    bilateral = cv.bilateralFilter(img, 5, 3, 0.9, borderType=cv.BORDER_REPLICATE)

    cv_3 = cv.normalize(cv_3, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv_5 = cv.normalize(cv_5, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    gaussian_3 = cv.normalize(gaussian_3, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    gaussian_5 = cv.normalize(gaussian_5, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    adaptive_mean = cv.normalize(adaptive_mean, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    bilateral = cv.normalize(bilateral, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    psnr1 = cv.PSNR(cv_3, clean_img)
    psnr2 = cv.PSNR(cv_5, clean_img)
    psnr3 = cv.PSNR(gaussian_3, clean_img)
    psnr4 = cv.PSNR(gaussian_5, clean_img)
    psnr5 = cv.PSNR(adaptive_mean, clean_img)
    psnr6 = cv.PSNR(bilateral, clean_img)

    print("------------- Question 1 ------------- ")
    print("3x3 Box Filter PSNR:", psnr1)
    print("5x5 Box Filter PSNR:", psnr2)
    print("3x3 Gaussian Filter PSNR:", psnr3)
    print("5x5 Gaussian Filter PSNR:", psnr4)
    print("Adaptive Mean Filter PSNR:", psnr5)
    print("Bilateral Filter PSNR:", psnr6)
    print("-------------------------------------- ")

    cv.imshow("3x3 Box", cv_3)
    cv.imshow("5x5 Box", cv_5)
    cv.imshow("3x3 Gaussian", cv_3)
    cv.imshow("5x5 Gaussian", cv_5)
    cv.imshow("Adaptive Mean", adaptive_mean)
    cv.imshow("Bilateral Filter", bilateral)


def question2(img, clean_img): # img okunur normalize edilir opencv filtreleri uygulanır psnr hesaplanır # 
    img = cv.normalize(img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    cv_3 = cv.blur(img, (3,3), borderType=cv.BORDER_REPLICATE)
    cv_5 = cv.blur(img, (5,5), borderType=cv.BORDER_REPLICATE)
    gaussian_3 = cv.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0, borderType=cv.BORDER_REPLICATE)
    gaussian_5 = cv.GaussianBlur(img, (5,5), sigmaX=0, sigmaY=0, borderType=cv.BORDER_REPLICATE)
    adaptive_mean = adaptive_mean_filter(img, 5, variance=0.0009)
    bilateral = cv.bilateralFilter(img, 3, 0.1, 1, borderType=cv.BORDER_REPLICATE)

    cv_3 = cv.normalize(cv_3, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    cv_5 = cv.normalize(cv_5, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    gaussian_3 = cv.normalize(gaussian_3, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    gaussian_5 = cv.normalize(gaussian_5, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    adaptive_mean = cv.normalize(adaptive_mean, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    bilateral = cv.normalize(bilateral, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    psnr1 = cv.PSNR(cv_3, clean_img)
    psnr2 = cv.PSNR(cv_5, clean_img)
    psnr3 = cv.PSNR(gaussian_3, clean_img)
    psnr4 = cv.PSNR(gaussian_5, clean_img)
    psnr5 = cv.PSNR(adaptive_mean, clean_img)
    psnr6 = cv.PSNR(bilateral, clean_img)

    print("------------- Question 2 ------------- ")
    print("3x3 Box Filter PSNR:", psnr1)
    print("5x5 Box Filter PSNR:", psnr2)
    print("3x3 Gaussian Filter PSNR:", psnr3)
    print("5x5 Gaussian Filter PSNR:", psnr4)
    print("Adaptive Mean Filter PSNR:", psnr5)
    print("Bilateral Filter PSNR:", psnr6)
    print("--------------------------------------\n ")

    cv.imshow("3x3 Box", cv_3)
    cv.imshow("5x5 Box", cv_5)
    cv.imshow("3x3 Gaussian", cv_3)
    cv.imshow("5x5 Gaussian", cv_5)
    cv.imshow("Adaptive Mean", adaptive_mean)
    cv.imshow("Bilateral Filter", bilateral)


def question3(img):
    img = cv.normalize(img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    print("------------- Question 3 ------------- ")
    print("Bilateral filter is working")
    bilateral_filtered = bilateral_filter(img, 3, 0.1, 1)
    print("Bilateral filter is finished\n")

    bilateral_filtered_cv = cv.bilateralFilter(img, 3, 0.1, 1, borderType=cv.BORDER_REPLICATE)
    difference_img = cv.subtract(bilateral_filtered, bilateral_filtered_cv)
    difference_img = np.uint8(difference_img)
    print("Difference Image Of Two Bilateral Filter:")
    print(difference_img)

    bilateral_filtered = cv.normalize(bilateral_filtered, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    bilateral_filtered_cv = cv.normalize(bilateral_filtered_cv, None, alpha=0, beta=255,norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

    cv.imshow("My Bilateral Filtered", bilateral_filtered)
    cv.imshow("Bilateral Filtered OpenCV", bilateral_filtered_cv)
    cv.imshow("Difference Image", difference_img)


def bilateral_filter(img, kernel_size, sigmaColor, sigmaSpace):
    width = img.shape[1]
    height = img.shape[0]

    filtered = np.zeros((width,height), dtype='float32')
    img_padded = border_padding(img, kernel_size, borderType='replicate_padding') # image padding yapılır

    for x in range(width):
        for y in range(height):
            Wsum = 0
            intensity = 0
            for i in range(x, x + kernel_size, 1):  # kernel size kadar bi komşulukta 
                for j in range(y, y + kernel_size, 1):  # Spatial ve range değerleri hesaplanır
                    gspatial = (gaussian(np.abs(img_padded[x+kernel_size//2][y+kernel_size//2] - img_padded[i][j]), sigmaSpace)) # intensity farkı
                    grange = gaussian(distance(i, j, x+kernel_size//2, y+kernel_size//2), sigmaColor) # euclidean distance

                    w = gspatial * grange   
                    intensity += img_padded[i][j] * w
                    Wsum += w
            
            filtered[x][y] = ((intensity / Wsum))   # yeni piksel değeri assign edilir
    
    return filtered


def gaussian(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))


def distance(x1,y1,x2,y2):
    return np.sqrt(np.abs((x1-x2)**2-(y1-y2)**2))


def adaptive_mean_filter(img, kernel_size, variance=0.0042): # q1 ve q2 için önceki ödevden copy paste
    denoised_img = np.zeros(img.shape, dtype='float32')
    img = border_padding(img, kernel_size, 'replicate_padding')

    width = img.shape[1]
    height = img.shape[0]

    for x in range(width - kernel_size + 1):
        for y in range(height - kernel_size + 1):
            window = []
            for i in range(x, x + kernel_size, 1):  # window oluşturulup pikseller append edilir
                for j in range(y, y + kernel_size, 1):
                    window.append(img[i][j])

            w_np = np.array(window)
            k_matrix = w_np.reshape(kernel_size, kernel_size)

            local_variance = k_matrix.var() # patchin local varyansı ve average intensity'si hesaplanır
            avg_intensity = k_matrix.mean()

            variance_division = variance / local_variance
            rhs = img[x + kernel_size // 2][y + kernel_size // 2] - avg_intensity

            denoised_img[x][y] = img[x + kernel_size // 2][y + kernel_size // 2] - (variance_division * rhs)

    return denoised_img

    
def border_padding(img, size: int, borderType):
    width = img.shape[1]
    height = img.shape[0]
    padding_size = size - 1

    first_row = []
    last_row = []
    if (borderType == 'zero_padding'):
        first_row = []
        for x in range(width):
            first_row.append(0)

        for x in range(width):
            last_row.append(0)
    else:
        for x in range(width):
            first_row.append(img[0][x])

        for x in range(width):
            last_row.append(img[height-1][x])

    # --- padding size kadar image'ın altına ve üstüne ekleme yaparız --- #
    first_pad = img
    for x in range(0, padding_size // 2, 1):
        first_pad = np.r_[[first_row], first_pad]

    row_padded = first_pad
    for x in range(0, padding_size // 2, 1):
        row_padded = np.r_[row_padded, [last_row]]

    # ------ column padding ------- #

    n_height = row_padded.shape[0]

    first_column = []
    last_column = []
    if (borderType == 'zero_padding'):
        for x in range(n_height):
            first_column.append(0)

        for x in range(n_height):
            last_column.append(0)

    else:
        for x in range(n_height):
            first_column.append(row_padded[x][0])

        for x in range(n_height):
            last_column.append(row_padded[x][width-1])

    # --- padding size kadar image'ın soluna ve sağına ekleme yaparız --- #
    c_first_pad = row_padded
    for x in range(0, padding_size // 2, 1):
        c_first_pad = np.c_[np.array(first_column), c_first_pad]

    final_padded = c_first_pad
    for x in range(0, padding_size // 2, 1):
        final_padded = np.c_[final_padded, np.array(last_column)]

    return final_padded


if __name__ == '__main__':
    main()