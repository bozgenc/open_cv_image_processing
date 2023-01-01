import cv2 as cv
import numpy as np
import statistics as s


def main():
    q1 = True # False yapınca ikinci sorunun çıktılarını verir

    if q1:
        img = cv.imread('noisyImage_gaussian.jpg', 0)
        clean_img = cv.imread('lena_grayscale_hq.jpg', 0)
        clean_img = cv.normalize(clean_img, None, alpha=0,beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        img = cv.normalize(img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

        output_1_1 = adaptive_mean_filter(img, 5)
        output_1_2 = cv.blur(img, (5, 5), borderType=cv.BORDER_CONSTANT)
        output_1_3 = cv.GaussianBlur(img, (5, 5), 0, 0, cv.BORDER_CONSTANT)

        psnr1 = cv.PSNR(output_1_1, clean_img)
        psnr2 = cv.PSNR(output_1_2, clean_img)
        psnr3 = cv.PSNR(output_1_3, clean_img)

        print("\nQuestion1\n---------------------------")
        print("PSNR Values: ")
        print("Adaptive Mean Filter: ", psnr1)
        print("Box Blur: ", psnr2)
        print("Gaussian Blur: ", psnr3)
        print("---------------------------")

        output_1_1 = cv.normalize(output_1_1, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        cv.imshow("Gaussian Noised", img)
        cv.imshow("Denoised With Adaptive Mean Filter", output_1_1)
        cv.imshow("Denoised With Box Filter", output_1_2)
        cv.imshow("Denoised With Gaussian Blur", output_1_3)

    # -------------------------- q2 -------------------------- |
    else:
        noisy_image = cv.imread("noisyImage_SaltPepper.jpg", 0)
        clean_img_lena = cv.imread('lena_grayscale_hq.jpg', 0)

        output_2_1 = adaptive_median_filter(noisy_image)

        output_2_2 = cv.medianBlur(noisy_image, 3)
        output_2_3 = cv.medianBlur(noisy_image, 5)
        output_2_4 = cv.medianBlur(noisy_image, 7)

        output_2_5 = weighted_median(noisy_image, 3, 3)
        output_2_6 = weighted_median(noisy_image, 5, 5)
        output_2_7 = weighted_median(noisy_image, 7, 7)

        psnr2_1 = cv.PSNR(output_2_1, clean_img_lena)
        psnr2_2 = cv.PSNR(output_2_2, clean_img_lena)
        psnr2_3 = cv.PSNR(output_2_3, clean_img_lena)
        psnr2_4 = cv.PSNR(output_2_4, clean_img_lena)
        psnr2_5 = cv.PSNR(output_2_5, clean_img_lena)
        psnr2_6 = cv.PSNR(output_2_6, clean_img_lena)
        psnr2_7 = cv.PSNR(output_2_7, clean_img_lena)

        print("\nQuestion2\n---------------------------")
        print("PSNR Values: ")
        print("Adaptive Median: ", psnr2_1)
        print("OpenCV 3x3 Median ", psnr2_2)
        print("OpenCv 5x5 Median ", psnr2_3)
        print("OpenCv 7x7 Median ", psnr2_4)
        print("My 3x3 3 Weighted ", psnr2_5)
        print("My 5x5 5 Weighted ", psnr2_6)
        print("My 7x7 7 Weighted", psnr2_7)

        cv.imshow("Salt Pepper Noise", noisy_image)
        cv.imshow("3 weighted 3x3 Median", output_2_5)
        cv.imshow("5 weighted 5x5 Median", output_2_6)
        cv.imshow("7 weighted 7x7 Median", output_2_7)
        cv.imshow("Denoised With Adaptive Median Filter", output_2_1)
        print("Soru 2 nin çıktılarını görmek için koddaki ilk satırdaki değişkeni False yapın.")
        print("---------------------------")
    cv.waitKey(0)


def adaptive_mean_filter(img, kernel_size, variance=0.004):
    denoised_img = np.zeros(img.shape, dtype='float32')
    img = border_padding(img, kernel_size, 'zero_padding')

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


def adaptive_median_filter(img):
    width = img.shape[1]
    height = img.shape[0]
    denoised_img = np.zeros(img.shape, dtype='uint8')

    padded_3 = border_padding(img, 3, 'replicate')
    padded_5 = border_padding(img, 5, 'replicate')
    padded_7 = border_padding(img, 7, 'replicate')

    for x in range(width):
        for y in range(height):
            kernel_size = 3
            window = []
            if kernel_size == 3:    # 3x3 lük kernel size ile başlarız
                for i in range(x, x + kernel_size, 1):
                    for j in range(y, y + kernel_size, 1):
                        window.append(padded_3[i][j])   # windowa pikselleri append ederiz
                
                w_np = np.array(window)
                k_matrix = w_np.reshape(kernel_size, kernel_size)

                min_intensity = np.amin(k_matrix)   # patchin min maks ve median'ını hesaplarız
                max_intensity = np.amax(k_matrix)
                median = np.median(k_matrix)

                if min_intensity < median and median < max_intensity:   # eğer median min ve max'ın arasındaysa
                    z_xy = img[x][y]
                    if min_intensity < z_xy and z_xy < max_intensity:   # pikselin kendisi de min ve max arasındaysa
                        denoised_img[x][y] = z_xy                       # çıktı pikselin kendisi olsun
                    else:
                        denoised_img[x][y] = median                     # çıktı medyan olsun
                else:
                    kernel_size += 2                                    # kernel size artılırarak tekrar hesaplansın

            if kernel_size == 5:
                window = []
                for i in range(x, x + kernel_size, 1):
                    for j in range(y, y + kernel_size, 1):
                        window.append(padded_5[i][j])
                
                w_np = np.array(window)
                k_matrix = w_np.reshape(kernel_size, kernel_size)

                min_intensity = np.amin(k_matrix)
                max_intensity = np.amax(k_matrix)
                median = np.median(k_matrix)

                if min_intensity < median and median < max_intensity:
                    z_xy = img[x][y]
                    if min_intensity < z_xy and z_xy < max_intensity:
                        denoised_img[x][y] = z_xy
                    else:
                        denoised_img[x][y] = median
                else:
                    kernel_size += 2
            
            if kernel_size == 7:
                window = []
                for i in range(x, x + kernel_size, 1):
                    for j in range(y, y + kernel_size, 1):
                        window.append(padded_7[i][j])
                
                w_np = np.array(window)
                k_matrix = w_np.reshape(kernel_size, kernel_size)

                min_intensity = np.amin(k_matrix)
                max_intensity = np.amax(k_matrix)
                median = np.median(k_matrix)

                if min_intensity < median and median < max_intensity:
                    z_xy = img[x][y]
                    if min_intensity < z_xy and z_xy < max_intensity:
                        denoised_img[x][y] = z_xy
                    else:
                        denoised_img[x][y] = median
                else:
                    denoised_img[x][y] = median

    return denoised_img

def weighted_median(img, size:int, weight): # hw3 ten weighted median filtre kodu
    if size % 2 == 0:
        print("Kernel size must be an odd number!")
        return img
    
    org_width = img.shape[1]
    org_height = img.shape[0]

    padded_img = border_padding(img, size, 'replicate')

    p_width = padded_img.shape[1]
    p_heigth = padded_img.shape[0]

    denoised_img = np.zeros((org_width, org_height), dtype='uint8')
    for i in range(p_width - size +1 ):
            for j in range(p_heigth - size + 1):
                neighbours = []
                pointer = 0
                for x in range(i, i + size, 1):
                    for y in range(j, j + size, 1):
                        if(pointer == (((size * size) - 1) / 2 )):
                            if weight == 3:
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                            elif weight == 5:
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                            elif weight == 7:
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])
                                neighbours.append(padded_img[x][y])

                        else:
                            neighbours.append(padded_img[x][y])
                        pointer += 1
                
                neighbours.sort()
                denoised_img[i][j] = s.median(neighbours)

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
