import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import warnings

# Sonucu vermesi iki sample image için ortalama 5-6 dakika sürüyor 

def main():
    warnings.filterwarnings("ignore")
    img = cv.imread("noisyImage_Gaussian.jpg", 0)
    img_2 = cv.imread("noisyImage_Gaussian_01.jpg", 0)
    print("Non Local Means Started\n")

    smoothed_img = non_local_means(img, 17, 5, 10)
    smoothed_img_cv = cv.fastNlMeansDenoising(img,h=10, templateWindowSize=5, searchWindowSize=17)
    gaussian_smoothed = cv.GaussianBlur(img, (5,5), 0)

    print("Non Local Means Started For Second Image \n")
    smoothed_img_2 = non_local_means(img_2, 17, 5, 10)
    smoothed_img_cv_2 = cv.fastNlMeansDenoising(img_2,h=10, templateWindowSize=5, searchWindowSize=17)
    gaussian_smoothed_2 = cv.GaussianBlur(img_2, (5,5), 0)

    images = []
    images = [smoothed_img, smoothed_img_cv, gaussian_smoothed, smoothed_img_2, smoothed_img_cv_2, gaussian_smoothed_2]

    print("Finished")
    show_outputs(images)


def show_outputs(images):
    plt.rcParams["figure.figsize"] = [15.00, 7.50]
    plt.rcParams["figure.autolayout"] = True
    rows = 3
    columns = 3

    for i in range(len(images)):
        plt.subplot(rows, columns, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        plt.title(getTitle(i))
    plt.show()


def getTitle(i):
    if i == 0 or i == 3:
        return "My Non Local Means"
    if i == 1 or i == 4:
        return "Non Local Means CV"
    if i == 2 or i == 5:
        return "Gaussian"


def non_local_means(img, search_window_size, kernel_size, h):
    padding_size = kernel_size // 2     # padding yapılır 
    search_offset = search_window_size // 2 # big window size'ın yarısı kadar bir offsete ihtiyaç var
    padded_img = np.pad(img, padding_size, "reflect")   # mirror padding yapılır

    output_img = np.zeros(padded_img.shape, 'uint8')
    h += 2
    progress = 10
    for x in range(padding_size, padded_img.shape[1] - padding_size):  
        for y in range(padding_size, padded_img.shape[0] - padding_size):
            if x >= 50 and x % 50 == 0 and y == padding_size:
                print_status(progress)
                progress += 10                                              # her iterasyonda smaill windowu oluşturuyoruz
            small_window = padded_img[x - padding_size: x + padding_size + 1, y - padding_size: y + padding_size + 1]
            s_window_left = 0 
            s_window_right = 0
            s_window_upper = 0 
            s_window_lower = 0

            if x - search_offset < 0:   # search window paddinglerin dışına taşıyorsa o bölgeleri ignore ediyoruz
                s_window_left = 0
            else: 
                s_window_left = x - search_offset
            if x + search_offset > padded_img.shape[1]:
                s_window_right = padded_img.shape[1]
            else:
                s_window_right = x + search_offset
            if y - search_offset < 0:
                s_window_upper = 0
            else:
                s_window_upper = y - search_offset
            if y + search_offset > padded_img.shape[0]:
                s_window_lower = padded_img.shape[0]
            else:
                s_window_lower = y + search_offset

            total_weight = 0
            intensity = 0
        
            for i in range(s_window_left, s_window_right - kernel_size): # büyük search window içinde
                for j in range(s_window_upper, s_window_lower - kernel_size):   # patchler alıp small window ile karşılaştırıp
                    small_nbhd = padded_img[i:i + kernel_size, j:j+kernel_size] # weightlerini hesaplayıp
                    euclidean_dist = np.sqrt(np.sum(np.square(small_nbhd - small_window))) 
                    weight = np.exp(-euclidean_dist/h)
                    total_weight += weight
                    intensity += weight * padded_img[i][j]
                
            intensity /= total_weight
            output_img[x-padding_size][y-padding_size] = intensity # yeni pixel değerini oluşturuyoruz


    return output_img[:output_img.shape[1]-kernel_size,:output_img.shape[0]-kernel_size]    # ekstra paddingleri çıkarıp outputu dönüyoruz


def print_status(status):   # ilerleme durumunu konsolda göstermek için progress bar ekledim
    print("%",status)
    progressBar = list("----------")
    for i in range(status//10):
        progressBar[i] = '#'
    print("[ ", end='')
    print("".join(progressBar), "]")
    if status == 100:
        print("Finishing up. Wait a couple of seconds...")
    print("\n")


if __name__ == '__main__':
    main()
