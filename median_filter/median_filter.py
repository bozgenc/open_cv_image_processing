import cv2 as cv
import numpy as np
import statistics as s
import matplotlib.pyplot as plt

img = cv.imread("noisyImage.jpg", 0)
cv.imshow("Original Noisy Image", img)

def border_padding(img, size: int):
    width = img.shape[1]
    height = img.shape[0]
    padding_size = size - 1 
    
    # --- original image üzerinden first row ve last rowları kopyalarız --- #
    first_row = []
    for x in range(width):
        first_row.append(img[0][x])

    last_row = []
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

    # --- row extended image üzerinden first column ve last columnları kopyalarız --- #
    first_column = []
    for x in range(n_height):
        first_column.append(row_padded[x][0])

    last_column = []
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


def median_filter(img, size: int):
    if size % 2 == 0:
        print("Kernel size must be an odd number!")
        return img
    
    org_width = img.shape[1]
    org_height = img.shape[0]

    padded_img = border_padding(img, size)

    p_width = padded_img.shape[1]
    p_heigth = padded_img.shape[0]

    # padding  yapılmış image raster scan ile dolaşışır 
    # window size komşuluğundaki pikseller sort edilip medyan bulunur 
    # ve yeni piksel değeri olarak seçilir
    denoised_img = np.zeros((org_width, org_height), dtype='uint8')
    for i in range(p_width - size +1 ):
        for j in range(p_heigth - size + 1):
            neighbours = []
            for x in range(i, i + size, 1):
                for y in range(j, j + size, 1):
                    neighbours.append(padded_img[x][y])
            
            neighbours.sort()
            denoised_img[i][j] = s.median(neighbours)

    return denoised_img


denoised_img = median_filter(img,5)
cv.imshow("Denoised with My Median Filter", denoised_img)

cv_denoised_img = cv.medianBlur(img,5)
cv.imshow("Denoised with OpenCv", cv_denoised_img)

# --- open cv ve kendi metodumun difference'ını aldım --- # 

abs_diff = abs(denoised_img - cv_denoised_img)
print("\nQuestion 1")
print("Absolute Difference Between OpenCv Median vs My Median Method")
print(abs_diff)

# --- sonra piksel farklarına bakılır, çıktının tamamı 0 ise open cv ile perfectly match demektir --- #
has_non_zero_intensity = False
for x in range(abs_diff.shape[1]):
    for y in range(abs_diff.shape[0]):
        if(abs_diff[x][y] != 0):
            has_non_zero_intensity = True

# --- non_zero_intensity False ise bütün pikseller sıfırdır, yani perfectly match olmuştur --- # 
print(has_non_zero_intensity)


# ---------- question 2 ---------- # 
# psnr değerlerini hesapladım, print harici matplotlib ile grafik üzerinde de gösterttim
print("\n-----------\nQuestion 2")
golden = cv.imread("lena_grayscale_hq.jpg", 0)
box_blurred = cv.blur(img, (5,5))
gaussian_blurred = cv.GaussianBlur(img, (7,7), 0, 0, cv.BORDER_DEFAULT)
median_applied = cv.medianBlur(img, 5)

psnr1 = cv.PSNR(golden, box_blurred)
psnr2 = cv.PSNR(golden, gaussian_blurred)
psnr3 = cv.PSNR(golden, median_applied)

print("Box blurred PSNR score: ", psnr1)
print("Gaussian blurred PSNR score: ", psnr2)
print("Median Blur PSNR score: ", psnr3)
print("-----------\n")

# ---------- question 3 ---------- # 
# ilk sorunun birebir aynısı padding yapılır sonra medyan piksel bulunur
# tek fark bu sefer center pikseli 3 kere eklenir

def weighted_median(img, size:int):
    if size % 2 == 0:
        print("Kernel size must be an odd number!")
        return img
    
    org_width = img.shape[1]
    org_height = img.shape[0]

    padded_img = border_padding(img, size)

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
                            neighbours.append(padded_img[x][y])
                            neighbours.append(padded_img[x][y])
                            neighbours.append(padded_img[x][y])
                        else:
                            neighbours.append(padded_img[x][y])
                        pointer += 1
                
                neighbours.sort()
                denoised_img[i][j] = s.median(neighbours)

    return denoised_img

print("Question 3")
weighted_denoised_img = weighted_median(img,5)
cv.imshow("Denoised with My Weighted Median", weighted_denoised_img)

psnr4 = cv.PSNR(golden, denoised_img)
psnr5 = cv.PSNR(golden, weighted_denoised_img)
print("PSNR score of my median filter: ", psnr4)
print("PSNR score of my weighted median filter: ", psnr5)

# ---------- question 4 ---------- # 
# 4.sorunun açıklaması soru4_aciklama.txt dosyasında

last = np.zeros((weighted_denoised_img.shape[1], weighted_denoised_img.shape[0]), dtype='uint8')
for x in range(weighted_denoised_img.shape[1]):
    for y in range(weighted_denoised_img.shape[0]):
        if(weighted_denoised_img[x][y] != 0):
            last[x][y] = weighted_denoised_img[x][y] - 10 

psnr6 = cv.PSNR(golden,last)
print("PSNR score of question 4: ", psnr6) 

# ---------- outputs ---------- # 

psnrs = {
    "Box Blur" : psnr1,
    "Gaussian Blur" : psnr2,
    "OpenCv Median Blur": psnr3, 
    "My Median Blur": psnr4, 
    "My Weight. Median Blur": psnr5,
    "Q4 Basic Transform": psnr6
}

numbers = list(psnrs.values())
names = list(psnrs.keys())

fig = plt.figure(figsize = (10, 5))
plt.bar(names, numbers, width=0.2)
plt.xlabel("PSNRS")
plt.ylabel("Values")
plt.title("PSNR Values of Different Blurring Methods")
plt.show()

cv.waitKey(0)