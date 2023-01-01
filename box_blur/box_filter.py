from math import ceil
import cv2 as cv
import numpy as np
from pyparsing import col

filepath = "lena_grayscale_hq.jpg" 
img = cv.imread(filepath ,0)
cv.imshow("Original Input", img)
show_q1_answers = False # bu değişkeni false yapınca program ikinci sorunun cevaplarını gösterir, true kalırsa ilk sorunun cevaplarını basar

def blur(img, size: int):
    if size % 2 == 0: 
        print("Kernel filter dimensions must be an odd number!")
        return img

     # input görselin dimensionlarını al
    width = img.shape[1]   
    height = img.shape[0]

    # kernel ile raster scan yaparken taşma olmaması için image'ı büyütüyoruz
    # örneğin 3x3 kernel için image'ın üstüne altına tek row 0 sağına ve soluna tek column 0-padding yapmamız lazım 
    # ya da örneğin 7x7 kernel için ise her kenara 3 sıra 0 padding yapılmalı
    # yani genel formül (box size - 1) / 2 kadar kenarlara padding yapılmasıdır
    # bu sebeple orijinal image boyutu + padding boyutu büyüklüğünde siyah boş bir görsel oluşturulur
    padding_size = (size - 1)  
    input_padded = np.zeros((width + padding_size, height + padding_size), dtype=np.uint8)  
        
    p_width = input_padded.shape[1]
    p_height = input_padded.shape[0]

    # daha sonra yeni oluşturduğum paddingli görselin ortasına orijinal image yerleştirilir
    # bu şekilde ilerleyince filter kernel image üzerinde dolaştığında taşma olmayacak ve tüm pikseller işlenecektir
    rowStart = padding_size // 2
    rowEnd = p_height - (padding_size // 2)
    columnStart = padding_size // 2
    columnEnd = p_width - (padding_size // 2)

    for x in range(rowStart, rowEnd):
        for y in range(columnStart, columnEnd):
            input_padded[x][y] = img[x - padding_size//2][y- padding_size//2]
        

    # son olarak orijinal input image boyutlarında bir output image oluşturulur
    # paddingli image üzerinden filtre uygulanır ve değerler output image üzerine yazılır
    blurred_img = np.zeros((width,height), dtype=np.uint8)
    for i in range(p_width-size + 1):
        for j in range(p_height-size + 1):
            sumPix = 0
            for k in range (i, i + size, 1):
                for p in range (j, j + size, 1):
                    sumPix += input_padded[k][p]
            
            new_intensity = round(sumPix / (size * size))
            blurred_img[i][j] = new_intensity

    return blurred_img

    
# ---------- my method ---------- # 
if(show_q1_answers):
    output_1_1 = blur(img, 3)
    output_1_2 = blur(img, 11)
    output_1_3 = blur(img, 21)

# ---------- open cv ---------- # 

output_2_1 = cv.blur(img, (3,3), borderType=cv.BORDER_CONSTANT)
output_2_2 = cv.blur(img, (11,11), borderType=cv.BORDER_CONSTANT)
output_2_3 = cv.blur(img, (21,21), borderType=cv.BORDER_CONSTANT)

# ---------- comparing open cv with my solution ---------- # 

if(show_q1_answers):
    abs1 = abs(output_1_1 - output_2_1)
    abs2 = abs(output_1_2 - output_2_2)
    abs3 = abs(output_1_3 - output_2_3)

    max_1 = -1
    for i in range(output_1_1.shape[1]):
        for j in range(output_1_1.shape[0]):
            if abs1[i][j] > max_1:
                max_1 = abs1[i][j]
    print("Maximum Difference 3x3 Box Filter vs OpenCV: ", max_1)

    max_2 = -1
    for i in range(output_1_2.shape[1]):
        for j in range(output_1_2.shape[0]):
            if abs2[i][j] > max_2:
                max_2 = abs2[i][j]
    print("Maximum Difference 3x3 Box Filter vs OpenCV:", max_2)

    max_3 = 0
    for i in range(output_1_3.shape[1]):
        for j in range(output_1_3.shape[0]):
            if abs3[i][j] > max_3:
                max_3 = abs3[i][j]
    print("Maximum Difference 3x3 Box Filter vs OpenCV:", max_3)

# ---------- outputs ---------- # 
if(show_q1_answers):
    cv.imshow("3x3 My Box Filter Applied", output_1_1) 
    cv.imshow("11x11 My Box Filter Applied", output_1_2)
    cv.imshow("21x21 My Box Filter Applied", output_1_3)

    cv.imshow("3x3 Open Cv Box Filter Applied", output_2_1)
    cv.imshow("11x11 Open Cv Box Filter Applied", output_2_2)
    cv.imshow("21x21 Open Cv Box Filter Applied", output_2_3)

    cv.imshow("3x3 Absolute Difference", abs1)
    cv.imshow("11x11 Absolute Difference", abs2)
    cv.imshow("21x21 Absolute Difference", abs3)

    cv.waitKey(0)

# ---------- Question 2 ---------- # 

img_2 = cv.imread(filepath,0)

# mantık ilk soru ile aslında aynı 
# burada yapılan önce image boyutlarının sağına ve soluna padding yapmak
# sonra da row filter uygulamak
# daha sonra oluşan image'ın transpose unu alıp tekrar sağa sola padding yapıp tekrar row filter uygulamak
# column filter uygulamak yerine transpose alıp row filter uygulamak birebir aynı sonucu verir ama önbellek erişimi olarak
# çok daha verimli olduğu için bu yöntemi tercih ettim

def blur_seperable(img, size: int):
    if size % 2 == 0:
        print("Kernel filter dimensions must be an odd number!")
        return img
    
    width = img.shape[1]
    height = img.shape[0]

    padding_size = size - 1
    input_padded = np.zeros((width, height + padding_size), dtype=np.uint8)  
        
    p_width = input_padded.shape[1]
    p_height = input_padded.shape[0]

    rowStart = 0
    rowEnd = p_height
    columnStart = padding_size // 2
    columnEnd = p_width - (padding_size // 2)

    for x in range(rowStart, rowEnd):
        for y in range(columnStart, columnEnd):
            input_padded[x][y] = img[x][y- padding_size//2]
        

    row_filtered = np.zeros((width,height), dtype=np.uint8)
    for i in range(p_width - size + 1):
        for j in range(p_height):
            sumPix = 0
            for k in range(j, j + size, 1):
                sumPix += input_padded[i][k]
            newIntensity = (sumPix * (1 / size))
            row_filtered[i][j] = newIntensity

    column_filtered = np.transpose(row_filtered)

    padded_2 = np.zeros((column_filtered.shape[1], column_filtered.shape[0] + padding_size), dtype=np.uint8)  
    p_width_2 = padded_2.shape[1]
    p_height_2 = padded_2.shape[0]

    rowStart = 0
    rowEnd = p_height_2
    columnStart = padding_size // 2
    columnEnd = p_width_2 - (padding_size // 2)

    for x in range(rowStart, rowEnd):
        for y in range(columnStart, columnEnd):
            padded_2[x][y] = column_filtered[x][y- padding_size//2]
    
    blurred_sep = np.zeros((width,height), dtype=np.uint8)
    for i in range(p_width_2 - size + 1):
        for j in range(p_height_2):
            sumPix = 0
            for k in range(j, j + size, 1):
                sumPix += padded_2[i][k]
            newIntensity = round(sumPix * (1 / size))
            blurred_sep[i][j] = newIntensity

    return np.transpose(blurred_sep)

# ---------- applying seperable filter  ---------- # 

if not show_q1_answers:
    output_3_1 = blur_seperable(img_2, 3)
    output_3_2 = blur_seperable(img_2, 11)
    output_3_3 = blur_seperable(img_2, 21)

# ---------- absolute difference between open cv and my method  ---------- # 

if not show_q1_answers:
    abs_s_1 = abs(output_2_1 - output_3_1)
    abs_s_2 = abs(output_2_2 - output_3_2)
    abs_s_3 = abs(output_2_3 - output_3_3)

    maxIntensity = -1
    for i in range(output_2_1.shape[1]):
        for j in range(output_2_1.shape[0]):
            if abs_s_1[i][j] > maxIntensity:
                maxIntensity = abs_s_1[i][j]


    maxIntensity_2 = -1
    for i in range(output_2_2.shape[1]):
        for j in range(output_2_2.shape[0]):
            if abs_s_2[i][j] > maxIntensity_2:
                maxIntensity_2 = abs_s_2[i][j]

    maxIntensity_3 = -1
    for i in range(output_2_3.shape[1]):
        for j in range(output_2_3.shape[0]):
            if abs_s_3[i][j] > maxIntensity_3:
                maxIntensity_3 = abs_s_3[i][j]

# ---------- outputs  ---------- # 

if not show_q1_answers:
    cv.imshow("3x3 My Seperable Filter ", output_3_1)
    cv.imshow("11x11 My Seperable Filter ", output_3_2)
    cv.imshow("21x21 My Seperable Filter ", output_3_2)

    print("Max Absolute Difference for 3x3 Seperable: ", maxIntensity)
    print("Max Absolute Difference for 11x11 Seperable: ", maxIntensity_2)
    print("Max Absolute Difference for 21x21 Seperable: ", maxIntensity_3)

    cv.imshow("Difference of 3x3 Seperable Filters", abs_s_1)
    cv.imshow("Difference of 11x11 Seperable Filters", abs_s_2)
    cv.imshow("Difference of 21x21 Seperable Filters", abs_s_3)
    cv.waitKey(0)
    