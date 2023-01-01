import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

flag = True # soru 1 ile ilgili outputları görmek için True kalsın, 
            # soru 2 ile ilgili outputları görmek için False yapın
lena = cv.imread("test2.jpg")
output_1 = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)

height = output_1.shape[0]
width = output_1.shape[1]

pixels = [] # created for representing intensity levels 
for x in range(256): 
    pixels.append(0)

for i in range(width):  # scanning through image and incrementing corresponding index in pixels array
    for j in range(height):
        pixel_density = output_1[i][j]
        pixels[pixel_density] += 1

for i in range(256):    # normalizing 
    pixels[i] = pixels[i] / (height * width)


mapped_pixels = [] # equalizing
for i in range(256):
    j = 0
    total_intensities = 0
    for j in range(i+1):
        total_intensities += pixels[j]
    mapped_pixels.append(round(255 * total_intensities))


for i in range(width):  # replacing the pixels intensity values with the new ones
    for j in range(height):
        output_1[i][j] = mapped_pixels[output_1[i][j]]

# ------------------- opencv ------------------- # 

output_2 = cv.cvtColor(lena, cv.COLOR_BGR2GRAY)
output_2 = cv.equalizeHist(output_2)

# ------------------- comparing histogram methods ------------------- # 

difference = abs(output_2 - output_1)
total = 0
for i in range(width):
    for j in range(height):
        total += difference[i][j]

print("Question 1 absolute difference: ", total)

# ------------------- question 2 ------------------- # 

lena_2_original = cv.imread("test2.jpg")
lena_2 = cv.cvtColor(lena_2_original, cv.COLOR_BGR2GRAY)

histogram = []
for i in range(256):
    histogram.append(0)

for i in range(width):  # compute histogram 
    for j in range(height):
        histogram[lena_2[i][j]] += 1

minVal = 65537
gMin = -1
for i in range(len(histogram)): # detect the lowest occuring gray level intensity in the picture
    if histogram[i] < minVal and histogram[i] != 0:
        minVal = histogram[i]
        gMin = i
      
cumulative_histogram = []   
for i in range(256):
    cumulative_histogram.append(0)

for i in range(len(histogram)): # compute cumulative histogram
    if i == 0:
        cumulative_histogram[i] = histogram[i]
    else:
        cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i] 

hMin = cumulative_histogram[gMin]   # detect the hMin

finalHistogram = []
for i in range(256):
    finalHistogram.append(0)

for i in range(len(cumulative_histogram)):  # map pixels to new intenstiy levels with the help of given formula
    div = cumulative_histogram[i] - hMin 
    div_by = (width * height) - hMin
    
    ans = round((div/div_by) * 255)
    finalHistogram[i] = ans

for i in range(width):  # change the original input 
    for j in range(height):
        lena_2[i][j] = finalHistogram[lena_2[i][j]]

# ------------------- comparing histogram methods ------------------- # 

difference_2 = abs(output_2 - lena_2)
total_2 = 0
for i in range(width):
    for j in range(height):
        total_2 += difference_2[i][j]

print("Question 2 absolute difference: ", total_2)

# ------------------- outputs  ------------------- # 

if flag: # soru 1 in çıktıları
    cv.imshow("Q1 Lena Original", lena)
    cv.imshow("Q1 Open CV Equalization", output_2)
    cv.imshow("Q1 Lena after My Equalization", output_1)
    cv.imshow("Q1 Absolute Difference", difference)
else:   # soru 2 nin çıktıları
    cv.imshow("Q2 Lena 2 Original", lena_2_original)
    cv.imshow("Q2 Lena 2 After My Equalization", lena_2) 
    cv.imshow("Q2 Open CV Equalization", output_2)
    cv.imshow("Q2 Absolute Difference", difference_2)

cv.waitKey(0)
