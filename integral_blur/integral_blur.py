import cv2 as cv
import numpy as np

def main():

    # ----------------- soru 1 --------------- # 
    img = cv.imread("lena_grayscale_hq.jpg", 0)
    cv.imshow("Original image", img)
    print("Soru 1")

    img_integral_cv = cv.integral(img)
    img_my_integral = integral(img)

    integral_difference = img_integral_cv - img_my_integral
    print("Difference")
    print(integral_difference)

    print("-----------------------------")

    # ----------------- soru 2 --------------- # 
  
    blurred_integral = integral_blur(img, img_my_integral, 3)
    cv.imshow("Integral Blurred", blurred_integral)

    blurred_cv = cv.blur(img, (3,3), borderType=cv.BORDER_CONSTANT)
    cv.imshow("OpenCV Blurred", blurred_cv)

    cv.waitKey(0)


def integral(img):  # soru 1 için integral alma metodu
    width = img.shape[1]
    height = img.shape[0]

    integral_img = np.zeros((width, height), dtype='int64')

    integral_img[0][0] = img[0][0] 
    for i in range(1,width):
        integral_img[0][i] = integral_img[0][i -1] + img[0][i]

    for i in range(1, height):
        integral_img[i][0] = integral_img[i -1][0] + img[i][0]

    for i in range(1,width):
        for j in range(1,height):
            integral_img[i][j] = integral_img[i -1][j -1] + leftovers(img, i,j)

    row = []      
    for x in range(width):
        row.append(0)
    
    column = []
    for y in range(height + 1):
        column.append(0)

    integral_img = np.r_[[row], integral_img]
    integral_img = np.c_[np.array(column), integral_img]

    return integral_img

def leftovers(img, x:int,y:int):    # soru 1 için ek metod
    leftover_values = 0
    for i in range(0, y +1):
        leftover_values += img[x][i]
    
    for j in range(0, x):
        leftover_values += img[j][y]

    return leftover_values


def integral_blur(img, integral, w:int): # soru 2 için integral ile box filter metodu
    width = img.shape[1]
    height = img.shape[0]
    blurred = np.zeros((width, height), dtype='uint8')
    integral = integral[1:,1:]
    
    for x in range(width-w):
        for y in range(height-w):
            topLeftX = x 
            topLeftY = y
            topRightX = x 
            topRightY =  y + w 
            bottomLeftX = x + w 
            bottomLeftY = y
            bottomRightX = x + w
            bottomRightY = y + w 

            blurred[x][y] = (integral[topLeftX][topLeftY] + integral[bottomRightX][bottomRightY] - integral[topRightX][topRightY] - integral[bottomLeftX][bottomLeftY]) / (w*w)

    return blurred



if __name__ == '__main__':
    main()
