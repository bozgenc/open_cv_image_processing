import cv2 as cv
import numpy as np
import os

def main():
    #descriptor = 'SIFT' # hangi descriptor'ı kullanmak istiyorsanız onu yorumdan çıkarın.
    descriptor = 'SURF'
    #descriptor = 'ORB'

    #matcher = "BruteForce" # Flann based matcher için bu satırı yoruma alıp alttaki satırı yorumdan çıkarın.
    matcher = 'FLANN'

    img1 = cv.imread('uni_test_1.jpg')
    img2 = cv.imread('uni_test_2.jpg')

    img1_width = img1.shape[1]
    img1_height = img1.shape[0]
    img2_width = img2.shape[1]
    img2_height = img2.shape[0]

    new_width = (img1_width + img2_width // 2) # oluşacak panorama image için width ve height belirlenir
    new_height = max(img1_height, img2_height)

    print("Stitching started.\n")
    print("--------------------------------")
    (keypoints1, features1) = find_keypoints(img1, descriptor)  # her iki image için keypoint ve feature extraction yapılır
    (keypoints2, features2) = find_keypoints(img2, descriptor)
    
    print("Keypoints A extracted")
    print("Features A extracted")
    print("--------------------------------")
    print("Keypoints B extracted")
    print("Features B extracted")
    print("--------------------------------")

    homography_matrix = findMatches(keypoints1, keypoints2, features1, features2, matcher)  # keypoint ve featurelar ile matching yapılır
    if(homography_matrix is None):
        print("Not enought keypoint matched.")
        os.exit(1)
    print("Homography Matrix Created")
    result = cv.warpPerspective(img2, homography_matrix,(new_width, new_height))    # homography matrix kullanılarak warp perspective yapılır
    result[0:img1_height, 0:img1_width] = img1                                      # ve panorama image oluşur

    print("Stitching completed.")
    print("--------------------------------\n")

    cv.imshow("Result", result)
    cv.waitKey(0)


def find_keypoints(img, descriptor_arg):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # gelen parametreye göre SIFT SURF ya da ORB oluşturulur
    if descriptor_arg == 'SIFT':
        print("Selected descriptor is SIFT")
        descriptor = cv.SIFT_create()
    elif descriptor_arg == 'SURF':
        print("Selected descriptor is SURF")
        descriptor = cv.xfeatures2d.SURF_create()
    elif descriptor_arg == 'ORB':
        print("Selected descriptor is ORB")
        descriptor = cv.ORB_create()
    
    print("--------------------------------")

    (keypoints, features) = descriptor.detectAndCompute(img, None)  # daha sonra descriptor çalışarak keypoint ve feature extract eder
    keypoints_np = np.float32([keypoints.pt for keypoints in keypoints]) # keypointleri openCV objesinden numpy arrayine çeviririz
    return (keypoints_np, features)


def findMatches(keypoints2, keypoints1, features2, features1, matcher_arg): # BruteForce ya da FLANN based matcher oluşturulur
    if matcher_arg == "BruteForce":
        print("Descriptor matcher is BruteForce")
        matcher = cv.DescriptorMatcher_create("BruteForce")
        knn_matches = matcher.knnMatch(features1, features2, 2)
    if matcher_arg == "FLANN": 
        print("Descriptor matcher is FLANN based")
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(np.float32(features1), np.float32(features2), 2) # matcher FLANN ise, featurelar float olmak zorunda
    
    print("--------------------------------")

    ratio_threshold = 0.75  # ratio threshold ile good pointler belirlenir. Lowe's Test
    good_points = []
    for x in knn_matches:
        if len(x) == 2 and x[0].distance < ratio_threshold * x[1].distance:
            good_points.append((x[0].trainIdx, x[0].queryIdx))

    if len(good_points) > 4:    # Eğer 4 ten fazla match varsa homography hesapla, yoksa o zaman yeteri kadar matching keypoint yok demek
        points1 = np.float32([keypoints1[i] for (_, i) in good_points])
        points2 = np.float32([keypoints2[i] for (i, _) in good_points])

        (H, status) = cv.findHomography(points1, points2, cv.RANSAC, 4.0)   # ransac ile outlier elimination yapılır, reProj threshold 4.0 seçtim
        return H    # return homography matrix
    else:
        return None




if __name__ == '__main__':
    main()