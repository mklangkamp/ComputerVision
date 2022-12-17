import cv2
import numpy as np


def main():
    # Open first image and scale down so it can be shown with imshow
    img1 = cv2.imread('RGBImg1_Working.jpg', cv2.IMREAD_GRAYSCALE)
    img1_d = cv2.imread('depthImg1_Working.tiff', cv2.IMREAD_GRAYSCALE)
    
    # img1 = (img1/256).astype('uint8')

    # Open second image and scale down so it can be shown with imshow
    img2 = cv2.imread('RGBImg2_Working.jpg', cv2.IMREAD_GRAYSCALE)
    img2_d = cv2.imread('depthImg2_Working.tiff', cv2.IMREAD_GRAYSCALE)
    
 
    # # Open third image and scale down so it can be shown with imshow
    # img3 = cv2.imread('P3.jpg')
    # img3 = cv2.resize(img3,(756,1008))

    # Stitch the three images
    # stitchedImg, allowance = stitch2images(img3, img2, minMatches = 100)
    # stitchedImg, allowance = stitch2images(img2, img1, minMatches = 100)

    stitchedImg, depthStitched, allowance = stitchDepthimages(img2,img1,img2_d, img1_d, minMatches=20)
    

    # Show the stitched img
    cv2.imshow("RGB Img1", img1)
    cv2.imshow("RGB Img2", img2)
    cv2.imshow("Depth Img1", img1_d)
    cv2.imshow("Depth Img2", img2_d)

    cv2.imshow("img", stitchedImg)
    cv2.imshow("depthStitchedimg", depthStitched)
    cv2.waitKey(0)

    # Save the stitched img
    cv2.imwrite("stitchedImg.jpg", stitchedImg)

    return

# Trims any black pixels from around an image
def trimBorder(img):
    for i in range(1,640):
        j = 640 - i
        if(img[0,j] != 0):
            trimEdge = j
            print("Trim edge: " + str(j))
            break
        else:
            trimEdge = None

    if(trimEdge == None):
        return img
    else:
        imgSize = np.shape(img)
        imgNew = img[0:imgSize[0],1:j]

    return imgNew


def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

        
# Need to put right hand/lower image as left input to function
def stitch2images(img1_raw, img2_raw, minMatches = 2):
    # img1_gray = cv2.cvtColor(img1_raw,cv2.COLOR_BGR2GRAY)
    # img2_gray = cv2.cvtColor(img2_raw,cv2.COLOR_BGR2GRAY)

    img1_gray = img1_raw
    img2_gray = img2_raw

    # Create SIFT Detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find key points and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)

    print("Detected and Computed")

    # Create Brute force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1,des2,k=2)


    # Create list of matches between two pictures that are good matches with lowest variation in distances
    foundGoodMatches = False
    allowance = 0.03

    while(not foundGoodMatches):
        goodMatches = []
        for m,n in matches:
            if m.distance < allowance*n.distance:
                goodMatches.append(m)
        
        # If a sufficient number of matches has been found, use those. Otherwise, increase the allowance in variation
        if(len(goodMatches) < minMatches):
            allowance = allowance + 0.03
            print(allowance)
        else:
            foundGoodMatches = True
            print("Final Variation Allowance: " + str(allowance))

    # Set the color of the lines/circles to be drawn
    draw_params = dict(matchColor=(255,0,255),
                        singlePointColor=None,
                        flags=2)

    # # Show matches one by one
    # for match in goodMatches:
    #     goodMatchesImage = cv2.drawMatches(img1_raw,kp1,img2_raw,kp2,[match],None,**draw_params)    
    #     cv2.imshow("original_image_drawMatches.jpg", goodMatchesImage)
    #     cv2.waitKey(0)

    goodMatchesImage = cv2.drawMatches(img1_raw,kp1,img2_raw,kp2,goodMatches,None,**draw_params)
    
    cv2.imshow("original_image_drawMatches.jpg", goodMatchesImage)
    cv2.waitKey(0)

    src_pts = []
    dst_pts = []
    for m in goodMatches:
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp2[m.trainIdx].pt)
    
    src_pts = np.float32(src_pts).reshape(-1,1,2)
    dst_pts = np.float32(dst_pts).reshape(-1,1,2)

    # Find Homography using Least Squares method
    M, mask = cv2.findHomography(src_pts, dst_pts, 0, 5.0) 

    # Get height and width of original image    
    h,w = img1_gray.shape

    # define the corner points of the image to place over the other image
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    # Transform the image using the homography
    dst = cv2.warpPerspective(img1_raw,M,(img2_raw.shape[1] + img1_raw.shape[1], img2_raw.shape[0]))
    dst[0:img2_raw.shape[0],0:img2_raw.shape[1]] = img2_raw

    stitchedImg = trim(dst)

    return stitchedImg, allowance


# Need to put right hand/lower image as left input to function
def stitchDepthimages(img1_raw, img2_raw, img1_depth, img2_depth, minMatches = 2):
    # img1_gray = cv2.cvtColor(img1_raw,cv2.COLOR_BGR2GRAY)
    # img2_gray = cv2.cvtColor(img2_raw,cv2.COLOR_BGR2GRAY)

    img1_gray = img1_raw
    img2_gray = img2_raw

    # Create SIFT Detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find key points and descriptors using SIFT
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)

    # Create Brute force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1,des2,k=2)


    # Create list of matches between two pictures that are good matches with lowest variation in distances
    foundGoodMatches = False
    allowance = 0.03

    while(not foundGoodMatches):
        goodMatches = []
        for m,n in matches:
            if m.distance < allowance*n.distance:
                goodMatches.append(m)
        
        # If a sufficient number of matches has been found, use those. Otherwise, increase the allowance in variation
        if(len(goodMatches) < minMatches):
            allowance = allowance + 0.03
            print(allowance)
        else:
            foundGoodMatches = True
            print("Final Variation Allowance: " + str(allowance))

    # Set the color of the lines/circles to be drawn
    draw_params = dict(matchColor=(255,0,255),
                        singlePointColor=None,
                        flags=2)

    # # Show matches one by one
    # for match in goodMatches:
    #     goodMatchesImage = cv2.drawMatches(img1_raw,kp1,img2_raw,kp2,[match],None,**draw_params)    
    #     cv2.imshow("original_image_drawMatches.jpg", goodMatchesImage)
    #     cv2.waitKey(0)

    goodMatchesImage = cv2.drawMatches(img1_raw,kp1,img2_raw,kp2,goodMatches,None,**draw_params)
    goodMatchesDepth = cv2.drawMatches(img1_depth,kp1,img2_depth,kp2,goodMatches,None,**draw_params)

    cv2.waitKey(0)

    src_pts = []
    dst_pts = []
    for m in goodMatches:
        src_pts.append(kp1[m.queryIdx].pt)
        dst_pts.append(kp2[m.trainIdx].pt)
    
    src_pts = np.float32(src_pts).reshape(-1,1,2)
    dst_pts = np.float32(dst_pts).reshape(-1,1,2)

    # Find Homography using Least Squares method
    M, mask = cv2.findHomography(src_pts, dst_pts, 0, 5.0) 

    # Get height and width of original image    
    h,w = img1_gray.shape

    # define the corner points of the image to place over the other image
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    # Transform the image using the homography
    dst = cv2.warpPerspective(img1_raw,M,(img2_raw.shape[1] + img1_raw.shape[1], img2_raw.shape[0]))
    dst[0:img2_raw.shape[0],0:img2_raw.shape[1]] = img2_raw

    # Transform depth image using homography
    dst2 = cv2.warpPerspective(img1_depth,M,(img2_depth.shape[1] + img1_depth.shape[1], img2_depth.shape[0]))
    
    # Typically around 30 pixels of black borders on the right side of the original depth image, so clip that off when combining the images. 
    dst2[0:img2_depth.shape[0],0:img2_depth.shape[1] - 30] = img2_depth[0:img2_depth.shape[0], 0:img2_depth.shape[1] - 30]

    stitchedImg = trim(dst)
    stitchedDepth = trim(dst2)

    return stitchedImg, stitchedDepth, allowance


def calcHomography(srcPts,dstPts):
    A = []
    b = []
    for i in range(len(srcPts)):
        s_x, s_y = srcPts[i]
        d_x, d_y = dstPts[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b,rcond=1)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))



if __name__ == "__main__":
    main()


# def stitchImages(ListOfImages):
#     print(len(ListOfImages))
#     if(len(ListOfImages) == 1):
#         return ListOfImages[0]
    
#     img1 = ListOfImages.pop(0)
#     stitchedImg = img1
#     allowance = 100
#     imgIndex = -1

#     # Find the next image that requires the least allowance, and try the images both ways (dont know which is left and which is right)
#     for i in range(len(ListOfImages)):
#         img = ListOfImages[i]
#         try:
#             newImg1, newAllowance1 = stitch2images(img1, img)
#         except:
#             print("Exception1")
#             newAllowance1 = 101
        
#         try: # Catch the errors that sometimes occur when the images cant be stitched
#             newImg2, newAllowance2 = stitch2images(img, img1)
#         except:
#             print("Exception2")
#             newAllowance2 = 101

#         if(newAllowance1 < allowance and newAllowance1 < newAllowance2):
#             allowance = newAllowance1
#             stitchedImg = newImg1
#             imgIndex = i
#         if(newAllowance2 < allowance and newAllowance2 < newAllowance1):
#             allowance = newAllowance2
#             stitchedImg = newImg2
#             imgIndex = i
    
#     ListOfImages.pop(i)

#     newListOfImages = [stitchedImg, ListOfImages]
#     cv2.imshow("TestStitch",stitchedImg)
#     cv2.waitKey(0)
#     return stitchImages(newListOfImages)