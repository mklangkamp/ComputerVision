from pykinect import nui
from pykinect.nui import JointId
import numpy
import numpy as np
import cv2
import time
import math

from ezKinect import ezKinect

# Calibration Data
calPtsDepth = [(550,65), (459,66), (376,58), (274,57), (140,53), (145,164), (225,166), (340,176), (438,179), (517,183), (545,270), (318,251), (224,251), (118,254), (168,347), (269,335), (364,333), (468,331), (568,331), (54,425), (51,38), (572,35), (591,396)]
calPtsRGB = [(528,106), (443,110), (336,103), (273,103), (149,98), (152,200), (227,203), (331,212), (421,214), (494,216), (520,293), (308,278), (223,279), (127,283), (171,365), (266,355), (353,354), (449,353), (542,354), (68,438), (62,84), (547,78), (563,411)]




def main():
    ezK = ezKinect()
    ezK.init(nui.ImageResolution.resolution_640x480, nui.ImageResolution.resolution_640x480)

    M = calcHomography(calPtsDepth,calPtsRGB)

    size = (640,480)

    numImgsSaved = 0

    while(True):
        # Normalize the depth img
        if(type(ezK.depthImg) == numpy.ndarray):
            depth = ezK.depthImg
            warpedImg = cv2.warpPerspective(depth,M,(depth.shape[1], depth.shape[0]))
            #
            
        # Show rgb video stream
        try:
            cv2.imshow('KINECT Video Stream', cv2.flip(ezK.rgbImg,1))

        except:
            pass
                
        # Display the normalized depth stream and print the coordinates of the right hand
        try:
            cv2.imshow('KINECT Depth Stream',cv2.flip(warpedImg,1))
        except:
            pass

        key = cv2.waitKey(1)  
        if(key == ord("b")):
            numImgsSaved = numImgsSaved + 1
            cv2.imwrite("depthImg" + str(numImgsSaved) + ".tiff",cv2.flip(warpedImg,1))
            cv2.imwrite("RGBImg" + str(numImgsSaved) + ".jpg",cv2.flip(ezK.rgbImg,1))
            print("Image " + str(numImgsSaved) +  " saved")

        if key == 27:
            cv2.destroyAllWindows()
            ezK.kinect.close()
            print("Images Saved")
            break



    # Begin Stitching Code -----------------------------------------------------------------------------------------------------
    # Open first image and scale down so it can be shown with imshow
    img1 = cv2.imread('RGBImg1.jpg', cv2.IMREAD_GRAYSCALE)
    img1_d = cv2.imread('depthImg1.tiff', cv2.IMREAD_GRAYSCALE)
    

    # Open second image and scale down so it can be shown with imshow
    img2 = cv2.imread('RGBImg2.jpg', cv2.IMREAD_GRAYSCALE)
    img2_d = cv2.imread('depthImg2.tiff', cv2.IMREAD_GRAYSCALE)

    stitchedImg, depthStitched, allowance = stitchDepthimages(img2,img1,img2_d, img1_d, minMatches=20)

    # Show the stitched img
    cv2.imshow("RGB Stitched", stitchedImg)
    cv2.imshow("Depth Stitched", depthStitched)
    cv2.imshow("Depth Img 1", img1_d)
    cv2.imshow("Depth Img 2", img2_d)
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
    
    cv2.imshow("original_image_drawMatches.jpg", goodMatchesImage)
    cv2.imshow("original_depth_image_drawMatches.jpg", goodMatchesDepth)

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


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame





if __name__ == "__main__":
    main()