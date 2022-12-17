from pykinect import nui
from pykinect.nui import JointId
import numpy
import numpy as np
import cv2
import time
import math

from ezKinect import ezKinect


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
        #     # depth = np.array(ezK.depthImg)
        #     # depth  = np.divide(depth,np.max(depth))
        #     # depth = depth*256
        #     # depth.astype(np.uint8)
            # depth = display(ezK.depthImg,0,256)
            # depth = cv2.GaussianBlur(ezK.depthImg,(11,11),5)
            depth = ezK.depthImg
            warpedImg = cv2.warpPerspective(depth,M,(depth.shape[1], depth.shape[0]))
            # warpedImg = trim(warpedImg)

            # warpedImg = cv2.resize(warpedImg,(640,480))
            
            # depth  = (np.divide(np.array(ezK.depthImg)-np.min(ezK.depthImg),np.max(ezK.depthImg)))*256
            # # print(np.max(depth), np.min(depth))
            # print(type(ezK.depthImg[0,0]))
            
        # Show rgb video stream
        try:
            # cv2.circle(ezK.rgbImg,(int(640/3),int(240/3)),10,(255,255,255),4)
            # cv2.circle(ezK.rgbImg,(int(2*640/3),int(2*240/3)),10,(255,255,255),4)
            # cv2.circle(ezK.rgbImg,(int(2*640/3),int(240/3)),10,(255,255,255),4)
            # cv2.circle(ezK.rgbImg,(int(640/3),int(2*240/3)),10,(255,255,255),4)
            cv2.imshow('KINECT Video Stream', cv2.flip(ezK.rgbImg,1))

        except:
            pass
                
        


        # Display the normalized depth stream and print the coordinates of the right hand
        try:
            x = -10
            # cv2.circle(warpedImg,(int(640/3)+x,int(240/3)),10,(255,255,255),4)
            # cv2.circle(warpedImg,(int(2*640/3)+x,int(2*240/3)),10,(255,255,255),4)
            # cv2.circle(warpedImg,(int(2*640/3)+x,int(240/3)),10,(255,255,255),4)
            # cv2.circle(warpedImg,(int(640/3)+x,int(2*240/3)),10,(255,255,255),4)
            cv2.imshow('KINECT Depth Stream',cv2.flip(warpedImg,1))
        except:
            pass

        key = cv2.waitKey(1)  
        if(key == ord("a")):
            numImgsSaved = numImgsSaved + 1
            cv2.imwrite("rgbCalImg" + str(numImgsSaved) + ".jpg",cv2.flip(ezK.rgbImg,1))
            cv2.imwrite("depthCalImg" + str(numImgsSaved) + ".tiff",cv2.flip(ezK.depthImg,1))
            print("Image 1 saved")

        if(key == ord("b")):
            numImgsSaved = numImgsSaved + 1
            cv2.imwrite("depthImg" + str(numImgsSaved) + ".tiff",cv2.flip(warpedImg,1))
            cv2.imwrite("RGBImg" + str(numImgsSaved) + ".jpg",cv2.flip(ezK.rgbImg,1))
            print("Image saved")

        if key == 27:
            # cv2.imwrite("depthImg.tiff",cv2.flip(depth,1))
            cv2.destroyAllWindows()
            ezK.kinect.close()
            print("Video should have been saved")
            break


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