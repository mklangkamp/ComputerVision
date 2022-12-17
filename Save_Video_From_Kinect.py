from pykinect import nui
from pykinect.nui import JointId
import numpy
import numpy as np
import cv2
import time
import math

from ezKinect import ezKinect

def display(image, display_min, display_max):
    # Here I set copy=True in order to ensure the original image is not
    # modified. If you don't mind modifying the original image, you can
    # set copy=False or skip this step.
    image = np.array(image, copy=True, dtype=np.uint8)

    # image.clip(display_min, display_max, out=image)
    # image -= display_min
    # image = image / ((display_min - display_max + 1) / 256.)
    # image = image.astype(np.uint8)

    image = np.array((image/128), dtype=np.uint8)

    return image
    # Display image

def main():
    ezK = ezKinect()
    ezK.init(nui.ImageResolution.resolution_640x480, nui.ImageResolution.resolution_640x480)


    size = (640,480)

    result = cv2.VideoWriter("Scan1.avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, size,False)
    
    while(True):
        # Normalize the depth img
        if(type(ezK.depthImg) == numpy.ndarray):
            # depth = np.array(ezK.depthImg)
            # depth  = np.divide(depth,np.max(depth))
            # depth = depth*256
            # depth.astype(np.uint8)
            depth = display(ezK.depthImg,0,256)
            result.write(depth)
            print(np.max(depth), np.min(depth))
            
        # Show rgb video stream
        # try:
        #     cv2.imshow('KINECT Video Stream', ezK.rgbImg)
        # except:
        #     pass
                
        


        # Display the normalized depth stream and print the coordinates of the right hand
        try:
            cv2.imshow('KINECT Depth Stream',cv2.flip(depth,1))
        except:
            pass

        key = cv2.waitKey(1)        
        if key == 27:           
            cv2.destroyAllWindows()
            ezK.kinect.close()
            result.release()
            print("Video should have been saved")
            break


    



if __name__ == "__main__":
    main()