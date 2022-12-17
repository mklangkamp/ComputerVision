
# from pykinect import nui
# from pykinect.nui import JointId
from pykinect import nui
from pykinect.nui import JointId
import numpy
from numpy import *
import numpy as np
import cv2
import datetime
import time

from pykinect.nui.structs import SkeletonData


from pykinect.nui import SkeletonTrackingState



class ezKinect:
    rgbImg = None
    depthImg = None
    Skeletons = [nui.SkeletonData]
    rgbResolution = None
    depthResolution = None
    jointsToDraw = []
    kinect = nui.Runtime()

    fpsLastTime = 1.0
    fps = 0.0

    

    # def __init__(self):

        

    def init(self, rgbResolution, depthResolution, ):
        # Init rgb camera image
        """Initializes the rgb and depth cameras on the Kinect

        Args:
            rgbResolution (nui.ImageResolution.ResolutionMxN): The desired resolution of the RGB Camera
            depthResolution (nui.ImageResolution.ResolutionMxN): The desired resolution of the Depth Camera
        """

        # Save the resolutions in the class
        ezKinect.rgbResolution = rgbResolution
        ezKinect.depthResolution = depthResolution

        # Initialize the callback function for the color image, and start the video stream
        ezKinect.kinect.video_frame_ready += ezKinect.getColorImage
        ezKinect.kinect.video_stream.open(nui.ImageStreamType.Video, 2,rgbResolution,nui.ImageType.Color)
        cv2.namedWindow('KINECT Video Stream', cv2.WINDOW_AUTOSIZE)

        # Initialize the callback function for the depth image, and start the video stream
        ezKinect.kinect.depth_frame_ready += ezKinect.getDepthImage
        ezKinect.kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, depthResolution, nui.ImageType.Depth)
        cv2.namedWindow('KINECT Depth Stream', cv2.WINDOW_AUTOSIZE)

        ezKinect.kinect.skeleton_engine.enabled = True
        ezKinect.kinect.skeleton_frame_ready += ezKinect.getSkeleton

    @staticmethod
    def getAllJointsXYZ():
        """Returns the list of coordinates of each body in the format: [[[body1Joint1X, body1Joint1Y, body1Joint1Z], [body1Joint2X, body1Joint2Y, body1Joint2Z], ...], [[body2Joint1X, body2Joint1Y, body2Joint1Z], ...] ]
           Returns the list of raw vector coordinates of each body in the format [[body1Vector1, body1Vector2...], [body2Vector1, body2Vector2...]...]
           Returns the two values as follows: [PixelCoordinateBodyValues, rawVectorCoordinateValues]

           Use pxCoord, rawCoord = ezK.getAllJointsXYZ()

        Returns:
            List: List of Bodies (Bodies contain joints, joints contain coordinates)
        """
        # Generate an empty array to hold the skeletons in
        bodies = []
        rawBodies = []

        for skeleton in ezKinect.Skeletons: # For each skeleton (body)
            # Generate an empty array to hold the joint locations for each skeleton in
            jointXY = []
            jointVectors = []

            if skeleton.eTrackingState == nui.SkeletonTrackingState.TRACKED: # If the given skeleton is being tracked (actually exists) then find the X,Y coordinates
                for i in range(0,JointId.count):
                    jointVector = ezKinect.getJointXY(skeleton, i)
                
                    # Convert the raw skeleton location in (i think) meters to pixel values, scaled by the resolution of the image
                    coord = ezKinect.kinect.skeleton_engine.skeleton_to_depth_image(jointVector, ezKinect.resolutionUnenumeration(ezKinect.rgbResolution)[1], ezKinect.resolutionUnenumeration(ezKinect.rgbResolution)[0])

                    # Convert the values to integers
                    xCoord = int(coord[0])
                    yCoord = int(coord[1])

                    # Append the coordinates to the list
                    jointXY.append([xCoord, yCoord, jointVector.z])
                    jointVectors.append(jointVector)

            # Append the list of coordinates for a given skeleton to the list
            bodies.append(jointXY)
            rawBodies.append(jointVectors)
        return [bodies, rawBodies]

    @staticmethod
    def getJointXY(skeleton, joint):
        """Takes a single skeleton, and a single joint index and returns the jointVector of that joint

        Args:
            skeleton (SkeletonData?): A single skeleton
            joint (int): An integer 0 to 19 that represents a specific joint coordinate

        Returns:
            JointVector?: A vector containing the X, Y, Z raw coordinate of the joint
        """
        nui.SkeletonData.get_skeleton_positions
        if skeleton.eTrackingState == nui.SkeletonTrackingState.TRACKED:
            skeletonPositions = skeleton.get_skeleton_positions()
            return skeletonPositions[joint]
        else:
            return None


    @staticmethod
    def getColorImage(frame):
        """Callback function for when an rgb image frame is ready
        """
        height,width = ezKinect.resolutionUnenumeration(ezKinect.rgbResolution)
        rgb = numpy.empty((height,width,4),numpy.uint8) 
        frame.image.copy_bits(rgb.ctypes.data) #copy the bit of the image to the array

        rgb = ezKinect.drawJointsOnImg(rgb,ezKinect.jointsToDraw)

        ezKinect.rgbImg = rgb 

    @staticmethod
    def getDepthImage(frame):
        """Callback function for when a depth image frame is ready
        """
        height,width = ezKinect.resolutionUnenumeration(ezKinect.depthResolution)
        #Original code that I found online
        depth = numpy.empty((height,width,1),numpy.uint8)   
        depthImg = (depth >> 3) & 4095 
        depthImg >>= 4
        frame.image.copy_bits(depthImg.ctypes.data)


        # #Other code that is meant to make it natively use the 16 bit data
        # depth = numpy.empty((height,width,1),numpy.uint16)
        # depthImg = depth   
        # depthImg = (depth >> 3) & 4095 
        # depthImg >>= 4
        # frame.image.copy_bits(depthImg.ctypes.data)

        depthImg = ezKinect.drawJointsOnImg(depthImg,ezKinect.jointsToDraw, jointColor=(0,0,0))

        ezKinect.depthImg = depthImg
        
        # print(time.time())
        ezKinect.fps = int(1/(time.time() - ezKinect.fpsLastTime))
        ezKinect.fpsLastTime = time.time()




    @staticmethod
    def resolutionUnenumeration(resolution):
        """Takes a nui.ImageResolution.ResolutionXxX and returns a tupe of the (height,width) of that resolution

        Args:
            resolution (nui.ImageResolution.ResolutionXxX): Desired image resolution

        Returns:
            tuple: (height, width) of desired image resolution
        """
        if(resolution == 0):
            return (60,80)
        if(resolution == 1):
            return (240,320)
        if(resolution == 2):
            return (480,640)
        if(resolution == 3):
            return (1024,1280)

    


    @staticmethod
    def getSkeleton(frame):
        """Callback function for when skeleton data is ready
        """
        ezKinect.Skeletons = frame.SkeletonData

    @staticmethod
    def drawJointsOnImg(img,jointList, jointColor = (0,255,0)):
        """Draws dots at every joint location included within jointList

        Args:
            img (img): Image from camera (depth camera works better, as joint positions are determined based on it)
            jointList ([JointID.x]): A list of joint IDs (0 to 19) including which joints you would like to have drawn
            jointColor (tuple, optional): Color to draw the joints. Defaults to (0,255,0).

        Returns:
            img: returns an image with the joints drawn on
        """
        if(ezKinect.jointsToDraw == []): # If no joints should be drawn, return the original image
            return img

        retImg = img

        bodyList = ezKinect.getAllJointsXYZ()[0] # Get just the pixel coordinates


        for body in bodyList: # For every body in the list
            if(body != []): # Only if there are coordinates associated with the body
                for index in jointList: # Put the desired joint coordinates onto the image
                    retImg = cv2.circle(retImg,tuple(body[index][0:2]),5,jointColor,-1) # Draw the circles where they belong

        return retImg
        

    @staticmethod
    def getSegmentOfImageAroundJoint(img, jointIDs, widthHeight, offsetXY):
        bodyList = ezKinect.getAllJointsXYZ()[0] # Get just the pixel coordinates
        bodyFound = False
        coordinates = []
        for body in bodyList:
            if(body != []):
                for id in jointIDs:
                    bodyFound = True
                    coordinates.append(body[id][0:2]) # Get the x and y coordinates of the desired bodyID(s)



        if(bodyFound):
            a = np.array(coordinates)
            newCoord = [int(np.average(a[:,0])) - offsetXY[0],int(np.average(a[:,1])) - offsetXY[1]]
            sizeOfImg = shape(img)
            xStart = clip(newCoord[0] - int(widthHeight[0]/2),0,sizeOfImg[1])
            xEnd = clip(newCoord[0] + int(widthHeight[0]/2),0,sizeOfImg[1]) # clip(xStart + offsetXY[0],0,sizeOfImg[1])#
            yStart = clip(newCoord[1] - int(widthHeight[1]/2),0,sizeOfImg[0])
            yEnd = clip(newCoord[1] + int(widthHeight[1]/2),0,sizeOfImg[0]) # clip(yStart + offsetXY[1],0,sizeOfImg[0])#
            
            segmentedImg = img[yStart:yEnd, xStart:xEnd]
        
            return (xStart,yStart),widthHeight, segmentedImg
        else:
            return (0,0), (0,0), numpy.zeros((100,100))

    @staticmethod
    def getXYZOfJoint(jointID):
        bodyList = ezKinect.getAllJointsXYZ()[0] # Get just the pixel coordinates
        coordinates = []
        for body in bodyList:
            if(body != []):
                return body[jointID] # Get the x and y and z coordinates of the desired bodyID(s)


    @staticmethod
    def getXYZOfJointMeters(jointID):
        bodyList = ezKinect.getAllJointsXYZ()[0] # Get just the pixel coordinates
        coordinates = []
        for body in bodyList:
            if(body != []):
                coord = body[jointID] # Get the x and y and z coordinates of the desired bodyID(s)

        coord[0] = ((coord[0] - 320)*numpy.tan(numpy.deg2rad(58.5/2))*coord[2])/320
        coord[1] = ((coord[1] - 240)*numpy.tan(numpy.deg2rad(46.6/2))*coord[2])/240

        return coord

