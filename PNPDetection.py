import libjevois as jevois
import cv2 as cv
import numpy as np
import math

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois PythonSandbox
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PNPDetection:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        self.counter = 0
        self.currentIndex = 0
        self.rollAvgSize = 5
        self.eulers = np.zeros((self.rollAvgSize, 3, 1))
        #self.avgRvecs = np.zeros((self.rollAvgSize, 3, 1))
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        outimg = self.processNoUSB(inframe)
        
        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCv(outimg)
        
        
        
        
    def processNoUSB(self, inframe):
        #jevois.sendSerial("I don't know if it worked...")
        #return
        #jevois.sendSerial("Hi!")
        
        errorHappened = False
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        other = inframe.getCvGRAY()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()
        
        tapeCoords = np.array([
            #Left tape
            (-4, 0, 0), #right
            (-5.87, .71, 0), #top
            (-7.822, -4.432, 0), #left
            (-5.952, -5.142, 0), #bottom
            #(-5.911, -2.216, 0), #center
            #Right tape
            (7.822, -4.432, 0), #right
            (5.87, .71, 0), #top
            (4, 0, 0), #left
            (5.952, -5.142, 0), #bottom
            #(5.911, -2.216, 0) #center
        ])
        #1.911, 2.216
        
        hsv = cv.cvtColor(inimg, cv.COLOR_BGR2HSV)

        lower = np.array([47,50,50])
        upper = np.array([85,255,255])

        binary = cv.inRange(hsv, lower, upper)
        
        
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        contours.sort(key=lambda c: c[0][0][0])
        
        smallContourIndices = []
        cntAreas = []
        for i in range(len(contours)):
            if(cv.contourArea(contours[i]) < 150):
                smallContourIndices.append(i)
            else:
                #cv.drawContours(inimg,[box],0,(0,0,255),2)
                cntAreas.append(cv.contourArea(contours[i]))
        contours = np.delete(contours, smallContourIndices, -1)
        jevois.LINFO(str(len(contours)))
        
        
        
        outimg = inimg
        #outimg = cv.drawContours(inimg, contours, -1, (0, 0, 255), 1)
        
        targets = []
        for i in range(len(contours) - 1):
            _, _, angle1 = cv.fitEllipse(contours[i])
            _, _, angle2 = cv.fitEllipse(contours[i + 1])
            
            if(angle1 < 90 and angle2 > 90):
                targets.append([ contours[i], contours[i + 1] ])
        
        if(len(targets) > 0):
            outimg = cv.drawContours(outimg, contours, -1, (0, 0, 255), 1)
            (rvec, tvec, euler) = self.doThePnPThing(outimg, targets[0], tapeCoords)
        
        if(len(targets)>0):
            distance = math.hypot(tvec[0][0], tvec[2][0])
            rmat = cv.Rodrigues(rvec)[0]
            
            #yaw = math.atan2(rmat[2,0], rmat[2,1])
            camTvec = np.dot(-rmat.transpose(), tvec)
            
            #self.avgCamTvecs[self.currentIndex] = camTvec
            #self.currentIndex = (self.currentIndex + 1) % self.rollAvgSize
            #camTvec = np.mean(self.avgCamTvecs, axis=0)
            # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
            fps = self.timer.stop()
            outheight = outimg.shape[0]
            outwidth = outimg.shape[1]
            cv.putText(outimg, str(camTvec)+" "+fps, (3, outheight - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            #cv.Rodrigues(np.array(np.radians([[45],[0],[0]]), dtype = "float"))
            
            #jevois.sendSerial("{},{},{},{},{},{},{},{},{},{},{},{}".format(camTvec[0][0], camTvec[1][0], camTvec[2][0], rmat[0][0], rmat[0][1], rmat[0][2], rmat[1][0], rmat[1][1], rmat[1][2], rmat[2][0], rmat[2][1], rmat[2][2]))
            jevois.sendSerial("{},{},{},{},{},{}".format(camTvec[0][0],camTvec[1][0],camTvec[2][0], euler[0][0],euler[1][0],euler[2][0]))
        else:
            #jevois.sendSerial("0,0,0,0,0,0,0,0,0,0,0,0")
            jevois.sendSerial("0,0,0,0,0,0")
        # Convert our OpenCv output image to video output format and send to host over USB:
        #outframe.sendCv(inimg)
        return outimg
        
        
        
        
        
        
        
        
        
        
        
    def doThePnPThing(self, outimg, targetCnts, tapeCoords):
        cntCorners = []
        for i in range(len(targetCnts)):
                tapePiece = targetCnts[i]
                # determine the most extreme points along the contour
                rect = cv.minAreaRect(tapePiece)
                box = cv.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
                box = np.array(box)
                #for point in box:
                #    cv.circle(outimg, tuple(point), 3, (0, 255, 0), -1)
                #jevois.LINFO(str(box))
                #box = box.sort(key=lambda p: p[1])
                #epsilon = .01 * cv.arcLength(tapePiece, True)
                #tapePiece = cv.approxPolyDP(tapePiece, epsilon, True)
                jevois.LINFO("points: "+str(len(tapePiece)))
                extLeft = tuple(tapePiece[tapePiece[:, :, 0].argmin()][0])
                extRight = tuple(tapePiece[tapePiece[:, :, 0].argmax()][0])
                extTop = tuple(tapePiece[tapePiece[:, :, 1].argmin()][0])
                extBot = tuple(tapePiece[tapePiece[:, :, 1].argmax()][0])
                
                #M = cv.moments(tapePiece)
                #center = tuple([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                
                extLeft = tuple(box[box[:, 0].argmin()])#tuple(tapePiece[tapePiece[:, :, 0].argmin()][0])
                extRight = tuple(box[box[:, 0].argmax()])
                extTop = tuple(box[box[:, 1].argmin()])
                extBot = tuple(box[box[:, 1].argmax()])
                #cv.circle(outimg, tuple(extBot), 3, (255, 0, 0), -1)
                tapeCorners = [extRight, extTop, extLeft, extBot]#, center]
                #tapeCorners = [extRight, extLeft, center]
            
                cntCorners.append(tapeCorners)
        
        
        for tapeCorners in cntCorners:
            for i in tapeCorners:
                cv.circle(outimg, i, 3, (255, 255, 255), -1)
        
        if(True):
            targetCorners = np.array(cntCorners[0] + cntCorners[1], dtype="double")
        else:
            targetCorners = np.array(cntCorners[1] + cntCorners[0], dtype="double")
        
        
        #camera_matrix = np.array([[ 340.77738726, 0.,           320.22956535],
        #                         [  0.,           339.09687397, 240.36063506],
        #                         [  0.,           0.,           1.         ]], dtype = "double")
        #dist_coeffs = np.array([-0.01514104, 0.13156106, -0.00467079, -0.01357011, -0.09328762], dtype = "double")
        camera_matrix = np.array(
                         [[658.76910391474928, 0., 342.43117338910145],
                         [0., 658.37316715574661, 250.93545202231513],
                         [0., 0., 1.]], dtype = "double"
                         )
        
        dist_coeffs = np.array([ 2.3614217917861352e-01, -2.8983267012643715e-01,
         -1.1519795733873316e-03, -8.7701317797552109e-04,
         -2.4354602371303335e+00 ], dtype = "double")
        if(self.counter >= 0):
            didItWork, rvec, tvec = cv.solvePnP(tapeCoords, targetCorners, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
            #self.avgRvecs[self.currentIndex] = rvec
            #self.currentIndex = (self.currentIndex + 1) % self.rollAvgSize
            #rvec = np.mean(self.avgRvecs, axis=0)
            self.counter = self.counter + 5
            
            rmat = cv.Rodrigues(rvec)[0]
            
            projectionMatrix = np.hstack((rmat, tvec))
            eulerAng = cv.decomposeProjectionMatrix(projectionMatrix)[-1]
            self.eulers[self.currentIndex] = eulerAng
            self.currentIndex = (self.currentIndex + 1) % self.rollAvgSize
            eulerAng = np.mean(self.eulers, axis=0)
            yaw = eulerAng[1]
            #rvec = np.radians([[180],[0],[0]])
            
            centerPoints, jacobian = cv.projectPoints(np.array([(-5.911, -2.216, 0.0), (5.911, -2.216, 0.0)]), rvec, tvec, camera_matrix, dist_coeffs)
            leftCenter = tuple(np.array(centerPoints[0][0], dtype="int"))
            rightCenter = tuple(np.array(centerPoints[1][0], dtype="int"))
            #cv.circle(outimg, centerPoint, 3, (0, 255, 0), -1)
            #cv.line(outimg, leftCenter, rightCenter, (0, 255, 0), 2)
            
            centerPoints, jacobian = cv.projectPoints(np.array([(0, 0, 0.0), (5.0, 0, 0), (0, 5.0, 0), (0, 0, 5.0)]), rvec, tvec, camera_matrix, dist_coeffs)
            origin = tuple(np.array(centerPoints[0][0], dtype="int"))
            xPoint = tuple(np.array(centerPoints[1][0], dtype="int"))
            yPoint = tuple(np.array(centerPoints[2][0], dtype="int"))
            zPoint = tuple(np.array(centerPoints[3][0], dtype="int"))
            #cv.circle(outimg, centerPoint, 3, (0, 255, 0), -1)
            cv.line(outimg, origin, xPoint, (255, 0, 0), 2)
            cv.line(outimg, origin, yPoint, (0, 255, 0), 2)
            cv.line(outimg, origin, zPoint, (0, 0, 255), 2)
            
        
        outimg = cv.drawContours(outimg, targetCnts, -1, (0, 0, 255), 2)
        return (rvec, tvec, eulerAng)
