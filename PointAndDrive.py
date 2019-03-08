import libjevois as jevois
import cv2 as cv
import numpy as np

## Detects targets
#
# Add some description of your module here.
#
# @author Team Steam
# 
# @videomapping YUYV 640 480 29.8 YUYV 640 480 29.8 TeamSteam TargetDetection
# @email teamsteamrobotics@gmail.com
# @address 123 first street, Los Angeles CA 90012, USA
# @copyright Copyright (C) 2018 by Team Steam
# @mainurl frcteam5119.com
# @supporturl frcteam5119.com
# @otherurl frcteam5119.com
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class TargetDetection:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        jevois.LINFO(str(dir(jevois)))
    
    def processNoUSB(self, inframe):
        inimg = inframe.getCvBGR()
        
        hsv = cv.cvtColor(inimg, cv.COLOR_BGR2HSV)
        
        lower = np.array([47,150,100])
        upper = np.array([85,255,255])

        binary = cv.inRange(hsv, lower, upper)
        
        contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
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
        #jevois.LINFO(str(len(contours)))
        
        
        
        outimg = inimg
        outimg = cv.drawContours(inimg, contours, -1, (0, 0, 255), 1)
        
        targets = []
        for i in range(len(contours) - 1):
            _, _, angle1 = cv.fitEllipse(contours[i])
            _, _, angle2 = cv.fitEllipse(contours[i + 1])
            
            if(angle1 < 90 and angle2 > 90):
                targets.append([contours[i], contours[i+1]])
        
        centerMostCenter = (0,0)
        
        for target in targets:
            leftM = cv.moments(target[0])
            rightM = cv.moments(target[1])
            leftCenter = tuple([int(leftM["m10"] / leftM["m00"]), int(leftM["m01"] / leftM["m00"])])
            rightCenter = tuple([int(rightM["m10"] / rightM["m00"]), int(rightM["m01"] / rightM["m00"])])
            centerSum = (leftCenter[0] + rightCenter[0], leftCenter[1] + rightCenter[1])
            center = (int(centerSum[0] / 2), int(centerSum[1] / 2))
            
            if(abs(320-center[0]) < abs(320 - centerMostCenter[0])):
                centerMostCenter = center
        
        cv.circle(outimg, centerMostCenter, 3, (255, 255, 255), -1)
        jevois.sendSerial("{},{}".format(centerMostCenter[0], centerMostCenter[1]))
        return outimg
        
        
        
        
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        jevois.LINFO(str(dir(jevois)))
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR. If you need a
        # grayscale image, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB() and getCvRGBA():
        inimg = inframe.getCvBGR()
        outimg = self.processNoUSB(inframe)
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv.putText(outimg, fps, (3, height - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Convert our output image to video output format and send to host over USB:
        outframe.sendCv(outimg)