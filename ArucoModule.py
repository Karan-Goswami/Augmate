import cv2
import cv2.aruco as aruco
import numpy as np
import os

from symbol import parameters

# creating function to find arucomarkers from the camera

# markersSize = 6 means 6*6
def findArucoMarkers(img, markersSize=6, totalMarkers=250, draw=True):

    # converting image from BGR to GRAY
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # getting size based on variables
    key = getattr(aruco, f'DICT_{markersSize}X{markersSize}_{totalMarkers}')

    arucoDict = aruco.Dictionary_get(key)

    arucoParam = aruco.DetectorParameters_create()

    # getting boundingboxes, ids, rejected_markers
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

# main function
def main():
    # enabling video capture device pass 0 for in-built laptops webcam

    cap = cv2.VideoCapture(0)

    # creating synchronize image reading
    while True:
        success, img = cap.read()
        cv2.imshow('Image', img)
        cv2.waitKey(1)


# Driver code
if __name__=='__main__':
    main()