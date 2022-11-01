#2715
import cv2
import cv2.aruco as aruco
import numpy as np
import os


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

    # Draw outlines on aruco markers and provides corner points
    if draw:
        # Aruco's own function to draw
        aruco.drawDetectedMarkers(img, bboxs)
    
    return [bboxs, ids]

# Aruco augment function for AR
# imgAug => img that we'll augment 
def augmentAruco(bbox, id, img, imgAug, drawID=True):
    # getting corner points
    topLeft = bbox[0][0][0], bbox[0][0][1]
    topRight = bbox[0][1][0], bbox[0][1][1]
    bottomRight = bbox[0][2][0], bbox[0][2][1]
    bottomLeft = bbox[0][3][0], bbox[0][3][1]

    # Size of img that will be augmented
    height, width, channels = imgAug.shape

    # Using Wrapperspective to get points for replacement or AR (**imp) 
    pts1 = np.array([topLeft, topRight, bottomRight, bottomLeft])
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # ---Now to get matrix for wrapperspective---
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    imgOut = img + imgOut

    # conditional statements
    if drawID:
        cv2.putText(imgOut, str(id), (int(topLeft[0]), int(topLeft[1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut


# main function
def main():
    # enabling video capture device pass 0 for in-built laptops webcam
    cap = cv2.VideoCapture(0)

    imgAug = cv2.imread("markers/23.png")

    # creating synchronize image reading
    while True:
        success, img = cap.read()
        
        # Call here (we get bboxs and ids of markers)
        arucoFound = findArucoMarkers(img)

        # checking if it detected anything or not
        # Looping through all the markers and augment each one
        if len(arucoFound[0])!=0:
            # Looping through together using zip
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                img = augmentAruco(bbox, id, img, imgAug)
        
        cv2.imshow('Image', img)
        k = cv2.waitKey(1)

        # Press esc key to close the camera window
        if k == 27:
            print('ESC')
            cv2.destroyAllWindows()
            break


# Driver code
if __name__=='__main__':
    main()