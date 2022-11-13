import cv2
import cv2.aruco as aruco
import numpy as np
import os
import augmentAruco as augA
import findAruco as findA
import loadAugImg as loadImg

# main function
def main():
    # enabling video capture device pass 0 for in-built laptops webcam
    cap = cv2.VideoCapture(0)

    # imgAug = cv2.imread("markers/24.png")
    augDict = loadImg.loadAugImages("markers")

    # creating synchronize image reading
    while True:
        success, img = cap.read()
        
        # Call here (we get bboxs and ids of markers)
        arucoFound = findA.findArucoMarkers(img)

        # checking if it detected anything or not
        # Looping through all the markers and augment each one
        if len(arucoFound[0])!=0:
            # Looping through together using zip
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDict.keys():
                    img = augA.augmentAruco(bbox, id, img, augDict[int(id)])
        
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