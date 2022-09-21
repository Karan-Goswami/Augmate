import cv2
import cv2.aruco as aruco
import numpy as np
import os

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