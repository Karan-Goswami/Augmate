import numpy as np
import cv2

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

