import cv2
import cv2.aruco as aruco

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