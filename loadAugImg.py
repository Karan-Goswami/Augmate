import os
import cv2

# Function to load images that we want to augment
def loadAugImages(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    augDict = dict()
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f"{path}/{imgPath}")
        augDict[key] = imgAug
    return augDict
