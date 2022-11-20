import math
import time
import cv2
import numpy as np
# package for hand detection
from cvzone.HandTrackingModule import HandDetector

cameraCapture = cv2.VideoCapture(0)

# detect the user's hand
detectHand = HandDetector(maxHands=1, detectionCon=0.8)

# margin for cropped image
margin = 20

# size of white image
sizeOfImage = 300

# count of saved image
count = 0

# folder path
folder = "dataSet/W"

# @param1 fixed size of image
# @param2 width of the image
# @param3 height of image
def calculateNewHeight(imgSize, width, height):
    """ returns the new calculated height for resizing"""
    constant = imgSize/width
    newHeight = math.ceil(constant * height)
    return newHeight

# @param1 fixed size of image
# @param2 width of the image
# @param3 height of image
def calculateNewWidth(imgSize, width, height):
    """ returns the new calculated width for resizing"""
    constant = imgSize / height
    newWidth = math.ceil(constant * width)
    return newWidth

# @param1 fixed size of image
# @param2 calculated height for resizing
def calculateHeightGap(imgSize, newHeight):
    """returns the calculated height gap"""
    calculatedHeightGap = math.ceil((imgSize - newHeight)/2)
    return calculatedHeightGap


# @param1 fixed size of image
# @param2 calculated width for resizing
def calculateWidthGap(imgSize, newWidth):
    """returns the calculated width gap"""
    calculatedWidthGap = math.ceil((imgSize - newWidth)/2)
    return calculatedWidthGap

# open the webcam and detect the user's hand
while True:
    # store the user's image after capturing
    successful, img = cameraCapture.read()

    # find the user's hand in image
    storeHands, img = detectHand.findHands(img)

    # if hands are detected, detect one hand at a time.
    if storeHands:
        hand = storeHands[0]

        # store the starting coordinates of image
        # the width and the height
        # @param1 area of detected hand
        x, y, w, h = hand['bbox']

        # cropped image for better data traing
        croppedImage = img[y-margin:y+h+margin, x-margin:x+w+margin]

        # to handle different sizes
        # we will use an image of fixed size
        # then overlay the cropped image on that
        whiteImage = np.ones((sizeOfImage, sizeOfImage, 3), np.uint8) * 255



        # handling the aspect ratio
        aspectRatio = h/w

        # if the height is bigger than width
        # stretch the height to 300 and then
        # calculate width
        if aspectRatio > 1:
            # calculate new width for resizing
            calculatedWidth = calculateNewWidth(sizeOfImage, w, h)

            # resizing the image
            # @param1 image to be resized
            # @param2 new dimensions (width, height)
            resizedImage = cv2.resize(croppedImage, (calculatedWidth, sizeOfImage))

            # compute the width gap
            widthGap = calculateWidthGap(sizeOfImage, calculatedWidth)

            # center the cropped image overlaying the white image
            whiteImage[:, widthGap:calculatedWidth + widthGap] = resizedImage

        else:
            # compute new height for resizing
            calculatedHeight = calculateNewHeight(sizeOfImage, w, h)

            # resizing the image
            # @param1 image to be resized
            # @param2 new dimensions (width, height)
            resizedImage = cv2.resize(croppedImage, (sizeOfImage, calculatedHeight))

            heightGap = calculateHeightGap(sizeOfImage, calculatedHeight)

            # center the cropped image overlaying the white image
            whiteImage[heightGap:calculatedHeight + heightGap, :] = resizedImage

        # show the cropped image
        cv2.imshow("ImageCrop", croppedImage)

        # show the white image overlayed with cropped image
        cv2.imshow("WhiteImage", whiteImage)


    # show the initial image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # save the image to destined path
    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', whiteImage)
        print(count)
