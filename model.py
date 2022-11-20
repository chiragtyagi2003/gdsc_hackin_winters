import math
import cv2
import numpy as np

#package to convert text to speech
import pyttsx3

# package for hand detection
from cvzone.HandTrackingModule import HandDetector

# package for classification
from cvzone.ClassificationModule import Classifier

cameraCapture = cv2.VideoCapture(0)

# detect the user's hand
detectHand = HandDetector(maxHands=1, detectionCon=0.8)

# margin for cropped image
margin = 20

# size of white image
sizeOfImage = 300

#count of saved image
count = 0

#folder path
folder = "dataSet/F"

#classifer from teachable machine
classifier=Classifier("keras_model.h5", "labels.txt")


# list for letter mapping
labels = ["A", "B", "C", "D", "E", "F", "H", "K", "N", "O", "T", "I", "L", "W"]

#variables to form string
ans = ""
ans2 = ""
ans3 = ""


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

# @param1 image from which the prediction
# has to be made
def fetchprediction(backWhiteImg):
    """returns the predicted value and index"""
    return classifier.getPrediction(backWhiteImg)

def convert(keyValue):
    """converts the sign language into text and speech"""

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
            calculatedWidth = calculateNewWidth(sizeOfImage, w, h)

            # resizing the image
            resizedImage = cv2.resize(croppedImage, (calculatedWidth, sizeOfImage))

            # compute the width gap
            widthGap = calculateWidthGap(sizeOfImage, calculatedWidth)

            # center the cropped image overlaying the white image
            whiteImage[:, widthGap:calculatedWidth + widthGap] = resizedImage

            # predicted letter and predicted index
            predictedLetter, predictedIndex = fetchprediction(whiteImage)

            # add to the string
            ans += labels[predictedIndex]

        else:
            # calculate new height of image for
            # resizing the image
            calculatedHeight = calculateNewHeight(sizeOfImage, w, h)

            # resizing the image
            # @param1 image to be resized
            # @param2 new dimensions (width, height)
            resizedImage = cv2.resize(croppedImage, (sizeOfImage, calculatedHeight))

            # compute the height gap
            # @param1 fixed size of image
            # @param2 new height calculated for resizing
            heightGap = calculateHeightGap(sizeOfImage, calculatedHeight)


            # center the cropped image overlaying the white image
            whiteImage[heightGap:calculatedHeight + heightGap, :] = resizedImage

            # predicted letter and predicted index
            predictedLetter, predictedIndex = fetchprediction(whiteImage)

            # add to the string
            ans += labels[predictedIndex]



        # show the cropped image
        cv2.imshow("ImageCrop", croppedImage)

        # show the white image overlayed with cropped image
        cv2.imshow("WhiteImage", whiteImage)


    # show the initial image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):

        # store the last letter
        ans2 = ans[len(ans) - 1]

        # add to final string
        ans3 += ans2

        # empty the var to store new letter
        ans2 = ""


    # enter space between two words
    if key == ord("p"):
        ans3 += " "


    # convert the text into audio speech
    if key == ord("r"):
        print(ans3.split())
        text_to_speech = pyttsx3.init()
        text_to_speech.say(ans3.split())
        text_to_speech.runAndWait()


