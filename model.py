import math
import time
import cv2
import numpy as np
import pyttsx3
# package for hand detection
from cvzone.HandTrackingModule import HandDetector

#package for classification
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
classifier=Classifier("keras_model.h5","labels.txt")

labels = ["A", "B", "C", "D", "E", "F", "H", "K", "N", "O", "T", "I", "L", "W"]

#variables to form string
ans = ""
ans2 = ""
ans3 = ""


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
            constant = sizeOfImage/h
            calculatedWidth = math.ceil(constant*w)

            # resizing the image
            # @param1 image to be resized
            # @param2 new dimensions (width, height)
            resizedImage = cv2.resize(croppedImage, (calculatedWidth, sizeOfImage))

            widthGap = math.ceil((sizeOfImage - calculatedWidth) / 2)

            # center the cropped image overlaying the white image
            whiteImage[:, widthGap:calculatedWidth + widthGap] = resizedImage

            predictedLetter, predictedIndex = classifier.getPredicted(whiteImage)

            ans += labels[predictedIndex]

        else:
            constant = sizeOfImage/w
            calculatedHeight = math.ceil(constant*h)

            # resizing the image
            # @param1 image to be resized
            # @param2 new dimensions (width, height)
            resizedImage = cv2.resize(croppedImage, (calculatedHeight, sizeOfImage))

            heightGap = math.ceil((sizeOfImage - calculatedHeight) / 2)


            # center the cropped image overlaying the white image
            whiteImage[:, heightGap:calculatedHeight + heightGap] = resizedImage

            predictedLetter, predictedIndex = classifier.getPredicted(whiteImage)

            ans += labels[predictedIndex]



        # show the cropped image
        cv2.imshow("ImageCrop", croppedImage)

        # show the white image overlayed with cropped image
        cv2.imshow("WhiteImage", whiteImage)





    # show the initial image
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):

        # store the last letter in one prediction
        ans2 = ans[len(ans) - 1]

        # store the final string
        ans3 += ans2
        print(ans3)

        # empty ans2 to store new letter
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

