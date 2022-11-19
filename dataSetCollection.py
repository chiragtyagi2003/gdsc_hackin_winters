import cv2

#package for hand detection
from cvzone.HandTrackingModule import HandDetector

cameraCapture = cv2.VideoCapture(0)


#detect the user's hand
detectHand = HandDetector(maxHands=1, detectionCon=0.8)

#open the webcam and detect the user's hand
while True:
    #store the user's image after capturing
    successful, img = cameraCapture.read()

    #find the user's hand in image
    storeHands, img = detectHand.findHands(img)

    #if hands are detected, detect one hand at a time.
    if storeHands:
        hand = storeHands[0]

    cv2.imshow("Image", img)
    cv2.waitKey(1)
