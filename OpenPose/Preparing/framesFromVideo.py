import os
import moviepy.editor as mp
import cv2


directory = "/media/pinkie/A6FC43F0FC43B977/AData/OpenPose_images/"
directoryToSave = "/media/pinkie/DataC/OpenPose_images/iphone_front_test_2/"
fileName = "iphone_front_test_2.mp4"

video = cv2.VideoCapture(directory + fileName)

i = 0

while video.isOpened():
    ret, frame = video.read()
    if ret == False:
        break
    cv2.imwrite(directoryToSave + "img" + str(i) + ".jpg", frame)
    print("Snímek číslo: " + str(i))
    i += 1

video.release()
cv2.destroyAllWindows()
