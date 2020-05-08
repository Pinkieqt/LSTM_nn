import os
from scipy import ndimage, misc
import cv2

directory = "/home/pinkie/Downloads/Iphone_frames"
directoryToSave = "/home/pinkie/Downloads/Iphone_frames/"

counter = 0

for filename in os.listdir(directory):
    counter += 1
    if filename.endswith(".jpg"):
        pathToFile = directory + "/" + filename
        image_to_rotate = ndimage.imread(pathToFile)

        #rotate
        rotatedImg = cv2.rotate(image_to_rotate, cv2.ROTATE_90_CLOCKWISE)

        y=474
        x=0
        h=1400
        w=1080
        croppedImg = rotatedImg[y:y+h, x:x+w]

        misc.imsave(pathToFile, croppedImg)
        print("Obrázek č. " + str(counter))
    else:
        print("other file")

print("done")