import os
import math
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


def plotPrediction(dataframe, threshold, source):
    dataframe["Anomaly"] = threshold
    dataframe["Percent"] = dataframe["Percent"] * 100
    dataframe.loc[dataframe['Percent'] > dataframe['Anomaly'],
                  'Anomaly'] = dataframe['Anomaly'] + 5
    dataframe.loc[dataframe['Percent'] < dataframe['Anomaly'],
                  'Anomaly'] = dataframe['Anomaly']

    # Plot when coughing, rotating head and when talking - ONLY FOR iphone_front_test_2 !!!!
    dataframe = viewClasses(source, dataframe)

    dataframe.plot(figsize=(16, 10))
    plt.xlabel("Snímek číslo: ")
    plt.ylabel("Přesnost v % (vyšší než " + str(threshold) + " = anomálie)")
    plt.show()


def animatePrediction(dataframe, fromimg, toimg, folderpath, predictSource):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = []
    for i in range(fromimg,  toimg):
        # Open image
        imgpath = folderpath + predictSource + "/img" + str(i) + ".jpg"
        img = np.array(Image.open(imgpath))
        im = plt.imshow(img, animated=True)
        t = ax.annotate(str("{:.4f}".format(dataframe[i][1] * 100) + "% , img:" + str(i)),
                        (img.shape[0] * 0.07, img.shape[1] * 0.1), color="white", size=15)
        ims.append([im, t])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=500)

    ani.save('./Media/BFRONT_nn_30_128_01_20.mp4', bitrate=1200)

    plt.show()


def plotImage(prediction, folderpath, predictSource, imgnumber):
    imgpath = folderpath + predictSource + "/img" + str(imgnumber) + ".jpg"
    img = np.array(Image.open(imgpath))

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    rect = patches.Rectangle(
        (img.shape[0] * 0.05, img.shape[1] * 0.05), img.shape[0] * 0.90,
        img.shape[1] * 0.90, linewidth=1, edgecolor="r", facecolor="none")

    plt.text(img.shape[0] * 0.07, img.shape[1] *
             0.1, str(float("{:.4f}".format(prediction)) * 100), color="red")

    # Load CSV points data
    ds = pd.read_csv('./Points/points_' + predictSource + '.csv',
                     sep="\t", index_col=0)

    arr = plotterPoints(ds, imgnumber)

    ax.add_patch(rect)

    plt.show()


def plotterPoints(dataset, imgnumber):
    print(dataset.shape)
    ds = np.array(dataset)
    counter = 1
    arr = []

    # Get data from points CSV
    for x in range(len(ds[imgnumber])):
        if counter == 1:
            arr.append(ds[imgnumber][x])
        if counter == 2:
            arr.append(ds[imgnumber][x])
            plt.scatter([arr[0]], [arr[1]], s=8, c="w")
            arr = []
        if counter == 3:
            counter = 0
        counter += 1
    return arr


def viewClasses(source, dataframe):
    if (source == "iphone_front_test_2"):
        dataframe["Coughing"] = np.nan
        dataframe["Head moving"] = np.nan
        dataframe["Talking"] = np.nan
        dataframe["Normal"] = np.nan
        for i in range(len(dataframe)):
            if (i > 783 and i < 876 or i > 881 and i < 1489):
                dataframe.set_value(i, "Coughing", -5)
            elif (i > 1914 and i < 2332):
                dataframe.set_value(i, "Head moving", -4)
            elif (i > 2618 and i < 3133):
                dataframe.set_value(i, "Talking", -6)
            else:
                dataframe.set_value(i, "Normal", -3)
    return dataframe
