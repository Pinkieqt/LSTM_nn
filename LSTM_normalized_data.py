"""
Ressouces:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
Repository: https://github.com/ar-ms/lstm-mnist


# Page : https://github.com/curiousily/Deep-Learning-For-Hackers/blob/master/13.time-series-human_activity_recognition.ipynb
# Page : https://www.curiousily.com/posts/time-series-classification-for-human-activity-recognition-with-lstms-in-keras/#evaluation
# Ostatní : https://github.com/curiousily/Deep-Learning-For-Hackers
# bidirectional lstm: https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/
"""

# Imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Plotter import plotPrediction, plotImage, animatePrediction
from sklearn.model_selection import train_test_split

from keras import Sequential, optimizers
from keras.engine.saving import load_model
from keras.layers import Bidirectional, LSTM, Dropout, Dense
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Cuda - run on gpu "1" for CPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class LSTMnn:
    def __init__(self, inputfile, lookback, features_count):
        self.input = inputfile
        self.LOOKBACK = lookback
        self.FEATURES_COUNT = features_count

    # Read generated dataset from CSV (created by lstm_dataset_parser.py)
    def __read_csv_dataset(self, csvpath):
        mg_data = pd.read_csv(csvpath)
        mg_data = mg_data.drop(mg_data.columns[[0]], 1)
        return mg_data

    # Generate labels for training data [0, 1] = anomaly, [1, 0] = normal
    def genYnew(self, ds):
        Y = []
        tmp = 11146  # first data training size
        for i in range(ds.shape[0]):
            # hardcoded
            # if(self.ds.loc[i, 'D8'] > 17 and self.ds.loc[i, 'D10'] > 42 and self.ds.loc[i, 'D12'] < -42):
            # if(data.loc[i, 'D8'] > 153 and data.loc[i, 'D10'] > 42 and data.loc[i, 'D12'] < 100):
            # if(data.loc[i, 'D8'] > 0.011 and data.loc[i, 'D10'] > 0.015 and data.loc[i, 'D12'] < -0.05):
            if(i >= 9330 and i <= 9468 or i >= 9520 and i <= 10066
               or i >= 10110 and i <= 10345 or i >= 10370 and i <= 10516 or i >= 10550 and i <= 10626
               or i >= tmp + 6210 and i <= tmp + 6566 or i >= tmp + 6582 and i <= tmp + 6811
               or i >= tmp + 6825 and i <= tmp + 6913 or i >= tmp + 6940 and i <= tmp + 6969
               or i >= tmp + 6981 and i <= tmp + 7175 or i >= tmp + 7191 and i <= tmp + 7243
               or i >= tmp + 7268 and i <= tmp + 7359 or i >= tmp + 7414 and i <= tmp + 7511
               or i >= tmp + 7535 and i <= tmp + 7568 or i >= tmp + 7713 and i <= tmp + 7741):
                # anomaly
                Y.append([0, 1])
            else:
                # not anomaly, normal
                Y.append([1, 0])
        array = np.array(Y)
        return array

    # Get data n-times back
    def __temporalize(self, X, y):
        output_X, output_y = [], []
        for i in range(len(X) - self.LOOKBACK - 1):
            t = []
            for j in range(1, self.LOOKBACK + 1):
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + self.LOOKBACK + 1])
        return np.squeeze(np.array(output_X)), np.array(output_y)

    # Prepare data for NN
    def prepare_data(self, shuffle):
        self.ds = self.__read_csv_dataset(self.input)
        input_X = self.ds.loc[:, self.ds.columns != 'y'].values
        # Label X -> Generate Y data - if anomaly or not
        # input_Y = self.__generateY(self.ds)
        input_Y = self.genYnew(self.ds)  # trying new method

        # Switch first half of input data with second
        # (coughing is in 90th% of data, need to have it somewhere else to train on it, no to validate on it)
        input_X = np.concatenate((input_X[7000:], input_X[:7000]))
        input_Y = np.concatenate((input_Y[7000:], input_Y[:7000]))

        plt.plot(input_X)
        plt.show()

        # Temporalize
        X, y = self.__temporalize(X=input_X, y=input_Y)

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(X), np.array(y), shuffle=shuffle, test_size=0.20)

        print('X training shape = ', x_train.shape)
        print('Y training shape', y_train.shape)
        print('X test shape = ', x_test.shape)
        print('Y test shape', y_test.shape)

        return x_train, x_test, y_train, y_test

    def initModel(self, x_train, y_train, dropoutrate, learningrate):
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    units=128,
                    input_shape=[x_train.shape[1], x_train.shape[2]]
                )
            )
        )
        model.add(Dropout(rate=dropoutrate))  # 0.5 def
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))

        # descreased lr
        ad = optimizers.adam(lr=learningrate)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=ad,
            metrics=['acc']
        )
        self.model = model

    def trainModel(self, x, y, epochs, batch_size):
        history = self.model.fit(
            x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=False)
        return history

    def saveModel(self, path, name):
        self.model.save(path + name)
        print("Model saved")

    def evaluateModel(self, x, y):
        res = self.model.evaluate(x, y)
        print(res)
        return res

    def loadModel(self, modelpath):
        self.model = load_model(modelpath)
        return self.model

    def predictOn(self, model, csvpath):
        # Dataset to test model on
        ds = self.__read_csv_dataset(csvpath)

        input_X = ds.loc[:, ds.columns != 'y'].values
        input_Y = self.genYnew(ds)

        # Temporalize the data
        X, y = self.__temporalize(X=input_X, y=input_Y)

        print("Predicting on data shape: ")
        print(X.shape)
        prediction = model.predict(X)
        print("Done predicting.")

        return prediction


def main():
    train = False
    predict = False

    features = 14
    lookback = 30

    nn = LSTMnn(
        './Normalized_final_datasets/train_front_and_side.csv', lookback, features)

    if(train):

        # do not shuffle!
        x_train, x_test, y_train, y_test = nn.prepare_data(shuffle=False)

        nn.initModel(x_train=x_train, y_train=y_train,
                     dropoutrate=0.1, learningrate=0.0001)

        epochs = 20
        batch_size = 128  # 256
        history = nn.trainModel(
            x_train, y_train, epochs=epochs, batch_size=batch_size)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        nn.saveModel("./", "lstm_n_30_128_01.h5")

        nn.evaluateModel(x_test, y_test)
    if(not train):
        # lookback, batchsize, dropout
        model = nn.loadModel('./Models/lstm_n_30_128_01.h5')

        # Change dataset to predict on here
        folderSource = "/media/pinkie/A6FC43F0FC43B977/AData/OpenPose_images/"
        predictSource = "iphone_front_test"

        # prediction = nn.predictOn(model,
        #                           './Final_datasets/features_dataset_' + predictSource + '.csv')

        prediction = nn.predictOn(
            model, './Normalized_final_datasets/' + predictSource + '.csv')

        # Create pandas dataframe and plot it
        score = pd.DataFrame(columns=['Percent'])
        for i in range(len(prediction)):
            score = score.append(
                {'Percent': prediction[i][1]}, ignore_index=True)
        threshold = 82.5

        # Show plot with threshold
        plotPrediction(score, threshold=threshold, source=predictSource)

        # Show single image with prediction
        imgNumberToPlot = 783
        # plotImage(prediction[imgNumberToPlot][1], folderSource, predictSource, imgNumberToPlot)

        # Show images animation
        animatePrediction(prediction, 700, 1400,
                          folderSource, predictSource)


if __name__ == "__main__":
    main()
