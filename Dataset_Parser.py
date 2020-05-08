


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES_XY = [(18, 36), (21, 39), (22, 42), (25, 45), (37, 41), (44, 46),
               (3, 30), (30, 13), (49, 55), (51, 57), (53, 59), (48, 54), (57, 8), (5, 11)]

COLUMNS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7',
           'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14']


class DatasetParser:
    def __init__(self, inputpath, outputpath):
        self.input = inputpath
        self.output = outputpath

    def __prepare_dataset(self):
        mg_data = pd.DataFrame()
        dataset = pd.read_csv(self.input, sep="\t", index_col=0)
        dataset_numpy = np.array(dataset)
        dataset_index = 0
        for element in dataset_numpy:
            dataset_numpy = pd.DataFrame(element.reshape(1, 210))
            dataset_numpy.index = [dataset_index]
            dataset_index += 1
            mg_data = mg_data.append(dataset_numpy)
            if(dataset_index % 10 == 0):
                print(dataset_index)

        myarray = []
        for x in range(210):
            myarray.append(str(x))
        mg_data.columns = myarray
        self.panda = mg_data
        return self.panda

    def __compute_features(self):
        distances_dataframe = pd.DataFrame(columns=COLUMNS)
        for x in range(len(self.panda)):
            distances = []
            for y in range(len(FEATURES_XY)):
                point1 = [self.panda.iat[x, 3 * FEATURES_XY[y][0]], self.panda.iat[x,
                                                                                   (3 * FEATURES_XY[y][0]) + 1], self.panda.iat[x, (3 * FEATURES_XY[y][0]) + 2]]
                point2 = [self.panda.iat[x, 3 * FEATURES_XY[y][1]], self.panda.iat[x,
                                                                                   (3 * FEATURES_XY[y][1]) + 1], self.panda.iat[x, (3 * FEATURES_XY[y][1]) + 2]]

                distances.append(math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(
                    point2[1] - point1[1], 2) + math.pow(point2[2] - point1[2], 2)))
            distances_dataframe.loc[x] = distances
            if(x % 10 == 0):
                print(x)
        self.features_panda = distances_dataframe
        return distances_dataframe

    def __read_csv_dataset(self):
        data = pd.read_csv(self.input)
        self.panda = data.drop(data.columns[[0]], 1)

    def parseCsv(self):
        ds = self.__prepare_dataset()
        ds = self.__compute_features()
        ds.to_csv(self.output)
        return ds

    def normalized(self, data):
        mean = data.mean()
        var = data.var()
        for i in range(len(data)):
            data.loc[i, :] = (data.loc[i, :] - mean)/var
        return data


def main():
    # 1. Load data from OpenPose and create Dataset
    # Arguments: Input (OpenPose csv), Output (Computed set of Features)
    dp = DatasetParser("./Points/points_iphone_front_test_2.csv",
                       "./Final_datasets/iphone_front_test_2.csv")
    dataset = dp.parseCsv()
    print("Dataset saved.")

    # 2. If needed, normalize computed set of features
    nds = dp.normalized(dataset)
    nds.to_csv("./Normalized_final_datasets/iphone_front_test_2.csv")
    print(dataset.head())


if __name__ == "__main__":
    main()
