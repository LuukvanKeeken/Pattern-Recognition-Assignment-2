
from distutils.command.build import build
from turtle import distance
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from DataHandler import DataHandler
import cv2
import os
import numpy as np
from GridSearch import GridSearch
import sys


def main():
    data_dir = f"Data\BigCats{os.sep}"
    dh = DataHandler()
    dh.load_data(data_dir)
    data_type = sys.argv[1]
    # preprocess data
    images = dh.preprocess_data(data_type)

    # Make Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    if data_type == "augmented":
        X_train = np.append(images, dh.augment_data(X_train), axis=0)
        y_train += y_train

    # Perform gridsearch to find optimal params
    gs = GridSearch(X_train, y_train, data_type)
    gs.gridsearch()


if __name__ == "__main__":
    main()
