
from distutils.command.build import build
from turtle import distance
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from DataHandler import DataHandler
from SIFT import SIFT
from Models import SVM
import cv2
import os
import numpy as np
import pickle
from GridSearch import GridSearch
from Validator import Validator
import sys


def main():
    data_dir = f"BigCats{os.sep}"
    dh = DataHandler()
    dh.loadData(data_dir)
    data_type = sys.argv[1]
    # preprocess data
    images = dh.preprocessData(data_type)

    # Make Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    if data_type == "augmented":
        X_train = np.append(images, dh.augmentData(X_train), axis=0)
        y_train += y_train

    # Perform gridsearch to find optimal params
    gs = GridSearch(X_train, y_train, data_type)
    gs.gridSearch()


if __name__ == "__main__":
    main()
