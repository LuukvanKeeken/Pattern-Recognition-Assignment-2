
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


if __name__=="__main__":
    
    data_dir = f"BigCats{os.sep}"
    dh = DataHandler()
    dh.loadData(data_dir)

    # show example images:
    dh.showExampleImages()

    # show class distribution
    dh.showClassDistribution()

    # preprocess data
    dh.preprocessData()

    # prepare cross validation
    v = Validator(dh.img_data, dh.class_labels)
    gd = GridSearch(v)

    # grid search on SVM
    gd.gridSearch('svm')
    # plot results:
    gd.plotSVMResults("Accuracy")

    # grid search on Random Forest
    gd.gridSearch('random_forest')
    gd.plotRandomForestResults("Accuracy")

    # grid search with KNN
    gd.gridSearch("knn")
    gd.plotKnnResults("Accuracy")
