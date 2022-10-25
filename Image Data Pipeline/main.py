
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


def build_vocab(descriptors):
    return KMeans(n_clusters=5, random_state=0).fit(descriptors) 

def build_histograms(descriptors, y, vocab, filename = "histograms.npy"):
    histograms = []
    # Reshape back into per image
    descriptors = descriptors.reshape((len(y),100,128))
    for descriptors_img, label in zip(descriptors,y):
        distances = cdist(descriptors_img, vocab.cluster_centers_)
        bin_numbers = np.argmin(distances, axis=1)
        histogram = (np.zeros(len(vocab.cluster_centers_)), label)
        for bin_number in bin_numbers:
            histogram[0][bin_number] += 1  
        histograms.append(histogram)
        print(histogram)
    np.save(filename,histograms)
    return histograms

def main():
    # Initial settings
    data_dir = f"BigCats{os.sep}"
    IMG_WIDTH = 50
    IMG_HEIGHT = 50
    dh = DataHandler()
    dh.loadData(data_dir, IMG_WIDTH, IMG_HEIGHT)

    dh.showClassDistribution()
    dh.showExampleImages()
    

    target_dict = dh.convertLabelsToNumeric()

    images = dh.preprocessData()
    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
    images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    sift = cv2.SIFT_create()

    # descriptors_train = np.load("descriptors_train.npy")
    # descriptors_test = np.load("descriptors_test.npy")
    descriptors_train = []    
    descriptors_test = []    
    for i, img in enumerate(X_train):
            print(f"{i}/{len(X_train)}")
            keypoints, descriptor = sift.detectAndCompute(img, None)
            descriptors_train.append(descriptor[0:100])
    descriptors_train = np.array([el for i in descriptors_train for el in i])
    np.save("descriptors_train.npy", descriptors_train)   
    for i, img in enumerate(X_test):
            print(f"{i}/{len(X_test)}")
            keypoints, descriptor = sift.detectAndCompute(img, None)
            descriptors_test.append(descriptor[0:100])
    descriptors_test = np.array([el for i in descriptors_test for el in i])
    np.save("descriptors_test.npy", descriptors_test)   
    print(descriptors_train)
    vocab = build_vocab(descriptors_train)

    train_histograms = build_histograms(descriptors_train, y_train, vocab)
    test_histograms = build_histograms(descriptors_test, y_test, vocab, filename="histograms_test")
    # Feature extraction
    # sift = SIFT()
    
    # for img in X_train:
    #     keypoints, descriptor = sift.computeKeypointsAndDescriptors(img)
    #     for keypoint in keypoints: 
    #         pass
    # dist = np.linalg.norm(a-b)

    # Training and Testing the Model
    model = SVM()
    model.train([hist[0] for hist in train_histograms], [hist[1] for hist in train_histograms])
    model.test([hist[0] for hist in test_histograms], [hist[1] for hist in test_histograms])



if __name__=="__main__":
    main()
