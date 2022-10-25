
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from DataHandler import DataHandler
from SIFT import SIFT
from Models import SVM
import cv2
import os
import numpy as np
import pickle

def build_vocab(images):
    sift = SIFT()
    descriptors = []
    

    for i, img in enumerate(images):
        print(f"{i}/{len(images)}")
        # cv2.imshow("img", img)
        # cv2.waitKey(0) 
        keypoints, descriptor = sift.computeKeypointsAndDescriptors(img)
        descriptors.append(descriptor)
    print(len(descriptors[0]))
    descriptors = [el for i in descriptors for el in i]
    print(len(descriptors))
    # np.save("descriptor", descriptors)
    # file = open("descriptor.pkl",'r', encoding='utf-8')
    vocab = KMeans(n_clusters=200, random_state=0).fit(descriptors)
    np.save("vocab", vocab)
    return vocab

def main():
        
    # Initial settings

    data_dir = f"BigCats{os.sep}"
    IMG_WIDTH = 20
    IMG_HEIGHT = 20
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
    vocab, keypoints = build_vocab(X_train)
    
    # Feature extraction
    # sift = SIFT()
    
    # for img in X_train:
    #     keypoints, descriptor = sift.computeKeypointsAndDescriptors(img)
    #     for keypoint in keypoints: 
    #         pass
    # dist = np.linalg.norm(a-b)

    # Training and Testing the Model
    model = SVM()
    model.train(X_train, y_train)
    model.test(X_test, y_test)



if __name__=="__main__":
    main()
