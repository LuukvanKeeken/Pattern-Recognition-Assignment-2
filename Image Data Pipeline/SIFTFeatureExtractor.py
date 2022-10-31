from sklearn.cluster import KMeans
import cv2
import os
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy


class SIFTFeatureExtractor():
    def __init__(self):
        pass

    def extract_features(self, X_train, y_train, n_keypoints, n_clusters, X_test=None, y_test=None):
        """Perform the full SIFT bag of visual words algorithm"""
        descriptors_train = self.build_descriptors(
            X_train, n_keypoints, "descriptors_train.npy")
        vocab = self.build_vocab(descriptors_train, n_clusters)
        train_histograms = self.build_histograms(
            descriptors_train, y_train, n_keypoints, vocab, filename="histograms_train")
        if X_test and y_test:
            descriptors_test = self.build_descriptors(
                X_test, n_keypoints, "descriptors_test.npy")
            test_histograms = self.build_histograms(
                descriptors_test, y_test, n_keypoints, vocab, filename="histograms_test")
            return train_histograms, test_histograms
        return train_histograms, None

    def build_vocab(self, descriptors, n_clusters):
        """Cluster the descriptors to get a vocabulary of n words"""
        k_means = KMeans(n_clusters=n_clusters,
                         random_state=0).fit(descriptors)
        return k_means

    def plot_keypoints(self, img, keypoints):
        """Plot the SIFT keypoints in an image"""
        for kp in keypoints:
            img = cv2.drawMarker(img,
                                 (int(kp[0].pt[0]), int(kp[0].pt[1])),
                                 (0, 0, 255),
                                 markerType=cv2.MARKER_SQUARE,
                                 markerSize=30,
                                 thickness=2,
                                 line_type=cv2.LINE_AA)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def build_histograms(self, descriptors, y, N_KEYPOINTS, vocab, filename="histograms.npy"):
        """Build histograms based on the computed vocabulary and the distance of image descriptors to that vocabulary"""
        histograms = []
        # Reshape back into per image
        descriptors = descriptors.reshape((len(y), N_KEYPOINTS, 128))
        for descriptors_img, label in zip(descriptors, y):
            distances = cdist(descriptors_img, vocab.cluster_centers_)
            bin_numbers = np.argmin(distances, axis=1)
            histogram = (np.zeros(len(vocab.cluster_centers_)), label)
            for bin_number in bin_numbers:
                histogram[0][bin_number] += 1
            histograms.append(histogram)
        normalize([hist[0] for hist in histograms], axis=1, norm='l1')
        np.save(f'saves{os.sep}{filename}', histograms)
        return histograms

    def build_descriptors(self, images, N_KEYPOINTS, filename):
        """Run SIFT to compute the local features of an image"""
        sift = cv2.SIFT_create()
        descriptors = []
        for i, img in enumerate(images):
            print(f"{i}/{len(images)}")
            keypoints, descriptor = sift.detectAndCompute(img, None)
            keypoints = np.array(keypoints)
            zipkeydesc = zip(keypoints, descriptor)
            zipkeydesc = sorted(
                zipkeydesc, key=lambda x: x[0].response, reverse=True)
            descriptor = [keydesc[1] for keydesc in zipkeydesc[0:N_KEYPOINTS]]
            descriptors.append(descriptor)
        descriptors = np.array([el for i in descriptors for el in i])
        np.save(f'saves{os.sep}{filename}', descriptors)
        return descriptors
