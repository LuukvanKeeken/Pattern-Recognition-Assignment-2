
from sklearn.model_selection import train_test_split
from DataHandler import DataHandler
from SIFTFeatureExtractor import SIFTFeatureExtractor
import os
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree, ensemble
from sklearn.metrics import f1_score, silhouette_score
from sklearn.cluster import KMeans
from hyperparameters import hyperparameters


def main():
    data_type = sys.argv[1]
    model_type = sys.argv[2]

    # Load and process Data
    data_dir = f"BigCats{os.sep}"
    dh = DataHandler()
    dh.load_data(data_dir)
    dh.plot_class_distribution()
    images = dh.preprocess_data(data_type)

    # Make Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
        images, dh.class_labels, test_size=0.2, shuffle=True, random_state=0
    )

    # Augment Data
    if data_type == "augmented":
        X_train = np.append(X_train, dh.augment_data(X_train), axis=0)
        y_train += y_train

    # Run feature reduction and selection
    if data_type != "unreduced":
        feature_extractor = SIFTFeatureExtractor()
        descriptors_train = feature_extractor.build_descriptors(
            X_train, hyperparameters[data_type]['keypoints'][model_type], "descriptors_train.npy")
        descriptors_test = feature_extractor.build_descriptors(
            X_test, hyperparameters[data_type]['keypoints'][model_type], "descriptors_test.npy")

        vocab = feature_extractor.build_vocab(
            descriptors_train, hyperparameters[data_type]['clusters_1'][model_type])
        train_histograms = feature_extractor.build_histograms(
            descriptors_train, y_train, hyperparameters[data_type]['keypoints'][model_type], vocab)
        test_histograms = feature_extractor.build_histograms(
            descriptors_test, y_test, hyperparameters[data_type]['keypoints'][model_type], vocab, filename="histograms_test")

        X_train, y_train = ([hist[0] for hist in train_histograms], [hist[1]
                                                                     for hist in train_histograms])
        X_test, y_test = ([hist[0] for hist in test_histograms], [hist[1]
                                                                  for hist in test_histograms])

    # Training and Testing the Models with optimal params
    if model_type == "svm":
        svm_model = svm.SVC(C=hyperparameters[data_type]['C'],
                            kernel=hyperparameters[data_type]['kernel'])
        svm_model.fit(X_train, y_train)
        print(f"Accuracy is: {svm_model.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(y_test, svm_model.predict(X_test), average='macro')}")

        dh.plot_confusion_matrix(model_type, data_type,
                                 svm_model.predict(X_test), y_test)
    elif model_type == "rf":
        rf = ensemble.RandomForestClassifier(
            n_estimators=hyperparameters[data_type]["estimators"], criterion=hyperparameters[data_type]['rf_metric'], random_state=0)
        rf.fit(X_train, y_train)
        print(f"Accuracy is: {rf.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(rf.predict(X_test), y_test, average='macro')}")
        dh.plot_confusion_matrix(model_type, data_type,
                                 rf.predict(X_test), y_test)
    elif model_type == "knn":
        knn = KNeighborsClassifier(
            n_neighbors=hyperparameters[data_type]['K'], weights=hyperparameters[data_type]["weights"])
        knn.fit(X_train, y_train)
        print(f"Accuracy is: {knn.score(X_test, y_test)}")
        print(
            f"F1 is: {f1_score(knn.predict(X_test), y_test, average='macro')}")

        dh.plot_confusion_matrix(model_type, data_type,
                                 knn.predict(X_test), y_test)
    elif model_type == "km":
        km = KMeans(n_clusters=hyperparameters[data_type]['clusters_2'],
                    random_state=0)
        X = np.append(X_train, X_test, axis=0)
        km.fit_predict(X)
        print(
            f"Silhouette score is: {silhouette_score(X, km.labels_, metric='euclidean')}")


if __name__ == "__main__":
    main()
