import csv
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import LabelPropagation

class Pipeline:

    def __init__(self, fileName):
        self.data, self.labels = self.readData(fileName)


    def readData(self, fileName):
        data = []
        labels = []
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if (line_count > 0):
                    data.append(row[0:-1])
                    labels.append(row[-1])
                line_count += 1

        data = np.array(data).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        return data, labels

    
    def splitData(self, training_portion, validation_portion, unlabeled_portion):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, stratify=self.labels, train_size=training_portion, random_state=15)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, stratify=self.y_train, test_size=validation_portion, random_state=95)
        self.X_train_lab, self.X_train_unlab, self.y_train_lab, self.y_train_unlab = train_test_split(self.X_train, self.y_train, stratify=self.y_train, test_size=unlabeled_portion, random_state=29)

        np.save('./Split Data/inputData.npy', self.data)
        np.save('./Split Data/labels.npy', self.labels)

        np.save('./Split Data/X_test.npy', self.X_test)
        np.save('./Split Data/y_test.npy', self.y_test)

        np.save('./Split Data/X_train_lab.npy', self.X_train_lab)
        np.save('./Split Data/y_train_lab.npy', self.y_train_lab)

        np.save('./Split Data/X_train_unlab.npy', self.X_train_unlab)
        np.save('./Split Data/y_train_unlab.npy', self.y_train_unlab)

        np.save('./Split Data/X_val.npy', self.X_val)
        np.save('./Split Data/y_val.npy', self.y_val)




if __name__=="__main__":
    # Read in the credit card data.
    rawCreditcardDataFile = '../Data/creditcard.csv'
    pipeline = Pipeline(rawCreditcardDataFile)

    # Split the data into 80% training and 20% testing data.
    # Of all the training data, split 10% for cross-validation.
    # Of the remaining training data, split 70% into unlabeled training data.
    training_portion = 0.8
    validation_portion = 0.1
    unlabeled_portion = 0.7
    pipeline.splitData(training_portion, validation_portion, unlabeled_portion)

    # Set the minimum and maximum values for K in the hyperparameter search.
    min_k = 1
    max_k = 9

    # Perform cross-validation to select the best number of neighbors K for the baseline KNN model
    print("Now training and testing the baseline KNN model ...")
    best_knn_model = None
    best_knn_model_val_acc = -1
    best_k = -1
    for k in range(min_k, max_k+1, 2):
        print(f"Cross-validating k = {k} ...")
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(pipeline.X_train_lab, pipeline.y_train_lab)
        acc = knn_model.score(pipeline.X_val, pipeline.y_val)
        if (acc > best_knn_model_val_acc):
            best_knn_model_val_acc = acc
            best_knn_model = knn_model
            best_k = k

    # Test the best model on the test data, report the metrics, and save the test predictions.
    best_knn_model_test_predictions = best_knn_model.predict(pipeline.X_test)
    print(f"Test accuracy of baseline KNN model: {accuracy_score(pipeline.y_test, best_knn_model_test_predictions)}")
    print(f"Test F1-score of baseline KNN model: {f1_score(pipeline.y_test, best_knn_model_test_predictions)}\n\n")
    np.save('./Model Predictions/knn_model_test_predictions.npy', best_knn_model_test_predictions)



    # Perform cross-validation to select the best number of neighbors K for the KNN label propagation model.
    print("Now training and testing the KNN label propagation model ...")
    X_lab_and_unlab = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y_lab_and_unlab = np.append(pipeline.y_train_lab, np.full_like(pipeline.y_train_unlab, -1))
    best_lp_model = None
    best_lp_model_val_acc = -1
    best_k = -1
    for k in range(min_k, max_k+1, 2):
        print(f"Cross-validating k = {k} ...")
        lp_model = LabelPropagation(kernel = 'knn', n_neighbors = k)
        lp_model.fit(X_lab_and_unlab, y_lab_and_unlab)
        acc = lp_model.score(pipeline.X_val, pipeline.y_val)
        if (acc > best_lp_model_val_acc):
                    best_lp_model_val_acc = acc
                    best_lp_model = lp_model
                    best_k = k

    # Test the best model on the test data, report the metrics, and save the test predictions.
    # Also save the predicted labels for the unlabeled data.
    predicted_labels = best_lp_model.transduction_[-len(pipeline.y_train_unlab):]
    np.save('./Model Predictions/predicted_labels.npy', predicted_labels)
    best_lp_model_test_predictions = best_lp_model.predict(pipeline.X_test)
    print(f"Test accuracy of LabelPropagation model: {accuracy_score(pipeline.y_test, best_lp_model_test_predictions)}")
    print(f"Test F1-score of LabelPropagation model: {f1_score(pipeline.y_test, best_lp_model_test_predictions)}")
    np.save('./Model Predictions/lp_model_test_predictions.npy', best_lp_model_test_predictions)



    # Perform cross-validation to select the best number of neighbors K for the KNN model trained on the extra data.
    print("Now training and testing a baseline KNN model, with extra training data supplied by the semi-supervised model ...")
    X_train_all = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y_train_all = np.append(pipeline.y_train_lab, predicted_labels)
    best_knn_model2 = None
    best_knn_model2_val_acc = -1
    best_k = -1
    for k in range(min_k, max_k+1, 2):
        print(f"Cross-validating k = {k} ...")
        knn_model2 = KNeighborsClassifier()
        knn_model2.fit(X_train_all, y_train_all)
        acc = knn_model2.score(pipeline.X_val, pipeline.y_val)
        if (acc > best_knn_model2_val_acc):
            best_knn_model2_val_acc = acc
            best_knn_model2 = knn_model2
            best_k = k

    # Test the best model on the test data, report the metrics, and save the test predictions.
    best_knn_model2_test_predictions = best_knn_model2.predict(pipeline.X_test)
    print(f"Test accuracy of KNN model trained on more data: {accuracy_score(pipeline.y_test, best_knn_model2_test_predictions)}")
    print(f"Test F1-score of KNN model trained on more data: {f1_score(pipeline.y_test, best_knn_model2_test_predictions)}")
    np.save('./Model Predictions/knn_model2_test_predictions.npy', best_knn_model2_test_predictions)