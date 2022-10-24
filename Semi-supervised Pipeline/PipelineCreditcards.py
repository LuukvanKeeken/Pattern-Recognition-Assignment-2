import csv
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

    
    def splitData(self, training_portion, unlabeled_portion):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, stratify=self.labels, train_size=training_portion, random_state=15)
        self.X_train_lab, self.X_train_unlab, self.y_train_lab, self.y_train_unlab = train_test_split(self.X_train, self.y_train, stratify=self.y_train, train_size=(1-unlabeled_portion), random_state=29)

        np.save('./Split Data/inputData.npy', self.data)
        np.save('./Split Data/labels.npy', self.labels)

        np.save('./Split Data/X_test.npy', self.X_test)
        np.save('./Split Data/y_test.npy', self.y_test)

        np.save('./Split Data/X_train_lab.npy', self.X_train_lab)
        np.save('./Split Data/y_train_lab.npy', self.y_train_lab)

        np.save('./Split Data/X_train_unlab.npy', self.X_train_unlab)
        np.save('./Split Data/y_train_unlab.npy', self.y_train_unlab)



if __name__=="__main__":
    rawCreditcardDataFile = '../Data/creditcard.csv'
    pipeline = Pipeline(rawCreditcardDataFile)
    pipeline.splitData(0.8, 0.7)

    print("Now training and testing the baseline KNN model ...")
    knn_model = KNeighborsClassifier()
    knn_model.fit(pipeline.X_train, pipeline.y_train)
    knn_model_test_predictions = knn_model.predict(pipeline.X_test)
    print(f"Test accuracy of baseline KNN model: {accuracy_score(pipeline.y_test, knn_model_test_predictions)}")
    print(f"Test F1-score of baseline KNN model: {f1_score(pipeline.y_test, knn_model_test_predictions)}\n\n")
    np.save('./Model Predictions/knn_model_test_predictions.npy', knn_model_test_predictions)

    print("Now training and testing the KNN label propagation model ...")
    X = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y = np.append(pipeline.y_train_lab, np.full_like(pipeline.y_train_unlab, -1))
    lp_model = LabelPropagation(kernel = 'knn', n_neighbors = 5)
    lp_model.fit(X, y)
    predicted_labels = lp_model.transduction_[-len(pipeline.y_train_unlab):]
    np.save('./Model Predictions/predicted_labels.npy', predicted_labels)
    lp_model_test_predictions = lp_model.predict(pipeline.X_test)
    print(f"Test accuracy of LabelPropagation model: {accuracy_score(pipeline.y_test, lp_model_test_predictions)}")
    print(f"Test F1-score of LabelPropagation model: {f1_score(pipeline.y_test, lp_model_test_predictions)}")
    np.save('./Model Predictions/lp_model_test_predictions.npy', lp_model_test_predictions)

    print("Now training and testing a baseline KNN model, with extra training data supplied by the semi-supervised model ...")
    X_train_all = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y_train_all = np.append(pipeline.y_train_lab, predicted_labels)
    knn_model2 = KNeighborsClassifier()
    knn_model2.fit(X_train_all, y_train_all)
    knn_model2_test_predictions = knn_model2.predict(pipeline.X_test)
    print(f"Test accuracy of KNN model trained on more data: {accuracy_score(pipeline.y_test, knn_model2_test_predictions)}")
    print(f"Test F1-score of KNN model trained on more data: {f1_score(pipeline.y_test, knn_model2_test_predictions)}")
    np.save('./Model Predictions/knn_model2_test_predictions.npy', knn_model2_test_predictions)