from asyncio.windows_utils import pipe
import csv
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import LabelPropagation
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb

splitDataFilePath = 'Semi-supervised Pipeline/Split Data/'
accsAndF1sFilePath = 'Semi-supervised Pipeline/Accuracies and F1s/'
modelPredictionsFilePath = 'Semi-supervised Pipeline/Model Predictions/'
finalResultsFilePath = 'Semi-supervised Pipeline/Final Results/'
rawCreditcardDataFile = './Data/creditcard.csv'

class Pipeline:

    def __init__(self, fileName):
        self.data, self.labels = self.readData(fileName)

    # Calculate means and standard deviations of accuracies and F1-scores.
    # Also plot the accuracies over the iterations and create confusion
    # matrices for the last iteration.
    def analyseData(self, knn1_accs, knn1_f1s, lp_accs, lp_f1s, knn2_accs, knn2_f1s, knn_model_test_predictions, lp_model_test_predictions, knn_model_2_test_predictions):
        knn1_avg_acc = np.mean(knn1_accs)
        knn1_stddev_acc = np.std(knn1_accs)
        knn1_avg_f1 = np.mean(knn1_f1s)
        knn1_stddev_f1 = np.std(knn1_f1s)
        lp_avg_acc = np.mean(lp_accs)
        lp_stddev_acc = np.std(lp_accs)
        lp_avg_f1 = np.mean(lp_f1s)
        lp_stddev_f1 = np.std(lp_f1s)
        knn2_avg_acc = np.mean(knn2_accs)
        knn2_stddev_acc = np.std(knn2_accs)
        knn2_avg_f1 = np.mean(knn2_f1s)
        knn2_stddev_f1 = np.std(knn2_f1s)

        print(f"First KNN model:")
        print(f"Mean accuracy: {knn1_avg_acc} ({round(knn1_avg_acc*len(self.X_test), 3)} out of {len(self.X_test)}), standard deviation: {knn1_stddev_acc}")
        print(f"Mean F1-score: {knn1_avg_f1}, standard deviation: {knn1_stddev_f1}\n")

        print(f"Label propagation model:")
        print(f"Mean accuracy: {lp_avg_acc} ({round(lp_avg_acc*len(self.X_test), 3)} out of {len(self.X_test)}), standard deviation: {lp_stddev_acc}")
        print(f"Mean F1-score: {lp_avg_f1}, standard deviation: {lp_stddev_f1}\n")

        print(f"Second KNN model:")
        print(f"Mean accuracy: {knn2_avg_acc} ({round(knn2_avg_acc*len(self.X_test), 3)} out of {len(self.X_test)}), standard deviation: {knn2_stddev_acc}")
        print(f"Mean F1-score: {knn2_avg_f1}, standard deviation: {knn2_stddev_f1}\n")

        iterations = np.arange(1, len(knn1_accs)+1, 1)
        plt.plot(iterations, knn_model_accs, label = 'First KNN')
        plt.plot(iterations, lp_model_accs, label = 'Label propagation')
        plt.plot(iterations, knn_model2_accs, label = 'Second KNN')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.title(f'Accuracies of the three model for {len(knn1_accs)} iterations')
        plt.savefig(finalResultsFilePath + 'iterations_plot.png')

        knn_conf_mat = confusion_matrix(knn_model_test_predictions, self.y_test)
        plt.figure()
        sb.heatmap(knn_conf_mat, annot=True, xticklabels=['No fraud', 'Fraud'], yticklabels=['No fraud', 'Fraud'])
        plt.title("Confusion matrix for first KNN model")
        plt.savefig(finalResultsFilePath + 'knn1_cm.png')

        lp_conf_mat = confusion_matrix(lp_model_test_predictions, self.y_test)
        plt.figure()
        sb.heatmap(lp_conf_mat, annot=True, xticklabels=['No fraud', 'Fraud'], yticklabels=['No fraud', 'Fraud'])
        plt.title("Confusion matrix for LP model")
        plt.savefig(finalResultsFilePath + 'lp_cm.png')

        knn2_conf_mat = confusion_matrix(knn_model_2_test_predictions, self.y_test)
        plt.figure()
        sb.heatmap(knn2_conf_mat, annot=True, xticklabels=['No fraud', 'Fraud'], yticklabels=['No fraud', 'Fraud'])
        plt.title("Confusion matrix for second KNN model")
        plt.savefig(finalResultsFilePath + 'knn2_cm.png')




    def readData(self, fileName):
        data = []
        labels = []
        with open(fileName) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if (line_count > 0):
                    data.append(row[1:-2])
                    labels.append(row[-1])
                line_count += 1

        data = np.array(data).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        return data, labels

    
    def splitData(self, training_portion, unlabeled_portion):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, stratify=self.labels, train_size=training_portion)
        self.X_train_lab, self.X_train_unlab, self.y_train_lab, self.y_train_unlab = train_test_split(self.X_train, self.y_train, stratify=self.y_train, test_size=unlabeled_portion)

        np.save(splitDataFilePath + 'inputData.npy', self.data)
        np.save(splitDataFilePath + 'labels.npy', self.labels)

        np.save(splitDataFilePath + 'X_test.npy', self.X_test)
        np.save(splitDataFilePath + 'y_test.npy', self.y_test)

        np.save(splitDataFilePath + 'X_train_lab.npy', self.X_train_lab)
        np.save(splitDataFilePath + 'y_train_lab.npy', self.y_train_lab)

        np.save(splitDataFilePath + 'X_train_unlab.npy', self.X_train_unlab)
        np.save(splitDataFilePath + 'y_train_unlab.npy', self.y_train_unlab)



if __name__=="__main__":
    if (not os.path.exists(splitDataFilePath)):
        os.mkdir(splitDataFilePath)
    if (not os.path.exists(accsAndF1sFilePath)):
        os.mkdir(accsAndF1sFilePath)
    if (not os.path.exists(modelPredictionsFilePath)):
        os.mkdir(modelPredictionsFilePath)
    if (not os.path.exists(finalResultsFilePath)):
        os.mkdir(finalResultsFilePath)
    
    knn_model_accs = []
    knn_model_f1s = []
    lp_model_accs = []
    lp_model_f1s = []
    knn_model2_accs = []
    knn_model2_f1s = []

    print("Now reading in the data ...")
    pipeline = Pipeline(rawCreditcardDataFile)
    
    # The accuracies and F1-scores are saved to file each round,
    # in order to not lose collected data when the script stops
    # for some reason.
    number_of_rounds = 100
    for i in range(number_of_rounds):
        print(f"Round {i+1} out of {number_of_rounds}")
        
        
        # Split the data into 80% training and 20% testing data.
        # Of the remaining training data, split 70% into unlabeled training data.
        print("Now splitting the data ...")
        training_portion = 0.8
        unlabeled_portion = 0.7
        pipeline.splitData(training_portion, unlabeled_portion)

        
        # Fit a simple KNN model to the labelled training data and test it on the test data
        print(f"Now training and testing first KNN model ...")
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(pipeline.X_train_lab, pipeline.y_train_lab)
        knn_model_test_predictions = knn_model.predict(pipeline.X_test)
        # Only save the test predictions of the last round to file
        if (i == number_of_rounds - 1):
            np.save(modelPredictionsFilePath + 'knn_model_test_predictions.npy', knn_model_test_predictions)

        knn_model_test_acc = accuracy_score(pipeline.y_test, knn_model_test_predictions)
        knn_model_test_f1 = f1_score(pipeline.y_test, knn_model_test_predictions)
        knn_model_accs.append(knn_model_test_acc)
        knn_model_f1s.append(knn_model_test_f1)
        print(f"Test accuracy of baseline KNN model: {knn_model_test_acc}")
        print(f"Test F1-score of baseline KNN model: {knn_model_test_f1}\n\n")
        np.save(accsAndF1sFilePath + 'knn_model_accs.npy', knn_model_accs)
        np.save(accsAndF1sFilePath + 'knn_model_f1s.npy', knn_model_f1s)


        # Fit a KNN-based Label Propagation model to all the training data and test it on the test data.
        # Then save its predictions for the unlabelled training data.
        print(f"Now training and testing Label Propagation model ...")
        X_lab_and_unlab = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
        y_lab_and_unlab = np.append(pipeline.y_train_lab, np.full_like(pipeline.y_train_unlab, -1))
        lp_model = LabelPropagation(kernel = 'knn', n_neighbors = 5)
        lp_model.fit(X_lab_and_unlab, y_lab_and_unlab)
        lp_model_test_predictions = lp_model.predict(pipeline.X_test)
        # Only save the test predictions of the last round to file
        if (i == number_of_rounds - 1):
            np.save(modelPredictionsFilePath + 'lp_model_test_predictions.npy', lp_model_test_predictions)
        lp_model_test_acc = accuracy_score(pipeline.y_test, lp_model_test_predictions)
        lp_model_test_f1 = f1_score(pipeline.y_test, lp_model_test_predictions)
        print(f"Test accuracy of Label Propagation model: {lp_model_test_acc}")
        print(f"Test F1-score of Label Propagation model: {lp_model_test_f1}\n\n")
        lp_model_accs.append(lp_model_test_acc)
        lp_model_f1s.append(lp_model_test_f1)
        predicted_labels = lp_model.transduction_[-len(pipeline.y_train_unlab):]
        np.save(accsAndF1sFilePath + 'lp_model_accs.npy', lp_model_accs)
        np.save(accsAndF1sFilePath + 'lp_model_f1s.npy', lp_model_f1s)


        # Fit a simple KNN model to the labelled training data plus the predicted labels of the label propagation model and test it on the test data.
        print(f"Now training and testing second KNN model ...")
        X_train_all = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
        y_train_all = np.append(pipeline.y_train_lab, predicted_labels)
        knn_model_2 = KNeighborsClassifier(n_neighbors = 5)
        knn_model_2.fit(X_train_all, y_train_all)
        knn_model_2_test_predictions = knn_model_2.predict(pipeline.X_test)
        # Only save the test predictions of the last round to file
        if (i == number_of_rounds - 1):
            np.save(modelPredictionsFilePath + 'knn_model2_test_predictions.npy', knn_model_2_test_predictions)
        knn_model_2_test_acc = accuracy_score(pipeline.y_test, knn_model_2_test_predictions)
        knn_model_2_test_f1 = f1_score(pipeline.y_test, knn_model_2_test_predictions)
        knn_model2_accs.append(knn_model_2_test_acc)
        knn_model2_f1s.append(knn_model_2_test_f1)
        np.save(accsAndF1sFilePath + 'knn_model2_accs.npy', knn_model2_accs)
        np.save(accsAndF1sFilePath + 'knn_model2_f1s.npy', knn_model2_f1s)
        print(f"Test accuracy of KNN model trained on more data: {knn_model_2_test_acc}")
        print(f"Test F1-score of KNN model trained on more data: {knn_model_2_test_f1}\n\n")

    pipeline.analyseData(knn_model_accs, knn_model_f1s, lp_model_accs, lp_model_f1s, knn_model2_accs, knn_model2_f1s, knn_model_test_predictions, lp_model_test_predictions, knn_model_2_test_predictions)





