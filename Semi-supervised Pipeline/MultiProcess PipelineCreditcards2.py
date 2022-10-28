import csv
from random import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.semi_supervised import LabelPropagation
import time
from multiprocessing import Process

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
                    data.append(row[1:-2])
                    labels.append(row[-1])
                line_count += 1

        data = np.array(data).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        return data, labels

    
    def splitData(self, training_portion, unlabeled_portion):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.labels, stratify=self.labels, train_size=training_portion)
        self.X_train_lab, self.X_train_unlab, self.y_train_lab, self.y_train_unlab = train_test_split(self.X_train, self.y_train, stratify=self.y_train, test_size=unlabeled_portion)

        np.save('./Semi-supervised Pipeline/Split Data/inputData.npy', self.data)
        np.save('./Semi-supervised Pipeline/Split Data/labels.npy', self.labels)

        np.save('./Semi-supervised Pipeline/Split Data/X_test.npy', self.X_test)
        np.save('./Semi-supervised Pipeline/Split Data/y_test.npy', self.y_test)

        np.save('./Semi-supervised Pipeline/Split Data/X_train_lab.npy', self.X_train_lab)
        np.save('./Semi-supervised Pipeline/Split Data/y_train_lab.npy', self.y_train_lab)

        np.save('./Semi-supervised Pipeline/Split Data/X_train_unlab.npy', self.X_train_unlab)
        np.save('./Semi-supervised Pipeline/Split Data/y_train_unlab.npy', self.y_train_unlab)

def task(procedIndex):
    knn_model_accs = []
    knn_model_f1s = []
    lp_model_accs = []
    lp_model_f1s = []
    knn_model2_accs = []
    knn_model2_f1s = []

    rawCreditcardDataFile = './Data/creditcard.csv'
    pipeline = Pipeline(rawCreditcardDataFile)

    #for i in range(10):
    
    #print(f"Round {i+1}")
    #print("process_"+str(procedIndex)+": Now reading in and splitting the data ...")
    
    # Split the data into 80% training and 20% testing data.
    # Of the remaining training data, split 70% into unlabeled training data.
    training_portion = 0.8
    unlabeled_portion = 0.7
    pipeline.splitData(training_portion, unlabeled_portion)
    
    # Fit a simple KNN model to the labelled training data and test it on the test data
    print("process_"+str(procedIndex)+": Now training and testing first KNN model ...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(pipeline.X_train_lab, pipeline.y_train_lab)
    knn_model_test_predictions = knn_model.predict(pipeline.X_test)
    knn_model_test_acc = accuracy_score(pipeline.y_test, knn_model_test_predictions)
    knn_model_test_f1 = f1_score(pipeline.y_test, knn_model_test_predictions)
    knn_model_accs.append(knn_model_test_acc)
    knn_model_f1s.append(knn_model_test_f1)
    #print(f"Test accuracy of baseline KNN model: {knn_model_test_acc}")
    #print(f"Test F1-score of baseline KNN model: {knn_model_test_f1}\n\n")
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model_accs_process_'+str(procedIndex)+'.npy', knn_model_accs)
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model_f1s_process_'+str(procedIndex)+'.npy', knn_model_f1s)

    # Fit a KNN-based Label Propagation model to all the training data and test it on the test data.
    # Then save its predictions for the unlabelled training data.
    print("process_"+str(procedIndex)+": Now training and testing Label Propagation model ...")
    X_lab_and_unlab = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y_lab_and_unlab = np.append(pipeline.y_train_lab, np.full_like(pipeline.y_train_unlab, -1))
    lp_model = LabelPropagation(kernel = 'knn', n_neighbors = 5)
    lp_model.fit(X_lab_and_unlab, y_lab_and_unlab)
    lp_model_test_predictions = lp_model.predict(pipeline.X_test)
    lp_model_test_acc = accuracy_score(pipeline.y_test, lp_model_test_predictions)
    lp_model_test_f1 = f1_score(pipeline.y_test, lp_model_test_predictions)
    #print(f"Test accuracy of Label Propagation model: {lp_model_test_acc}")
    #print(f"Test F1-score of Label Propagation model: {lp_model_test_f1}")
    lp_model_accs.append(lp_model_test_acc)
    lp_model_f1s.append(lp_model_test_f1)
    predicted_labels = lp_model.transduction_[-len(pipeline.y_train_unlab):]
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/lp_model_accs_process_'+str(procedIndex)+'.npy', lp_model_accs)
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/lp_model_f1s_process_'+str(procedIndex)+'.npy', lp_model_f1s)

    # Fit a simple KNN model to the labelled training data plus the predicted labels of the label propagation model and test it on the test data.
    print("process_"+str(procedIndex)+": Now training and testing second KNN model ...")
    X_train_all = np.append(pipeline.X_train_lab, pipeline.X_train_unlab, axis=0)
    y_train_all = np.append(pipeline.y_train_lab, predicted_labels)
    knn_model_2 = KNeighborsClassifier(n_neighbors = 5)
    knn_model_2.fit(X_train_all, y_train_all)
    knn_model_2_test_predictions = knn_model_2.predict(pipeline.X_test)
    knn_model_2_test_acc = accuracy_score(pipeline.y_test, knn_model_2_test_predictions)
    knn_model_2_test_f1 = f1_score(pipeline.y_test, knn_model_2_test_predictions)
    knn_model2_accs.append(knn_model_2_test_acc)
    knn_model2_f1s.append(knn_model_2_test_f1)
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model2_accs_process_'+str(procedIndex)+'.npy', knn_model2_accs)
    np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model2_f1s_process_'+str(procedIndex)+'.npy', knn_model2_f1s)
    #print(f"Test accuracy of KNN model trained on more data: {knn_model_2_test_acc}")
    #print(f"Test F1-score of KNN model trained on more data: {knn_model_2_test_f1}")
                
    

if __name__=="__main__":
    # create all tasks
    start_time = time.time()
    processes = [Process(target=task, args=(i,)) for i in range(65,73)]

    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    print('Done', flush=True)
    print("The search took %s seconds." % round((time.time() - start_time),2))
