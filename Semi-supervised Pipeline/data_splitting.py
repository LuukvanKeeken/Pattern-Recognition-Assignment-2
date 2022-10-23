import csv
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

rawCreditCardDataFile = '../Data/creditcard.csv'

data = []
labels = []
with open(rawCreditCardDataFile) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if (line_count > 0):
            data.append(row[0:-1])
            labels.append(row[-1])
        line_count += 1


data = np.array(data).astype(np.float32)
labels = np.array(labels).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, train_size=0.8, random_state=15)

X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = train_test_split(X_train, y_train, stratify=y_train, train_size=0.3, random_state=29)

np.save('./Split Data/inputData.npy', data)
np.save('./Split Data/labels.npy', labels)

np.save('./Split Data/X_test.npy', X_test)
np.save('./Split Data/y_test.npy', y_test)

np.save('./Split Data/X_train_lab.npy', X_train_lab)
np.save('./Split Data/y_train_lab.npy', y_train_lab)

np.save('./Split Data/X_train_unlab.npy', X_train_unlab)
np.save('./Split Data/y_train_unlab.npy', y_train_unlab)