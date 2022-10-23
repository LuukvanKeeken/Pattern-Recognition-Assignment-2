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

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels, random_state=15)

X_train_lab, X_train_unlab, y_train_lab, y_train_unlab = train_test_split(X_train, y_train, stratify=y_train, random_state=29)