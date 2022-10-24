from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train_lab = np.load('./Split Data/X_train_lab.npy')
y_train_lab = np.load('./Split Data/y_train_lab.npy')
X_test = np.load('./Split Data/X_test.npy')
y_test = np.load('./Split Data/y_test.npy')

knn_model = KNeighborsClassifier()
knn_model.fit(X_train_lab, y_train_lab)

print(knn_model.score(X_test, y_test))