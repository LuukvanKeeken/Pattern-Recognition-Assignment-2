from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X_train_lab = np.load('./Split Data/X_train_lab.npy')
y_train_lab = np.load('./Split Data/y_train_lab.npy')
X_train_unlab = np.load('./Split Data/X_train_unlab.npy')
predicted_labels = np.load('./Split Data/predicted_labels.npy')
X_test = np.load('./Split Data/X_test.npy')
y_test = np.load('./Split Data/y_test.npy')


X_train_all = np.append(X_train_lab, X_train_unlab, axis=0)
y_train_all = np.append(y_train_lab, predicted_labels)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train_all, y_train_all)

print(knn_model.score(X_test, y_test))