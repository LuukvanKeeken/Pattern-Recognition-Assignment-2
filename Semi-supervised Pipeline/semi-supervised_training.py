from sklearn.semi_supervised import LabelPropagation
import numpy as np
import pickle

lp_model = LabelPropagation(kernel='knn')

X_train_lab = np.load('./Split Data/X_train_lab.npy')
y_train_lab = np.load('./Split Data/y_train_lab.npy')
X_train_unlab = np.load('./Split Data/X_train_unlab.npy')
y_train_unlab = np.load('./Split Data/y_train_unlab.npy')
y_train_unlab_labels_removed = np.full_like(y_train_unlab, -1)
X_test = np.load('./Split Data/X_test.npy')
y_test = np.load('./Split Data/y_test.npy')

X = np.append(X_train_lab, X_train_unlab, axis=0)
y = np.append(y_train_lab, y_train_unlab_labels_removed)

lp_model.fit(X, y)

predicted_labels = lp_model.transduction_[-len(y_train_unlab):]
np.save('./Split Data/predicted_labels.npy', predicted_labels)

print(lp_model.score(X_test, y_test))

pickle.dump(lp_model, open('trained_LP_model.sav', 'wb'))