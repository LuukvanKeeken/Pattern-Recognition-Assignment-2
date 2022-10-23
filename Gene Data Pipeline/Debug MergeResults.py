import numpy as np
import pickle

GridSearchClassifiersFile = './Gene Data Pipeline/Data/ClassifiersGridSearch.npy'
classifiers1_2 =  './Gene Data Pipeline/Data/Classifiers1_2GridSearch.npy'
classfier3 = './Gene Data Pipeline/Data/AdditionalClassifierGridSearch.npy'


with open (classifiers1_2, 'rb') as fp:
    array1 = np.array(pickle.load(fp))

with open (classfier3, 'rb') as fp:
    array2 = np.array(pickle.load(fp))

newResults = []
for row in array1:
    components = row[0]
    indices2 = np.where(array2[:,0]==components)[0]
    if len(indices2) != 1:
        kjkkljln        
    row2 = array2[indices2[0]]
    addRow = np.delete(row2, 0)
    mergedRow = np.append(row, addRow)
    newResults.append(mergedRow)

with open(GridSearchClassifiersFile, 'wb') as fp:
            pickle.dump(newResults, fp)