import os
import numpy as np
# assign directory
directory = './Semi-supervised Pipeline/IndividualAccuracies and F1s/'
 


knn_model_f1s = []
knn_model_accs = []
knn_model2_f1s = []
knn_model2_accs = []
lp_model_f1s = []
lp_model_accs = []


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    filename = directory+filename
    if os.path.isfile(f):
        if "knn_model_f1s" in filename:
            model = np.load(filename)
            if len(model)==1:
                knn_model_f1s.append(model[0])
            else:
                if len(model)==9:
                    model = model[0:8]
                knn_model_f1s.extend(model)
        if "knn_model_accs" in filename:
            model = np.load(filename)
            if len(model)==1:
                knn_model_accs.append(model[0])
            else:
                if len(model)==9:
                    model = model[0:8]
                knn_model_accs.extend(model)
        if "knn_model2_accs" in filename:
            model = np.load(filename)
            if len(model)==1:
                knn_model2_accs.append(model[0])
            else:
                knn_model2_accs.extend(model)
        if "knn_model2_f1s" in filename:
            model = np.load(filename)
            if len(model)==1:
                knn_model2_f1s.append(model[0])
            else:
                knn_model2_f1s.extend(model)
        if "lp_model_accs" in filename:
            model = np.load(filename)
            if len(model)==1:
                lp_model_accs.append(model[0])
            else:
                lp_model_accs.extend(model)
        if "lp_model_f1s" in filename:
            model = np.load(filename)
            if len(model)==1:
                lp_model_f1s.append(model[0])
            else:
                lp_model_f1s.extend(model)

np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model_f1s.npy', knn_model_f1s)
np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model_accs.npy', knn_model_accs)
np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model2_f1s.npy', knn_model2_f1s)
np.save('./Semi-supervised Pipeline/Accuracies and F1s/knn_model2_accs.npy', knn_model2_accs)
np.save('./Semi-supervised Pipeline/Accuracies and F1s/lp_model_f1s.npy', lp_model_f1s)
np.save('./Semi-supervised Pipeline/Accuracies and F1s/lp_model_accs.npy', lp_model_accs)
        
print()