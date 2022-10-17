import matplotlib.pyplot as plt
from sklearn import  svm, metrics



class Model:
    
    def __init__(self):
        self.classifier = svm.SVC()

    def train(self, X_train, y_train):
       
        self.classifier.fit(X_train, y_train)

    def test(self, X_test, y_test):
    
        predicted = self.classifier.predict(X_test)

        print(
            f"Classification report for classifier {self.classifier}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

        cm = metrics.confusion_matrix(y_test, predicted, labels=self.classifier.classes_)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=self.classifier.classes_)
        disp.plot()
        print(f"Confusion matrix:\n{disp.confusion_matrix}")

        plt.show()

