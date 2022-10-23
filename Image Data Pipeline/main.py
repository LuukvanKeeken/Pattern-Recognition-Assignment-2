from sklearn.model_selection import train_test_split
from DataHandler import DataHandler
from Models import SVM
import os


def main():
    
    # Initial settings

    data_dir = f"BigCats{os.sep}"
    IMG_WIDTH = 200
    IMG_HEIGHT = 200
    dh = DataHandler()
    dh.loadData(data_dir, IMG_WIDTH, IMG_HEIGHT)

    dh.showClassDistribution()
    dh.showExampleImages()
    

    target_dict = dh.convertLabelsToNumeric()

    dh.preprocessData()

    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
    dh.img_data, dh.class_labels, test_size=0.2, shuffle=True
    )

    # Training and Testing the Model
    model = SVM()
    model.train(X_train, y_train)
    model.test(X_test, y_test)



if __name__=="__main__":
    main()
