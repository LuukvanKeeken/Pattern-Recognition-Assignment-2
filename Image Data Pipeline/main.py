from sklearn.model_selection import train_test_split
from DataHandler import DataHandler
from SIFT import SIFT
from Models import SVM
import cv2
import os

def main():
    
    # Initial settings

    data_dir = f"BigCats{os.sep}"
    IMG_WIDTH = 50
    IMG_HEIGHT = 50
    dh = DataHandler()
    dh.loadData(data_dir, IMG_WIDTH, IMG_HEIGHT)

    dh.showClassDistribution()
    dh.showExampleImages()
    

    target_dict = dh.convertLabelsToNumeric()

    images = dh.preprocessData()

    # Feature extraction
    sift = SIFT()
    vocabulary = []
    
    img = cv2.imread("BigCats/Jaguar/jaguar-859412__340.jpg")
    print(img)
    cv2.imshow("img", img)
    cv2.waitKey(0) 
    for img in images:
        cv2.imshow("img", img)
        cv2.waitKey(0) 
        print(img)
        keypoints, descriptors = sift.computeKeypointsAndDescriptors(img)
        print(keypoints)
    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(
    descriptors, dh.class_labels, test_size=0.2, shuffle=True
    )
    
    # Training and Testing the Model
    model = SVM()
    model.train(X_train, y_train)
    model.test(X_test, y_test)



if __name__=="__main__":
    main()
