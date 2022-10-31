import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, confusion_matrix


class DataHandler:
    def __init__(self):
        self.img_data = []
        self.class_labels = []

    def load_data(self, img_folder):
        """Loads images from given class folders"""
        for class_label in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, class_label)):
                image_path = os.path.join(img_folder, class_label,  file)
                image = cv2.imread(image_path)
                self.img_data.append(image)
                self.class_labels.append(class_label)

    def show_example_images(self):
        """Shows five random images out of the dataset"""
        plt.figure(figsize=(20, 6))
        for i in range(5):
            idx = random.choice(range(len(self.img_data)))
            ax = plt.subplot(1, 5, i+1)
            ax.title.set_text(self.class_labels[idx])
            plt.imshow(self.img_data[idx])
        plt.savefig("Figures/ExampleImages.png")

    def plot_class_distribution(self):
        """Plots a histogram containing the class distribution"""
        counts = Counter(self.class_labels)
        ticks = range(len(counts))
        df = pd.DataFrame.from_dict(counts, orient='index')
        df.plot(kind='barh', legend=False,
                title='Class distribution of BigCats dataset')
        plt.savefig("Figures/ClassDistribution.pdf")

    def convert_to_greyscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_salt_and_pepper_noise(self, image, prob):
        """Applies salt and pepper noise to an image."""
        output = np.copy(np.array(image))

        # add salt
        nb_salt = np.ceil(prob * output.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(nb_salt))
                  for i in output.shape]
        output[coords] = 255

        # add pepper
        nb_pepper = np.ceil(prob * output.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(nb_pepper))
                  for i in output.shape]
        output[coords] = 0
        return output

    def apply_salt_noise(self, image, prob):
        """Applies salt noise to an image"""
        output = np.copy(np.array(image))
        # add salt
        nb_salt = np.ceil(prob * output.size)
        coords = [np.random.randint(0, i - 1, int(nb_salt))
                  for i in output.shape]
        output[coords] = 255
        cv2.imshow(output)
        cv2.waitKey(0)
        return output

    def preprocess_data(self, data_type):
        """Converts images to right format for SIFT and optionally applies augmentation"""
        if data_type == "reduced" or data_type == "augmented":
            images = np.array([self.convert_to_greyscale(img)
                               for img in self.img_data])
        else:
            images = np.array([cv2.resize(img, (200, 200)).flatten()
                              for img in self.img_data])
        return images

    def augment_data(self, images):
        augmented = np.array(
            [self.apply_salt_and_pepper_noise(image, 0.5) for image in images])
        return augmented

    def plot_confusion_matrix(self, model_type, data_type, predictions, true_labels, labelNames=["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]):
        """Plot a confusion matrix based on the made predictions and save it according to model and data type."""
        title = "Confusion matrix " + model_type
        fileName = "BigCatsConfusionMatrix"+model_type

        if data_type == "augmented":
            title += " on the augmented"
            fileName += "Augmented"
        elif data_type == "reduced":
            title += " reduced data set"
            fileName += "sift"
        else:
            title += " unreduced data set"
            fileName += "Original"

        confusionMatrix = confusion_matrix(
            predictions, true_labels)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(confusionMatrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confusionMatrix.shape[0]):
            for j in range(confusionMatrix.shape[1]):
                ax.text(
                    x=j, y=i, s=confusionMatrix[i, j], va='center', ha='center', size='xx-large')
        plt.xticks([0, 1, 2, 3, 4], labelNames)
        plt.yticks([0, 1, 2, 3, 4], labelNames)

        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.title(title, y=1.07)
        figureName = f"Figures{os.sep}"+fileName+model_type
        plt.savefig(figureName, dpi=300, bbox_inches='tight')
