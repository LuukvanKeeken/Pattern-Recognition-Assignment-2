import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random


class DataHandler:

    def __init__(self):
        self.img_data = []
        self.class_labels = []

    def loadData(self, img_folder, IMG_HEIGHT, IMG_WIDTH):
   
    
        for class_label in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, class_label)):
        
                image_path= os.path.join(img_folder, class_label,  file)
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA) # resize images to make it uniform
                image=np.array(image)
                image = image.astype('float32')
                image /= 255 # scale down images from 0-255 to 0-1 for better convergence

                self.img_data.append(image)
                self.class_labels.append(class_label)


    def showExampleImages(self):
        # Show randomly 5 images

        plt.figure(figsize=(20,6))

        for i in range(5):
            idx = random.choice(range(len(self.img_data)))
            ax = plt.subplot(1, 5, i+1)
            ax.title.set_text(self.class_labels[idx])
            plt.imshow(self.img_data[idx])

    def showClassDistribution(self):
        plt.hist(self.class_labels)
        plt.show()

    def convertLabelsToNumeric(self):
        # Convert class labels to numeric values
        target_dict = {k: v for v, k in enumerate(np.unique(self.class_labels))}
   
        # Convert all labels in the dataset to numeric values
        self.class_labels =  [target_dict[self.class_labels[i]] for i in range(len(self.class_labels))]
        return target_dict

    def preprocessData(self):
        # For now only flatten the images
        images = np.array(self.img_data)
        self.img_data = images.reshape((len(self.img_data), -1))
        return images