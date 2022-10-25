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
                image= cv2.imread(image_path)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA) # resize images to make it uniform
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image=np.array(image)
                image = image.astype('float32')
                #image /= 255 # scale down images from 0-255 to 0-1 for better convergence (doesnt work with sift)

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
        
        plt.savefig("Figures/ExampleImages.png")

    def showClassDistribution(self):
        plt.hist(self.class_labels)
        plt.xticks(self.class_labels, fontsize=12)
        plt.savefig("Figures/ClassDistribution.png")
        plt.show()

    def convertLabelsToNumeric(self):
        # Convert class labels to numeric values
        target_dict = {k: v for v, k in enumerate(np.unique(self.class_labels))}
   
        # Convert all labels in the dataset to numeric values
        self.class_labels =  [target_dict[self.class_labels[i]] for i in range(len(self.class_labels))]
        return target_dict

    
    def convert_to_greyscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def apply_salt_and_pepper_noise(self, image, prob):
        """Applies salt and pepper noise to an image."""
        output = np.copy(np.array(image))

        # add salt
        nb_salt = np.ceil(prob * output.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
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
        print(output)
        # add salt
        nb_salt = np.ceil(prob * output.size)
        coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
        output[coords] = 255

        return output


    def preprocessData(self):
        # For now only flatten the images
        images = np.array(self.img_data)
        #images = []
        # Optional fourier transform:
        #for i in self.img_data:
           # images.append(np.fft.fft2(i))
        
        #images = np.array(images)
        self.img_data = images.reshape((len(self.img_data), -1))
        # salt_and_pepper_images = np.array([self.apply_salt_and_pepper_noise(image, 0.5) for image in images])
        return images
    