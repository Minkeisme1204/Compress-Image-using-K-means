import sys 
import os 
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('/home/minkescanor/Desktop/WORKPLACE/Personal/Compress-Image-using-K-means/')
from utils.k_means import K_means_model
from utils.extract_colors import read_img
data_path = '/home/minkescanor/Desktop/WORKPLACE/Personal/Compress-Image-using-K-means/dataset'
epsilon = 0.000001

if __name__ == '__main__':
    k_clusters = []

    list_data_img = os.listdir(data_path)
    for img in list_data_img:
        img_path = os.path.join(data_path, img)
        img_data, original_shape = read_img(img_path)
        lost = 0
        print(img_data.shape)
        best_model = any
        index = []
        for i in range(16, 255*255*255 + 1):
            model = K_means_model(input_data=img_data, k=i)
            temp = model.run()
            idx = temp[0]

            c = model.cost_function(idx)
            k_clusters.append(temp[1])
            if (c - lost < epsilon):
                print("Model Converged Successfully") 
                best_model = model
                index = idx
                break
            else:
                lost = c
        
            print("cluster {},  loss {}".format(i, c))

        idx = np.array(idx, dtype=int)
        X_recovered = model.centroids[idx, :]
        X_recovered = np.reshape(X_recovered, original_shape) 
        print(X_recovered, X_recovered.shape)
        print(X_recovered)

        fig, axes = plt.subplots(1, 2, figsize=(16, 16))

        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Display original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Display compressed image
        axes[1].imshow(X_recovered)
        axes[1].set_title('Compressed with %d colours'%best_model.K)
        axes[0].axis('off')

        plt.show()