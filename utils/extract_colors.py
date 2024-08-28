import numpy as np
import matplotlib.pyplot as plt 
import cv2

filename = '/home/minkescanor/Desktop/WORKPLACE/Personal/Compress-Image-using-K-means/dataset/lagoon1.jpg'
def read_img(filename):
    img = cv2.imread(filename)
    original_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(type(img))
    img = img / 255
    # print(img)
    X_img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))

    print (X_img.shape)
    # plt.imshow(img)
    # plt.show()
    return X_img, original_shape
