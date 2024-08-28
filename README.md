# Compress image using Machine Learning algorithm K means 
Each image data point is clustered into a number of clusters which starts from 17 colors 
Model will try to cluster data points into the number of clusters until its cost function value J is smaller than epsilon.
For the epoch or iteration number, it is set to a default value of 20 (it still works properly with 10).

The result of compressing image indicates that the image should be compressed to a significantly smaller size, compared to the original image, that equals to approximately 1/3 of the original image