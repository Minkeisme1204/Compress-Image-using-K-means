import numpy as np 
import matplotlib.pyplot as plt

class K_means_model(object):
    #  Attributes
    #  Centroids 
    #  X: input data points 
    #  K: number of clusters
    def __init__(self, input_data, k, iters=10):
        self.K = k
        self.X = input_data
        self.iters = iters
        # Init random centroids to 
        randidx = np.random.permutation(self.X.shape[0])
        print(randidx[:self.K])
        self.centroids = self.X[randidx[:self.K]]
        print((self.centroids.shape))
        print((self.X.shape))

    def find_closest_centroids(self):
        # Compute distance from each point to each centroid
        # Assign the closest centroid to each point
        idx = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            distance = np.linalg.norm(self.centroids[0] - self.X[i])
            tmp = 0
            for j in range(1, self.K):
                norm_ij = np.linalg.norm(self.centroids[j] - self.X[i])
                if norm_ij < distance:
                    distance = norm_ij
                    tmp = j
                    # print("OK")
            idx[i] = tmp
        # print("idxxx", idx)
        return idx
    
    def compute_centroids(self, idx):
        # Given the closest centroids, compute new centroids
        centroids = np.zeros((self.K, self.X.shape[1]))
        for i in range(self.K):
            points = []
            for j, c_i in enumerate(idx):
                if (c_i == i):
                    points.append(self.X[j])
            centroids[i] = np.mean(points, axis=0)
        
        return centroids

    def plot_progress_Kmeans(self, X, centroids, previous_centroids, idx, K, i):
        # Plots the data points with colors assigned to each centroid and 
        # shows the movement of centroids as the K-means algorithm progresses.

        # Args:
        #     X (ndarray): (m, n) Data points
        #     centroids (ndarray): (K, n) Current centroids
        #     previous_centroids (ndarray): (K, n) Previous centroids from the last iteration
        #     idx (ndarray): (m,) Index of the closest centroid for each example in X
        #     K (int): Number of centroids
        #     i (int): Current iteration number
    
        # Plot the examples, colored by their centroid assignment
        plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='rainbow', marker='o', s=30, alpha=0.6)

        # Plot the centroids as black X's
        plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, linewidths=3, label='Centroids')
        
        # Plot the previous centroids to show the movement
        for j in range(K):
            plt.plot([previous_centroids[j, 0], centroids[j, 0]], [previous_centroids[j, 1], centroids[j, 1]], 'k--')
        
        plt.title(f'K-Means Clustering Progress after Iteration {i+1}')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()
    def run(self, plot_progess=False):
        # Run the K-means algorithm
        centroids = self.centroids
        previous_centroids = centroids
        idx = np.zeros(self.X.shape[0])
        for i in range(self.iters):
            print("Training with {} clusters at Iteration {}".format(self.K, i))
            idx = self.find_closest_centroids()
            
            centroids = self.compute_centroids(idx)
            if np.array_equal(self.centroids, centroids):
                break
            self.centroids = centroids

        if (plot_progess == True):
            self.plot_progress_Kmeans(self.X, centroids, previous_centroids, idx, self.K, i)
        return idx, self.K
    
    def cost_function(self, idx):
        print(self.X[0].shape, self.centroids[0].shape)
        print(type(idx))
        m, n = self.X.shape
        K = self.centroids.shape[0]
        idx = np.array(idx, dtype=int)
        # Compute the cost function
        J = 0
        for i in range(m):
            # Compute the distance between the data point and its assigned centroid
            distance = np.linalg.norm(self.X[i] - self.centroids[idx[i]]) ** 2
            J += distance

        # Average over all examples
        J /= m
        return J


    
    
            

