import numpy as np

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    first_center=generator.randint(0,n)
    centers=[]
    centers.append(first_center)
    k=n_cluster
    while k-1>0:
        d_array=[]
        for i in range(n):
            d=[]
            for j in range(len(centers)):
                d.append(np.sum(np.power(x[i]-x[centers[j]],2)))
            d_array.append(min(d))
        next=np.argmax(np.array(d_array))
        centers.append(next)
        k=k-1
    # DO NOT CHANGE CODE BELOW THIS LINE
    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        mean=[]
        for i in range(self.n_cluster):
            mean.append(x[self.centers[i]])
        membership=np.zeros((N,self.n_cluster),dtype=int)
        t=1
        dis=10**10
        while t<=self.max_iter:
            mean_copy=mean.copy()
            membership = np.zeros((N, self.n_cluster), dtype=int)
            # for n in range(N):
            #     #d=[]
            #     # for j in range(self.n_cluster):
            #     #     membership[n][j] = 0
            #     #     d.append(np.sum(np.power(x[n] - mean[j], 2)))
            #     d=np.sum(np.power(mean-x[n], 2),axis=1)
            #     min_cluster = np.argmin(d)
            #     membership[n][min_cluster] = 1
            d = []
            for j in range(self.n_cluster):
                d.append(np.sum(np.power(x - mean[j], 2), axis=1))
            min=np.argmin(np.array(d),axis=0)
            for i in range(N):
                membership[i][min[i]]=1
            new_dis=0
            # for n in range(N):
            #     for j in range(self.n_cluster):
            #         if membership[n][j]==1:
            #             new_dis=new_dis+np.sum(np.power(x[n] - mean[j], 2))
            new_dis=np.sum(np.power(x-np.dot(membership,np.array(mean)),2))
            print(new_dis - dis,t)
            if abs(new_dis-dis)/N<self.e:
                break
            dis=new_dis

            # for j in range(self.n_cluster):
            #     count = 0
            #     x_n=np.zeros(D)
            #     for n in range(N):
            #         count+=membership[n][j]
            #         x_n=x_n+membership[n][j]*x[n]
            #     if count!=0:
            #         mean[j]=x_n/count
            tmp1=np.dot(membership.T,np.array(x))
            tmp2=np.sum(membership,axis=0)
            for j in range(self.n_cluster):
                if tmp2[j]!=0:
                    mean[j]=tmp1[j]/tmp2[j]

            t += 1
        self.max_iter=t
        y=[]
        for n in range(N):
            for j in range(self.n_cluster):
                if membership[n][j]==1:
                    y.append(j)
        #y=np.argmax(membership,axis=1)
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(mean), np.array(y), self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        k_mean=KMeans(self.n_cluster,self.max_iter,self.e,self.generator)
        means, membership, number_of_updates=k_mean.fit(x,centroid_func)
        centroids=means
        centroid_labels=[]
        for j in range(self.n_cluster):
            d = []
            for n in range(N):
                if membership[n]==j:
                    d.append(y[n])
            if len(d)==0:
                centroid_labels.append(0)
            else:
                centroid_labels.append(np.bincount(d).argmax())
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = np.array(centroid_labels)
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        labels=[]
        for n in range(N):
            d = []
            for j in range(self.n_cluster):
                d.append(np.sum(np.power(x[n] - self.centroids[j], 2)))
            min_cluster = np.argmin(np.array(d))
            labels.append(self.centroid_labels[min_cluster])

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    size=(image.shape[0],image.shape[1],3)
    new_im=np.array(image.reshape((image.shape[0]*image.shape[1],3)),copy=True)
    d = []
    for j in range(code_vectors.shape[0]):
        d.append(np.sum(np.power(new_im - code_vectors[j], 2), axis=1))
    min = np.argmin(np.array(d), axis=0)

    for i in range(new_im.shape[0]):
        for k in range(3):
            new_im[i][k]=code_vectors[min[i]][k]
    new_im=new_im.reshape(size)
    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
