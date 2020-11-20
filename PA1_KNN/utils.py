import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """

    assert len(real_labels) == len(predicted_labels)

    score1 = 0
    score2 = 0
    score3 = 0
    for i in range(len(real_labels)):
        score1 += real_labels[i]
        score2 += predicted_labels[i]
        score3 += real_labels[i]*predicted_labels[i]

    return 2 * score3 / (score1 + score2)

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        distance = 0
        for i in range(len(point1)):
            distance += abs(point1[i]-point2[i]) ** 3

        return distance ** (1 / 3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        distance = 0
        for i in range(len(point1)):
            distance += (point1[i]-point2[i]) ** 2

        return distance ** (1 / 2)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        return np.inner(point1, point2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        xy = 0
        x = 0
        y = 0
        for i in range(len(point1)):
            x += point1[i] ** 2
            y += point2[i] ** 2
            xy += point2[i] * point1[i]

        return 1-(xy / ((x * y) ** (1/2)))

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """

        distance = 0
        for i in range(len(point1)):
            distance += (point1[i]-point2[i]) ** 2

        return -np.exp(-1/2 * distance)


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        temp_f1_score = 0
        for k in range(1, min(len(x_train), 30), 2):
            for x in distance_funcs:
                distance_function = distance_funcs[x]
                model = KNN(k, distance_function)
                model.train(features=x_train, labels=y_train)
                pre_labels = model.predict(features=x_val)
                score = f1_score(y_val, pre_labels)
                if score > temp_f1_score:
                    temp_f1_score = score
                    print(temp_f1_score, k, distance_function)
                    self.best_k = k
                    self.best_distance_function = x
                    self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        temp_f1_score = 0
        for k in range(1, min(len(x_train), 30), 2):
            for x in distance_funcs:
                for y in scaling_classes:
                    distance_function = distance_funcs[x]
                    scaler=scaling_classes[y]()
                    model = KNN(k, distance_function)
                    scale_x_train=scaler(x_train)
                    scale_x_val=scaler(x_val)
                    model.train(features=scale_x_train, labels=y_train)
                    pre_labels = model.predict(features=scale_x_val)
                    score = f1_score(y_val, pre_labels)
                    if score > temp_f1_score:
                        temp_f1_score = score
                        # print(temp_f1_score, k, distance_function)
                        self.best_k = k
                        self.best_distance_function = x
                        self.best_model = model
                        self.best_scaler = y


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        nom_features = []
        for feature in features:
            s = 0
            nom_feature = []
            for i in range(len(feature)):
                s += feature[i] ** 2
            for i in range(len(feature)):
                if s == 0:
                    nom_feature.append(0)
                else:
                    nom_feature.append(feature[i] / (s ** (1/2)))
            nom_features.append(nom_feature)
        return nom_features

class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):

        pass


    count=0
    minmax=[]
    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        self.count+=1

        if self.count<=1:
            for i in range(len(features[0])):
                max = 0
                min = 0
                for j in range(len(features)):
                    if max<features[j][i]:
                        max=features[j][i]
                    elif min>features[j][i]:
                        min=features[j][i]
                self.minmax.append([min, max])

        nom_features = []
        for i in range(len(features)):
            nom_feature=[]
            for j in range(len(features[0])):
                nom_feature.append((features[i][j]-self.minmax[j][0]) / (self.minmax[j][1]-self.minmax[j][0]))
            nom_features.append(nom_feature)
        return nom_features
