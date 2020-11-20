import numpy as np
# ''' All data utilities are here'''


def sample_branch_data():
    branch = [[0, 4, 2], [2, 0, 4]]
    return branch


def sample_decision_tree_data():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
    labels = [0, 0, 1, 1]
    return features, labels


def sample_decision_tree_test():
    features = [['a', 'b'], ['b', 'a'], ['b', 'c']]
    labels = [0, 0, 1]
    return features, labels


def sample_decision_tree_pruning():
    features = [[0, 0, 0, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [2, 1, 0, 0],
                [2, 2, 1, 0],
                [2, 2, 1, 1],
                [1, 2, 1, 1],
                [0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]
                ]
    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    validation = [[1, 0, 1, 1],
                  [1, 2, 0, 0],
                  [0, 1, 0, 1],
                  [0, 2, 0, 0],
                  [0, 1, 0, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 1],
                  [2, 0, 0, 1],
                  [2, 1, 1, 1],
                  [2, 1, 0, 1],
                  [2, 1, 0, 0]]
    v_labels = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    return features, labels, validation, v_labels


def load_decision_tree_data():
    f = open('car.data', 'r')
    white = [[int(num) for num in line.split(',')] for line in f]
    white = np.asarray(white)

    [N, d] = white.shape

    ntr = int(np.round(N * 0.66))
    ntest = N - ntr

    Xtrain = white[:ntr].T[:-1].T
    ytrain = white[:ntr].T[-1].T
    Xtest = white[-ntest:].T[:-1].T
    ytest = white[-ntest:].T[-1].T

    return Xtrain, ytrain, Xtest, ytest


def most_common(lst):
    return max(set(lst), key=lst.count)

