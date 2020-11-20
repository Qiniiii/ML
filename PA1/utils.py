import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches_count = []
    leaf_num = 0
    entropy = []
    gain=0
    # calculate count
    for i in range(len(branches)):
        branches_count.append(0)
        for j in range(len(branches[i])):
            branches_count[i] += branches[i][j]
    for i in range(len(branches_count)):
        leaf_num += branches_count[i]

    # calculate entropy and s
    for i in range(len(branches)):
        entropy.append(0)
        for j in range(len(branches[i])):
            if branches[i][j] == 0:
                entropy[i] += 0
            else:
                entropy[i] -= branches[i][j]/branches_count[i] * np.log2(branches[i][j]/branches_count[i])
    for i in range(len(branches)):
        gain -= branches_count[i]/leaf_num*entropy[i]
    gain = S+gain
    return gain

    raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    decisionTree.root_node.pruning(decisionTree.root_node,X_test,y_test,decisionTree.root_node)


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
