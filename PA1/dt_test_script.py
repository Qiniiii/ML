import data
import hw1_dt as decision_tree
import utils as Utils


def information_gain_test():
    branch = data.sample_branch_data()
    score = Utils.Information_Gain(0, branch)
    print('Your information gain: ', score)
    print('My information gain: ', -0.91829583405448956)


def decision_tree_test():
    features, labels = data.sample_decision_tree_data()

    # build the tree
    dTree = decision_tree.DecisionTree()

    dTree.train(features, labels)

    # print
    print('Your decision tree: ')
    Utils.print_tree(dTree)
    print('My decision tree: ')
    print('branch 0{\n\tdeep: 0\n\tnum of samples for each class: 2 : 2 \n\tsplit by dim 0\n\tbranch 0->0{\n\t\tdeep: '
          '1\n\t\tnum of samples for each class: 1 \n\t\tclass:0\n\t}\n\tbranch 0->1{\n\t\tdeep: 1\n\t\tnum of '
          'samples for each class: 1 : 1 \n\t\tsplit by dim 0\n\t\tbranch 0->1->0{\n\t\t\tdeep: 2\n\t\t\tnum of '
          'samples for each class: 1 \n\t\t\tclass:0\n\t\t}\n\t\tbranch 0->1->1{\n\t\t\tdeep: 2\n\t\t\tnum of '
          'samples for each class: 1 \n\t\t\tclass:1\n\t\t}\n\t}\n\tbranch 0->2{\n\t\tdeep: 1\n\t\tnum of '
          'samples for each class: 1 \n\t\tclass:1\n\t}\n}')

    # data
    X_test, y_test = data.sample_decision_tree_test()

    # testing
    y_est_test = dTree.predict(X_test)
    print('Your estimate test: ', y_est_test)
    print('My estimate test: ', [0, 0, 1])


def pruning_decision_tree_test():
    # load data
    X_train, y_train, X_test, y_test = data.sample_decision_tree_pruning()

    # build the tree
    dTree = decision_tree.DecisionTree()
    dTree.train(X_train, y_train)

    # print
    print('Your decision tree:')
    Utils.print_tree(dTree)
    print('My decision tree:')
    print('branch 0{\n\tdeep: 0\n\tnum of samples for each class: 5 : 9 \n\tsplit by dim 0\n\tbranch 0->0{\n\t\tdeep: 1'
          '\n\t\tnum of samples for each class: 3 : 2 \n\t\tsplit by dim 1\n\t\tbranch 0->0->0{\n\t\t\tdeep: 2\n\t\t\t'
          'num of samples for each class: 3 \n\t\t\tclass:0\n\t\t}\n\t\tbranch 0->0->1{\n\t\t\tdeep: 2\n\t\t\tnum of '
          'samples for each class: 2 \n\t\t\tclass:1\n\t\t}\n\t}\n\tbranch 0->1{\n\t\tdeep: 1\n\t\tnum of samples for '
          'each class: 4 \n\t\tclass:1\n\t}\n\tbranch 0->2{\n\t\tdeep: 1\n\t\tnum of samples for each class: 2 : 3 '
          '\n\t\tsplit by dim 2\n\t\tbranch 0->2->0{\n\t\t\tdeep: 2\n\t\t\tnum of samples for each class: 3 \n\t\t\t'
          'class:1\n\t\t}\n\t\tbranch 0->2->1{\n\t\t\tdeep: 2\n\t\t\tnum of samples for each class: 2 \n\t\t\tclass:0'
          '\n\t\t}\n\t}\n}')

    Utils.reduced_error_prunning(dTree, X_test, y_test)

    print('Your decision tree after pruning:')
    Utils.print_tree(dTree)
    print('My decision tree after pruning:')
    print('branch 0{\n\tdeep: 0\n\tnum of samples for each class: 5 : 9 \n\tsplit by dim 0\n\tbranch 0->0{\n\t\tdeep: '
          '1\n\t\tnum of samples for each class: 3 : 2 \n\t\tsplit by dim 1\n\t\tbranch 0->0->0{\n\t\t\tdeep: 2\n\t\t\t'
          'num of samples for each class: 3 \n\t\t\tclass:0\n\t\t}\n\t\tbranch 0->0->1{\n\t\t\tdeep: 2\n\t\t\tnum of '
          'samples for each class: 2 \n\t\t\tclass:1\n\t\t}\n\t}\n\tbranch 0->1{\n\t\tdeep: 1\n\t\tnum of samples for '
          'each class: 4 \n\t\tclass:1\n\t}\n\tbranch 0->2{\n\t\tdeep: 1\n\t\tnum of samples for each class: 2 : 3 '
          '\n\t\tclass:1\n\t}\n}')


if __name__ == "__main__":
    information_gain_test()
    decision_tree_test()
    pruning_decision_tree_test()
