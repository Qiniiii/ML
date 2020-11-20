import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def pruning(self,node, X_test, y_test,root):

        if len(node.children) == 0:
            return
        for child in node.children:
            self.pruning(child,X_test,y_test,root)

        count1=0
        for i in range(len(X_test)):
            temp=X_test[i].copy()
            if root.predict(temp)==y_test[i]:
                count1+=1

        node.splittable=False
        count2=0
        for i in range(len(X_test)):
            temp = X_test[i].copy()
            if root.predict(temp)==y_test[i]:
                count2+=1
        if count2<count1:
            node.splittable=True
        return




    #TODO: try to split current node
    def split(self):
        dif_label=np.unique(self.labels)

        info_gain = []
        col_values= []
        fea_split=[]

        for i in range(len(self.features[0])):
            col=[]
            for j in range(len(self.features)):
                col.append(self.features[j][i])

            dif_fea=np.unique(col)
            branch=[]
            fea_split.append(dif_fea)
            col_values.append(len(dif_fea))
            for k in range(len(dif_fea)):
                count=[]
                for label in dif_label:
                    count.append(0)
                for h in range(len(col)):
                    for g in range(len(dif_label)):
                        if dif_fea[k]==col[h] and dif_label[g]== self.labels[h]:
                            count[g]+=1
                branch.append(count)
            info_gain.append(Util.Information_Gain(0, branch))

        max_gain=-300000
        sum=0
        for i in range(len(info_gain)):
            if info_gain[i]!=0:
                sum=1
        if sum==0:
            self.splittable=False
            return
        else:
            self.splittable = True
            for i in range(len(info_gain)):
                if info_gain[i]>max_gain:
                    max_gain=info_gain[i]
                    self.dim_split=i
                    self.feature_uniq_split = fea_split[i]
                elif info_gain[i]==max_gain:
                    if col_values[i]>col_values[self.dim_split]:
                        max_gain = info_gain[i]
                        self.dim_split=i
                        self.feature_uniq_split= fea_split[i]

            for value in self.feature_uniq_split:
                label=[]
                feature=[]
                for i in range(len(self.features)):
                    if self.features[i][self.dim_split]==value:
                       label.append(self.labels[i])
                       temp=self.features[i].copy()
                       temp.pop(self.dim_split)
                       feature.append(temp)
                self.children.append(TreeNode(feature,label,len(np.unique(label))))

            for child in self.children:
                if child.splittable:
                    child.split()










    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            index=-1
            if len(feature)<=self.dim_split:
                return self.cls_max
            for i in range(len(self.feature_uniq_split)):
                if feature[self.dim_split]==self.feature_uniq_split[i]:
                    index=i
            if index==-1:
                return self.cls_max
            else:
                feature.pop(self.dim_split)
                return self.children[index].predict(feature)
        else:
            return self.cls_max
