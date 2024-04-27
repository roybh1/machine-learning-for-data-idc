import math
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {
    1: {0.5: 0.45, 0.25: 1.32, 0.1: 2.71, 0.05: 3.84, 0.0001: 100000},
    2: {0.5: 1.39, 0.25: 2.77, 0.1: 4.60, 0.05: 5.99, 0.0001: 100000},
    3: {0.5: 2.37, 0.25: 4.11, 0.1: 6.25, 0.05: 7.82, 0.0001: 100000},
    4: {0.5: 3.36, 0.25: 5.38, 0.1: 7.78, 0.05: 9.49, 0.0001: 100000},
    5: {0.5: 4.35, 0.25: 6.63, 0.1: 9.24, 0.05: 11.07, 0.0001: 100000},
    6: {0.5: 5.35, 0.25: 7.84, 0.1: 10.64, 0.05: 12.59, 0.0001: 100000},
    7: {0.5: 6.35, 0.25: 9.04, 0.1: 12.01, 0.05: 14.07, 0.0001: 100000},
    8: {0.5: 7.34, 0.25: 10.22, 0.1: 13.36, 0.05: 15.51, 0.0001: 100000},
    9: {0.5: 8.34, 0.25: 11.39, 0.1: 14.68, 0.05: 16.92, 0.0001: 100000},
    10: {0.5: 9.34, 0.25: 12.55, 0.1: 15.99, 0.05: 18.31, 0.0001: 100000},
    11: {0.5: 10.34, 0.25: 13.7, 0.1: 17.27, 0.05: 19.68, 0.0001: 100000},
}


def calc_gini(data: np.ndarray):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    data_size = len(data)
    labels_set = set(data[:, -1])
    label_count = {label: 0 for label in labels_set}

    for row in data:
        label_count[row[-1]] += 1

    return 1.0 - sum([(label_count[label] / data_size) ** 2 for label in label_count])


# def calc_gini_reference(data: np.ndarray):
#    """
#    Calculate gini impurity measure of a dataset.
#
#    Input:
#    - data: any dataset where the last column holds the labels.
#
#    Returns:
#    - gini: The gini impurity value.
#    """
#    data_size = len(data)
#    _, count = np.unique(data[:, -1], return_counts=True)
#
#    return 1.0 - np.sum((count/data_size)**2)


def calc_entropy(data: np.ndarray):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    return _calc_entropy(data, -1)


def _calc_entropy(data: np.ndarray, feature: int = -1):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    data_size = len(data)
    A = set(data[:, feature])
    A_count = {v: 0 for v in A}

    for row in data:
        A_count[row[feature]] += 1

    s = 0
    for v in A:
        dist = A_count[v] / data_size
        s += dist * math.log2(dist)

    return -1.0 * s


# def calc_entropy_reference(data: np.ndarray):
#    """
#    Calculate the entropy of a dataset.
#
#    Input:
#    - data: any dataset where the last column holds the labels.
#
#    Returns:
#    - entropy: The entropy value.
#    """
#    data_size = len(data)
#    _, count = np.unique(data[:, -1], return_counts=True)
#
#    dist = count / data_size
#
#    return -1.0 * np.sum(np.dot(dist, np.log2(dist)))


class DecisionNode:
    def __init__(
        self,
        data: np.ndarray,
        impurity_func: Callable,
        feature: int = -1,
        depth: int = 0,
        chi: int = 1,
        max_depth: int = 1000,
        gain_ratio: bool = False,
    ):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children: List[DecisionNode] = []  # array that holds this nodes children
        self.children_values: List = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self) -> str:
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        # Creating variables for calculate node predict.
        pred = None
        param, count = np.unique(self.data[:, -1], return_counts=True)
        dict_param_count = dict(zip(param, count))

        # Finding the prediction of the node.
        pred = max(dict_param_count, key=dict_param_count.get)
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def calc_feature_importance(self, n_total_sample: int):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        params, counts = np.unique(self.data[:, self.feature], return_counts=True)
        groups = {}

        s = 0.0
        for i in range(len(params)):
            param = params[i]
            group = self.data[self.data[:, self.feature] == param]
            count = counts[i]
            groups[param] = group
            s += (count / n_total_sample) * self.impurity_func(group)

        self.feature_importance = (
            len(self.data) / n_total_sample
        ) * self.impurity_func(self.data) - s

    def goodness_of_split(self, feature: int) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Calculate the goodness of split of a dataset given a feature and
                impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        params, counts = np.unique(self.data[:, feature], return_counts=True)
        groups = {}

        impurity_attribute = 0.0
        for i in range(len(params)):
            param = params[i]
            group = self.data[self.data[:, feature] == param]
            groups[param] = group
            count = counts[i]
            impurity_attribute += count / len(self.data) * self.impurity_func(group)

        goodness_of_split = self.impurity_func(self.data) - impurity_attribute

        if self.gain_ratio:
            # assumes impurity_function is calc_entropy
            # calc split_information
            split_information = _calc_entropy(self.data, feature)

            # calc gain ratio
            gain_ratio = goodness_of_split / split_information
            return gain_ratio, groups

        # calc goodness of split
        return goodness_of_split, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        features = len(self.data.T) - 1
        best_split = -1
        best_feature = -1
        for feature in range(features):
            goodness_of_split_by_feature, _ = self.goodness_of_split(feature)
            if goodness_of_split_by_feature > best_split:
                best_split = goodness_of_split_by_feature
                best_feature = feature

        self.feature = best_feature

        _, groups = self.goodness_of_split(self.feature)

        # FIXME address self.chi
        if (self.depth < self.max_depth) and (self.check_chi_test()):
            for param, group in groups.items():
                child = DecisionNode(
                    group,
                    self.impurity_func,
                    depth=self.depth + 1,
                    chi=self.chi,
                    max_depth=self.max_depth,
                    gain_ratio=self.gain_ratio,
                )
                self.add_child(child, param)

    def check_chi_test(self):
        return True
        # labels, label_counts = np.unique(self.data[:, -1], return_counts=True) # for labels (Y)
        # label_distribution_dict = {labels[i]: label_counts[i]/len(self.data) for i in range(len(labels))}

        # best_feature_params = np.unique(self.data[:, self.feature])

        # for i in range(len(best_feature_params)):
        #    param = best_feature_params[i]
        #    group = self.data[self.data[:, self.feature] == param]
        #    d_f =
        #    p_f =
        #    n_f =
        #    e_i =


class DecisionTree:
    def __init__(
        self,
        data: np.ndarray,
        impurity_func: Callable,
        feature: int = -1,
        chi: int = 1,
        max_depth: int = 1000,
        gain_ratio: bool = False,
    ):
        self.data = data  # the relevant data for the tree
        self.impurity_func = (
            impurity_func  # the impurity function to be used in the tree
        )
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root: Optional[DecisionNode] = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        root = DecisionNode(
            self.data,
            self.impurity_func,
            depth=0,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio,
        )
        root.split()

        queue = [root]

        while queue:
            current_node = queue.pop(0)
            if (current_node.goodness_of_split(current_node.feature)[0] > 0) and (
                self.impurity_func(current_node.data) > 0
            ):
                current_node.split()
                for child in current_node.children:
                    queue.append(child)

            else:
                break

        self.root = root

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

    def depth(self):
        return self.root.depth()


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pass
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
