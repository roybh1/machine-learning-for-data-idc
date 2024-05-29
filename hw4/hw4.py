import numpy as np
import pandas as pd


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values.

    Returns:
    - The Pearson correlation coefficient between the two columns.
    """
    y_mean = np.mean(y, axis=0)
    x_mean = np.mean(x)

    numerator = np.dot((x - x_mean), (y - y_mean))
    denominator = (
        np.dot((x - x_mean), (x - x_mean)) * np.dot((y - y_mean), (y - y_mean))
    ) ** 0.5

    return numerator / denominator


def feature_selection(X: pd.DataFrame, y: pd.Series, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).
    """
    feature_to_pearson_corr = {}
    for feat_name, feat_data in X.items():
        if feat_name not in ["date", "id"]:
            feature_to_pearson_corr[feat_name] = pearson_correlation(
                feat_data.to_numpy(), y
            )

    sorted_items = sorted(feature_to_pearson_corr.items(), key=lambda item: item[1])
    return [item[0] for item in sorted_items[:n_features]]


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def apply_bias_trick(self, X: np.ndarray):
        """
        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).
        """
        return np.c_[np.ones(len(X)), X]

    def compute_cost(self, X, y):
        sig = 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
        epsilon = 1e-5  # small value to avoid log(0).
        J = (-1.0 / len(y)) * (np.dot(y.T, np.log(sig + epsilon)) + np.dot((1 - y).T, np.log(1 - sig + epsilon)))
        return J

    def _compute_partial_derivative(self, X, y, _by_j):
        """
        _by_j: index to compute partial deriv by
        """
        h = 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
        return np.sum((h - y) * X[:, _by_j]) / X.shape[0]

    def fit_1(self, X, y):
        X = self.apply_bias_trick(X)
        np.random.seed(self.random_state)

        self.theta = np.random.random(X.shape[1])

        for _ in range(self.n_iter):
            h = 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
            gradient = np.dot(X.T, (h - y))
            self.theta = self.theta - self.eta * gradient

            # Calculate cost and check for convergence.
            J = self.compute_cost(X, y)
            self.Js.append(J)
            self.thetas.append(self.theta.copy())

            if len(self.Js) > 1 and abs(J - self.Js[-2]) < self.eps:
                break

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        X = self.apply_bias_trick(X)

        # set random seed
        np.random.seed(self.random_state)
        shape = X.shape[1]
        self.theta = np.random.random(size=shape)

        for i in range(self.n_iter):
            h = 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
            gradient = np.dot(X.T, (h - y))
            self.theta = self.theta - self.eta * gradient

            cost = self.compute_cost(X, y)
            self.Js.append(cost)
            self.thetas.append(self.theta.copy())

            if len(self.Js) > 1:
                if abs(self.Js[-2] - self.Js[-1]) < self.eps:
                    break

        self.theta = self.thetas[-1]


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        X = self.apply_bias_trick(X)

        sig = 1 / (1 + np.exp(-1 * np.dot(X, self.theta)))
        return np.round(sig).astype(int)


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    mean_accuracy = None
    np.random.seed(random_state)

    #1. shuffle the data and creates folds
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    size = X.shape[0] // folds
    accuracies = []

    #2. train the model on each fold
    for fold in range(folds):
        start = fold * size
        end = (fold + 1) * size

        X_val = X[start:end]
        y_val = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]], axis=0)
        y_train = np.concatenate([y[:start], y[end:]], axis=0)

        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_val)

        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)

    #3. calculate aggregated metrics
    mean_accuracy = np.mean(accuracies)

    return mean_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """

    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((data-mu)/sigma)**2)


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    """
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    """

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {
        "lor_train_acc": lor_train_acc,
        "lor_test_acc": lor_test_acc,
        "bayes_train_acc": bayes_train_acc,
        "bayes_test_acc": bayes_test_acc,
    }


def generate_datasets():
    #from scipy.stats import multivariate_normal

    """
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    """
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }
