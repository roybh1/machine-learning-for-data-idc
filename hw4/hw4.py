import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal

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
    Calculate normal density function for a given data,
    mean and standard deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.
    """

    return (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((data-mu)/sigma)**2)
    #return (np.exp(np.square((data - mu)) / (-2 * np.square(sigma)))) / (np.sqrt(2 * np.pi * np.square(sigma)))


class EM(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of Gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM process
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
        self.costs = []

        self.N = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        indexes = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indexes].reshape(self.k)
        self.sigmas = np.random.random_integers(self.k)
        self.weights = np.ones(self.k) / self.k
        self.N = data.shape[0]

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        res = self.weights * norm_pdf(data, self.mus, self.sigmas)
        sum = np.sum(res, axis=1, keepdims=True)
        self.responsibilities = res / sum

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        self.weights = np.mean(self.responsibilities, axis=0) # avg on columns
        self.mus = (np.sum(self.responsibilities * data.reshape(-1, 1), axis=0)
                    / np.sum(self.responsibilities, axis=0))
        variance = np.mean(
            self.responsibilities * np.square(data.reshape(-1, 1) - self.mus), axis=0
        ) / self.weights
        self.sigmas = np.sqrt(variance)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        for i in range(self.n_iter):
            self.costs.append(self.calc_cost(data))
            self.expectation(data)
            self.maximization(data)
            if len(self.costs) > 1:
                if abs(self.costs[-1] - self.costs[-2]) < self.eps:
                    return

    def calc_cost(self, data):
        return np.sum(-1 * np.log(self.weights * norm_pdf(data, self.mus, self.sigmas)))

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm density function for a given data,
    mean and standard deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.
    """
    return np.sum(weights*norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of Gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.gaussians = None
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
        self.prior = {label: len(y[y == label]) / len(y) for label in np.unique(y)}
        self.gaussians = {}
        for label in np.unique(y):
            self.gaussians[label] = {}
            for feature in range(X.shape[1]):
                em = EM(self.k)
                em.fit(X[y == label][:, feature].reshape(-1,1))
                self.gaussians[label][feature] = em.get_dist_params()

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            posteriors = []
            for label in self.prior:
                likelihood = self.calc_likelihood(instance, label)
                prior = self.prior[label]
                posterior = likelihood * prior
                posteriors.append((posterior, label))
            preds.append(max(posteriors, key=lambda t: t[0])[1])
        return preds

    def calc_likelihood(self, X, label):
        """
        calc likelihood for X
        """
        likelihood = 1
        for feature in range(X.shape[0]):
            weights, mus, sigmas = self.gaussians[label][feature]
            likelihood = likelihood * gmm_pdf(X[feature], weights, mus, sigmas)
        return likelihood


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
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)

    lor_predictions_x_train = lor.predict(x_train)
    lor_predictions_x_test = lor.predict(x_test)

    lor_train_acc = np.count_nonzero(lor_predictions_x_train == y_train) / len(y_train)
    lor_test_acc = np.count_nonzero(lor_predictions_x_test == y_test) / len(y_test)

    naive_bayes = NaiveBayesGaussian(k=k)
    naive_bayes.fit(x_train, y_train)
    nb_predictions_x_train = naive_bayes.predict(x_train)
    nb_predictions_x_test = naive_bayes.predict(x_test)

    bayes_train_acc = np.count_nonzero(nb_predictions_x_train == y_train) / len(y_train)
    bayes_test_acc = np.count_nonzero(nb_predictions_x_test == y_test) / len(y_test)

    return {
        "lor_train_acc": lor_train_acc,
        "lor_test_acc": lor_test_acc,
        "bayes_train_acc": bayes_train_acc,
        "bayes_test_acc": bayes_test_acc,
    }


def generate_datasets():
    """
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    """
    np.random.seed(0)

    # class 0
    mean1_a = [0, 0, 0]
    cov1_a = np.diag([1, 1, 1])
    class0_part1_a = multivariate_normal.rvs(mean=mean1_a, cov=cov1_a, size=500)

    mean2_a = [3, 3, 3]
    cov2_a = np.diag([1, 1, 1])
    class0_part2_a = multivariate_normal.rvs(mean=mean2_a, cov=cov2_a, size=500)

    # class 1
    mean1_b = [0, 5, 10]
    cov1_b = np.diag([1, 1, 1])
    class1_part1_a = multivariate_normal.rvs(mean=mean1_b, cov=cov1_b, size=500)

    mean2_b = [5, 10, 15]
    cov2_b = np.diag([1, 1, 1])
    class1_part2_a = multivariate_normal.rvs(mean=mean2_b, cov=cov2_b, size=500)

    data_class0_a = np.concatenate([class0_part1_a, class0_part2_a])
    data_class1_a = np.concatenate([class1_part1_a, class1_part2_a])

    labels_class0_a = np.zeros(data_class0_a.shape[0])
    labels_class1_a = np.ones(data_class1_a.shape[0])

    dataset_a_features = np.concatenate([data_class0_a, data_class1_a])
    dataset_a_labels = np.concatenate([labels_class0_a, labels_class1_a])

    # class 0
    mean1_c = [2, 2, 2]
    cov1_c = [[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]]
    class0_part1_b = multivariate_normal.rvs(mean=mean1_c, cov=cov1_c, size=500)

    mean2_c = [8, 8, 8]
    cov2_c = [[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]]
    class0_part2_b = multivariate_normal.rvs(mean=mean2_c, cov=cov2_c, size=500)

    # class 1
    mean1_d = [6, 6, 6]
    cov1_d = [[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]]
    class1_part1_b = multivariate_normal.rvs(mean=mean1_d, cov=cov1_d, size=500)

    mean2_d = [12, 12, 12]
    cov2_d = [[1, 0.8, 0.6], [0.8, 1, 0.8], [0.6, 0.8, 1]]
    class1_part2_b = multivariate_normal.rvs(mean=mean2_d, cov=cov2_d, size=500)

    data_class0_b = np.concatenate([class0_part1_b, class0_part2_b])
    data_class1_b = np.concatenate([class1_part1_b, class1_part2_b])

    labels_class0_b = np.zeros(data_class0_b.shape[0])
    labels_class1_b = np.ones(data_class1_b.shape[0])

    dataset_b_features = np.concatenate([data_class0_b, data_class1_b])
    dataset_b_labels = np.concatenate([labels_class0_b, labels_class1_b])

    return {
        "dataset_a_features": dataset_a_features,
        "dataset_a_labels": dataset_a_labels,
        "dataset_b_features": dataset_b_features,
        "dataset_b_labels": dataset_b_labels,
    }
