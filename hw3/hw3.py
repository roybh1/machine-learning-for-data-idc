from typing import Dict, Tuple

import numpy as np
from numpy.lib import math


class conditional_independence:

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y: Dict[Tuple[int, int], float] = {
            (0, 0): 0.3,  # not 0.3*0.3
            (0, 1): 0.1,  # not 0.3*0.7
            (1, 0): 0.2,  # not 0.7*0.3
            (1, 1): 0.4,  # not not 0.7*0.7
        }  # P(X=x, Y=y)

        self.X_C: Dict[Tuple[int, int], float] = {
            (0, 0): self.X[0] * self.C[0],
            (0, 1): self.X[0] * self.C[1],
            (1, 0): self.X[1] * self.C[0],
            (1, 1): self.X[1] * self.C[1],
        }  # P(X=x, C=y)

        self.Y_C: Dict[Tuple[int, int], float] = {
            (0, 0): self.Y[0] * self.C[0],
            (0, 1): self.Y[0] * self.C[1],
            (1, 0): self.Y[1] * self.C[0],
            (1, 1): self.Y[1] * self.C[1],
        }  # P(Y=y, C=c)

        self.X_Y_C: Dict[Tuple[int, int, int], float] = {
            (0, 0, 0): (self.X_C[(0, 0)] / self.C[0])
            * (self.Y_C[(0, 0)] / self.C[0])
            * self.C[0],
            (0, 0, 1): (self.X_C[(0, 1)] / self.C[1])
            * (self.Y_C[(0, 1)] / self.C[1])
            * self.C[1],
            (0, 1, 0): (self.X_C[(0, 0)] / self.C[0])
            * (self.Y_C[(1, 0)] / self.C[0])
            * self.C[0],
            (0, 1, 1): (self.X_C[(0, 1)] / self.C[1])
            * (self.Y_C[(1, 1)] / self.C[1])
            * self.C[1],
            (1, 0, 0): (self.X_C[(1, 0)] / self.C[0])
            * (self.Y_C[(0, 0)] / self.C[0])
            * self.C[0],
            (1, 0, 1): (self.X_C[(1, 1)] / self.C[1])
            * (self.Y_C[(0, 1)] / self.C[1])
            * self.C[1],
            (1, 1, 0): (self.X_C[(1, 0)] / self.C[0])
            * (self.Y_C[(1, 0)] / self.C[0])
            * self.C[0],
            (1, 1, 1): (self.X_C[(1, 1)] / self.C[1])
            * (self.Y_C[(1, 1)] / self.C[1])
            * self.C[1],
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        for x in [0, 1]:
            for y in [0, 1]:
                if not np.isclose(self.X_Y[(x, y)], self.X[x] * self.Y[y]):
                    return True
        return False

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        for c in [0, 1]:
            P_X_given_C = {x: self.X_C[(x, c)] / self.C[c] for x in [0, 1]}
            P_Y_given_C = {y: self.Y_C[(y, c)] / self.C[c] for y in [0, 1]}
            for x in [0, 1]:
                for y in [0, 1]:
                    P_XY_given_C = self.X_Y_C[(x, y, c)] / self.C[c]
                    if not np.isclose(P_XY_given_C, P_X_given_C[x] * P_Y_given_C[y]):
                        return False
        return True


def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    return k * math.log(rate) - rate - math.log(math.factorial(k))


def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of
        rates[i]
    """
    likelihoods = []

    for rate in rates:
        likelihood = 0
        for sample in samples:
            likelihood += poisson_log_pmf(sample, rate)

        likelihoods.append(likelihood)

    return np.array(likelihoods)


def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood
    """
    return rates[get_poisson_log_likelihoods(samples, rates).argmax()]


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    return np.mean(samples)


def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mean and std for the
    given x.
    """
    square = (x - mean) ** 2
    exp = np.exp(-(square / (2 * (std**2))))
    sqr = np.sqrt(2 * (np.pi) * (std) ** 2)

    p = exp / sqr
    return p

class NaiveNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class
        conditional normal distribution.
        The mean and std are computed from a given data set.

        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the
            last column
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value

        self.dataset_given_class_value = self.dataset[
                self.dataset[:, -1] == self.class_value, :-1
        ]

        self.mean = np.mean(self.dataset_given_class_value, axis=0)
        self.std = np.std(self.dataset_given_class_value, axis=0)


    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution
        """
        return len(self.dataset_given_class_value) / len(self.dataset)


    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to
        the dataset distribution.
        """
        likelihood = 1
        for feature_i in range(len(self.dataset.T) - 1):
            likelihood *= normal_pdf(
                x[feature_i], self.mean[feature_i], self.std[feature_i]
            )
        return likelihood


    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to
        the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions.
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability
        for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods
                     for the distribution of class 1.
        """
        self.ccd0: NaiveNormalClassDistribution = ccd0
        self.ccd1: NaiveNormalClassDistribution = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the
        object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return (
            0
            if self.ccd0.get_instance_posterior(x)
            >= self.ccd1.get_instance_posterior(x)
            else 1
        )


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.

    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where
        the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for
        each instance in the testset.

    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    accurate = 0.0
    counter = 0
    for row in test_set:
        counter += 1
        predicted_label = map_classifier.predict(row[:-1])
        actual_label = row[-1]
        if predicted_label == actual_label:
            accurate += 1.0

    return accurate / len(test_set)


def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal density function for a given x, mean and covariance
    matrix.

    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.

    Returns the normal distribution pdf according to the given mean and var for the
    given x.
    """
    n = x.shape[0]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = (x - mean).reshape((n, 1))
    exponent = -0.5 * diff.T @ inv @ diff

    pdf = (1.0 / (np.sqrt((2 * np.pi) ** n * det))) * np.exp(exponent)
    return pdf


class MultiNormalClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class
        conditinonal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a
        given data set.

        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.dataset = dataset
        self.class_value = class_value

        self.dataset_given_class_value = self.dataset[
            self.dataset[:, -1] == self.class_value, :-1
        ]

        self.mean = np.mean(self.dataset_given_class_value, axis=0)
        self.cov = np.cov(self.dataset_given_class_value.T)

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution
        """
        return len(self.dataset_given_class_value) / len(self.dataset)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset
        distribution.
        """
        return multi_normal_pdf(x, self.mean, self.cov)

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to
        the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MaxPrior:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object
        constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return (
            0
            if self.ccd0.get_prior()
            >= self.ccd1.get_prior()
            else 1
        )


class MaxLikelihood:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1,
        and will predicit an instance
        by the class that outputs the highest likelihood probability for the given 
        instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the 
            distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the 
            distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object
        constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return (
            0
            if np.prod(self.ccd0.get_instance_likelihood(x))
            >= np.prod(self.ccd1.get_instance_likelihood(x))
            else 1
        )


EPSILLON = 1e-6  # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.


class DiscreteNBClassDistribution:
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete 
        naive bayes
        distribution for a specific class. The probabilites are computed with laplace 
        smoothing.

        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given
        class.
        """
        self.dataset = dataset
        self.class_value = class_value

        self.dataset_given_class_value = self.dataset[
                self.dataset[:, -1] == self.class_value
        ]

        self.mean = np.mean(self.dataset_given_class_value, axis=0)
        self.std = np.std(self.dataset_given_class_value, axis=0)

    def get_prior(self):
        """
        Returns the prior porbability of the class
        according to the dataset distribution.
        """
        return len(self.dataset_given_class_value) / len(self.dataset)

    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under
        the class according to the dataset distribution.
        """
        likelihood = 1
        n_i = len(self.dataset_given_class_value)

        for i, feature in enumerate(self.dataset_given_class_value.T[:-1]):
            V_j = len(set(feature))
            n_i_j = len(feature[x[i] == feature[:]])

            # Calculation of the likelihood
            if n_i + V_j != 0:
                likelihood *= (n_i_j + 1) / (n_i + V_j)
            else:
                likelihood *= (n_i_j + 1) / (n_i + V_j + EPSILLON)

        return likelihood

    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()


class MAPClassifier_DNB:
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier.
        This class will hold 2 class distributions, one for class 0 and one for class 1,
        and will predict an instance
        by the class that outputs the highest posterior probability for the given 
        instance.

        Input
            - ccd0 : An object contating the relevant parameters and methods for the 
                distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the 
                distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1


    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the 
        object constructor.

        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        return (
            0
            if self.ccd0.get_instance_posterior(x)
            >= self.ccd1.get_instance_posterior(x)
            else 1
        )

    def compute_accuracy1(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        accurate = 0.0
        counter = 0
        for row in test_set:
            counter += 1
            predicted_label = self.predict(row[:-1])
            actual_label = row[-1]
            if predicted_label == actual_label:
                accurate += 1.0

        return accurate / len(test_set)

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        num_correct = 0

        # Loop that check if the instance's prediction is correctly classified.
        for instance in test_set:
            true_class = instance[-1]
            predicted_class = self.predict(instance[:-1])
            if predicted_class == true_class:
                num_correct += 1

        # Calculate the accuracy.
        acc = num_correct / test_set.shape[0]
        return acc
