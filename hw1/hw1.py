###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    normalized_X = (X - X.min()) / (X.max() - X.min())
    normalized_y = (y - y.min()) / (y.max() - y.min())

    return normalized_X, normalized_y

def apply_bias_trick(X: np.ndarray):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    
    return np.c_[np.ones(len(X)), X]

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    m = len(X)

    sigma = 0
    for i in range(m):
        sigma+=(_compute_hypothesis(X[i], theta)-y[i]) ** 2
    
    return (1.0/(2*m)) * (sigma)

def _compute_hypothesis(x_i, th):
    sum = 0
    for j in range(len(x_i)):
        sum+=x_i[j]*th[j]
    return sum

def gradient_descent(X, y, theta, alpha, num_iters, stop=False, stop_if_gt=False, debug=False):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    J_history = [] # Use a python list to save the cost value in every iteration
    init_cost = compute_cost(X, y, theta)
    if debug:
        print(f"theta {theta}, for cost: {init_cost}")

    theta = theta.copy() # optional: theta outside the function will not change

    for i in range(num_iters):
        theta_temps = []
        for j in range(len(theta)):
            partial_deriv = _compute_partial_derivative(X,y, theta, j)
            if debug:
                print(f"For theta {theta[j]}, id {j} we will subtract: {alpha*partial_deriv}")
            theta_temps.append(theta[j] - alpha*partial_deriv)

        # update theta
        theta = np.array(theta_temps)
        cost = compute_cost(X, y, theta)
        J_history.append(cost)
        if debug:
            print(f"Iter: {i} New theta arrived: {theta}, for cost: {cost}")

        if len(J_history) > 1:
            if stop: 
                if abs(J_history[-1] - J_history[-2]) < 1e-8:
                    return theta, J_history

            if stop_if_gt:
                if J_history[-2] < J_history[-1]:
                    return theta, J_history

    
    return theta, J_history

def _compute_partial_derivative(X, y, theta, _by_j):
    """
    _by_j: index to compute partial deriv by
    """
    m = len(X)
    sum = 0
    for i in range(m):
        temp = 0.0
        for j in range(len(theta)): # dim agnostic method
            temp+=theta[j]*X[i][j]
        sum+=(temp-y[i])*X[i][_by_j]

    return (1.0/m) * sum

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    X_t = np.transpose(X)
    pinv_x = np.matmul(np.linalg.inv(np.matmul(X_t, X)), X_t)
    return np.matmul(pinv_x, y)

def efficient_gradient_descent(X, y, theta, alpha, num_iters, stop_if_gt=False, debug=False):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    return gradient_descent(X, y, theta, alpha, num_iters, stop=True, stop_if_gt=stop_if_gt, debug=debug)

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}

    theta = compute_pinv(X=X_val, y=y_val)
    #theta = np.array([-1, 2])

    for alpha in alphas:
        print(f"Trying alpha: {alpha}")
        _, j_history = efficient_gradient_descent(
            X=X_train, 
            y=y_train, 
            theta=theta, 
            alpha=alpha, 
            num_iters=iterations,
            stop_if_gt=True
        )
        
        min_loss = j_history[0]
        for i in range(0, len(j_history), 100):
            if min_loss >= j_history[i]:
                min_loss = j_history[i]
            else:
                alpha_dict[alpha] = min_loss
                break

        alpha_dict[alpha] = min_loss

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
