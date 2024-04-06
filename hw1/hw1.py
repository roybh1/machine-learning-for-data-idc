###### Your ID ######
# ID1: 208951111
# ID2: 322315755
#####################

# imports 
import numpy as np
import pandas as pd

debug = False

def set_debug(v):
    """
    set prints in the code for debugging
    """
    debug=v

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
    normalized_X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    normalized_y = (y - y.mean(axis=0)) / (y.max(axis=0) - y.min(axis=0))

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

    m = X.shape[0]

    return (1.0/(2*m)) * np.sum((np.dot(X, theta) - y) ** 2)

def gradient_descent(X, y, theta, alpha, num_iters, stop=False):
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

    theta = theta.copy() # optional: theta outside the function will not change

    for i in range(num_iters):
        if debug and i % 100 == 0:
            pass
            #print(f"Iteration num {i}")
        theta_temps = []
        for j in range(len(theta)):
            partial_deriv = _compute_partial_derivative(X, y, theta, j)
            theta_temps.append(theta[j] - alpha*partial_deriv)

        # update theta
        theta = np.array(theta_temps)
        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if len(J_history) > 1:
            if stop: 
                if J_history[-2] - J_history[-1] < 1e-8:
                    return theta, J_history
    
    return theta, J_history

def _compute_partial_derivative(X, y, theta, _by_j):
    """
    _by_j: index to compute partial deriv by
    """
    return np.sum((np.dot(X, theta) - y) * X[:, _by_j]) / X.shape[0]

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

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
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

    return gradient_descent(X, y, theta, alpha, num_iters, stop=True)

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

    np.random.seed(42)
    init_theta = np.random.random(size=np.size(X_train, 1))

    for alpha in alphas:
        if debug:
            print(f"Trying alpha: {alpha}")
        theta, _= efficient_gradient_descent(
            X=X_train, 
            y=y_train, 
            theta=init_theta, 
            alpha=alpha, 
            num_iters=iterations,
        )
        
        alpha_dict[alpha] = compute_cost(X_val, y_val, theta)

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
    np.random.seed(42)

    features_to_test = []
    best_features_per_iter = {i:[] for i in range(6)}

    for i in range(1,6):
        min_cost = 1 # arbitrarily large 
        for feature in range(np.size(X_train,1)):
            if feature not in best_features_per_iter[i]:
                features_to_test = best_features_per_iter[i-1] + [feature] # get the best features from previous iter
                if debug:
                    print(f"Considering feature: {feature}")
                x_val = apply_bias_trick(X_val[:, features_to_test])
                x_train = apply_bias_trick(X_train[:, features_to_test])
                init_theta = np.random.random(size=np.size(x_train, 1))
                if debug:
                    print(f"Considering feature set: {features_to_test}")
                theta, _ = efficient_gradient_descent(
                    X=x_train,
                    y=y_train, 
                    theta=init_theta, 
                    alpha=best_alpha, 
                    num_iters=iterations, 
                )
                current_cost = compute_cost(X=x_val, y=y_val, theta=theta)
                if not best_features_per_iter[i]:
                    best_features_per_iter[i] = features_to_test
                    min_cost = current_cost
                elif min_cost > current_cost:
                    if debug:
                        print(f"found new min cost for {i} features: \n\
                                old min_cost {min_cost}, new min_cost: {current_cost}. diff = {min_cost - current_cost} \n\
                                old selected_features {best_features_per_iter[i]}, new selected_features: {features_to_test}")
                    min_cost = current_cost
                    best_features_per_iter[i] = features_to_test 

    return best_features_per_iter[5] 


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    new_df = df.copy()
    new_columns = {}
    df_length = len(df.columns)
    for i in range(df_length):
        col1 = df[df.columns[i]]
        col1_name = df.columns[i]
        for j in range(i, df_length):
            col2 = df[df.columns[j]]
            col2_name = df.columns[j]
            new_label = f"{col1_name} * {col2_name}"
            new_columns[new_label] = col1 * col2
    new_columns_df = pd.DataFrame(new_columns)
    new_df = pd.concat([df, new_columns_df], axis=1)    
    return new_df
