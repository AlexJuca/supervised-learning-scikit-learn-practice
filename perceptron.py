import numpy as np

class Perceptron:
    """ Perceptron classifier

    Parameters
    ------------
    alpha : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset

    Attributes
    -----------
    weights : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch

    """
    def __init__(self, alpha=0.01, n_iterations=10):
        self.alpha = alpha
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iterations):
            errors = 0
            for xi, target in zip(X, y):
                update = self.alpha * (target - predict(self, xi))
                self.weights[1:] += update * xi
                print(self.weights)
                self.weights[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
                print("--- ERROS ---")
                print(self.errors_)
                # if(self.errors_[len(self.errors_) - 1] == 0):
                 #   break

        return self

def net_input(self, X):
    """Calculate net input """
    return np.dot(X, self.weights[1:]) + self.weights[0]

def predict(self, X):
    """Return the class label after unit step:"""
    return np.where(net_input(self, X) >= 0.0, 1, -1)
