import numpy as np
import matplotlib.pyplot as plt

import PS1.src.util as util
from PS1.src.linear_model import LinearModel


def main(tau, train_path, eval_path, png_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    mse = np.mean((y_pred - y_eval)**2)
    print(f'MSE={mse}')

    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(png_path)
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau: int):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = float(tau)
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            W = np.diag(
                np.exp(-np.sum(
                    (self.x - x[i])**2, axis=1) / (2 * self.tau**2)))
            y_pred[i] = np.linalg.inv(self.x.T.dot(W).dot(self.x)).dot(
                self.x.T).dot(W).dot(self.y).T.dot(x[i])

        return y_pred
        # *** END CODE HERE ***


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print(
            "Usage: python PS01b.py <tau> <train_path> <eval_path> <png_path>")
    else:
        tau, train_path, eval_path, png_path = sys.argv[1], sys.argv[
            2], sys.argv[3], sys.argv[4]
        main(tau, train_path, eval_path, png_path)
