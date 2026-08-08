import numpy as np

import PS1.src.util as util
from PS1.src.linear_model import LinearModel


def main(train_path, eval_path, pred_path, png_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, png_path)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n + 1)

        y = np.reshape(y, (-1))

        # parameters
        phi = np.sum(y == 1) / m
        mu_0 = np.sum(x[y == 0], axis=0) / (m - sum(y == 1))
        mu_1 = np.sum(x[y == 1], axis=0) / sum(y == 1)
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) +
                 (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / m

        # theta
        sigma_inv = np.linalg.pinv(sigma)
        self.theta[1:] = sigma_inv.dot(mu_1 - mu_0)
        self.theta[0] = (mu_0 + mu_1).T.dot(
            sigma_inv.dot(mu_0 - mu_1) - np.log((1 - phi) / phi)) / 2

        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print(
            "Usage: python PS01e.py <train_path> <eval_path> <pred_path> <png_path>"
        )
    else:
        train_path, eval_path, pred_path, png_path = sys.argv[1], sys.argv[
            2], sys.argv[3], sys.argv[4]
        main(train_path, eval_path, pred_path, png_path)
