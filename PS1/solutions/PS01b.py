import numpy as np
import PS1.src.util as util

from PS1.src.linear_model import LinearModel


def main(train_path, eval_path, pred_path, png_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, png_path)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros((n, 1))

        y = np.reshape(y, (m, 1))

        while True:
            prev_theta = np.copy(self.theta)
            predictions = 1 / (1 + np.exp(-x @ self.theta))
            gradient = (1 / m) * (x.T @ (predictions - y))
            S = predictions * (1 - predictions)
            hessian = (x.T @ (S.T * x)) / m

            self.theta -= np.linalg.pinv(hessian) @ gradient

            if np.linalg.norm(self.theta - prev_theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE ***


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python PS01b.py <train_path> <eval_path> <pred_path>")
    else:
        train_path, eval_path, pred_path, png_path = sys.argv[1], sys.argv[
            2], sys.argv[3], sys.argv[4]
        main(train_path, eval_path, pred_path, png_path)
