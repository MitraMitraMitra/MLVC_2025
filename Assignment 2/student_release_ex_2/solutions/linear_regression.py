import numpy as np

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# _add_bias(), fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################

class LinearRegression:
    """
    Simple linear regression with explicit bias term.
    """

    def __init__(self):
        self.w = None
        self.n_features_ = None

    @staticmethod
    def _add_bias(X):
        """
        Augment the feature matrix with a bias (intercept) column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Input data.

        Task
        ----
        Ensure X is two-dimensional. Then prepend a column of ones 
        so the bias w0 can be represented as part of the weight vector.

        Returns
        -------
        Xb : np.ndarray, shape (n_samples, n_features+1)
            Feature matrix with leading bias column.
        """

        X = np.asarray(X, dtype=np.float64)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Ensure X is two-dimensional.
        if X.ndim != 2:
            X = X.reshape(-1, 1)

        # Create a column of ones (bias) to be prepended.
        ones = np.ones((X.shape[0], 1), dtype = X.dtype)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return np.hstack([ones, X])

    def fit(self, X, y):
        """
        Fit linear regression model parameters using least squares.

        Model
        -----
        y ≈ w0 + w1 x1 + ... + wd xd

        Task
        ----
        * Add a bias column to X.
        * Solve for weight vector w that minimizes squared error:
              w* = argmin ||Xb w - y||^2
        * You may use NumPy’s least squares solver.
        * Store number of features (without bias) in self.n_features_.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearRegression
            Fitted model.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        # Add bias column to X.
        Xb = self._add_bias(X)

        # Solve for weight vector w that minimizes squared error:
        #       w* = argmin ||Xb w - y||^2
        self.w, *_ = np.linalg.lstsq(Xb, y, rcond = None)

        # Store number of features (without bias) in self.n_features_.
        if X.ndim == 1:
            self.n_features_ = 1
        else:
            self.n_features_ = X.shape[1]
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return self

    def predict(self, X):
        """
        Predict target values for given input.

        Task
        ----
        * Verify that the model has been fit (weights exist).
        * Add a bias column to X.
        * Check that feature dimensions match training.
        * Return the dot product Xb @ w.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Verify that the model has been fit (weights exist).
        if self.w is None:
            raise ValueError("This model has been not been fitted yet.")

        # Add a bias column to X.
        X = np.asarray(X, dtype=np.float64)
        Xb = self._add_bias(X)

        # Check that feature dimensions match training.
        expected_n_cols = self.n_features_ + 1
        if Xb.shape[1] != expected_n_cols:
            raise ValueError(
                "X has "
                + str(Xb.shape[1] - 1)
                + " features, but the model was trained with "
                + str(self.n_features_)
                + " features."
            )
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return Xb @ self.w
