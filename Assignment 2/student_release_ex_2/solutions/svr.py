# ---------- epsilon_svr.py ----------
import cvxopt
import numpy as np

########### TO-DO ###########
# Implement the methods marked with "BEGINNING OF YOUR CODE" and "END OF YOUR CODE":
# fit(), predict()
# Do not change the function signatures
# Do not change any other code
#############################

class EpsilonSVR:
    """
    ε-Support Vector Regression (dual form).

    This implementation works only with sklearn-compatible kernels.
    The kernel must be a callable with signature
        K(X, Y) -> ndarray of shape (len(X), len(Y)),
    e.g. an instance from sklearn.gaussian_process.kernels (RBF, Matern, etc.).

    Dual optimization problem
    -------------------------
    Given training data {(x_i, y_i)}_{i=1..n}, we solve for alpha, alpha* ∈ R^n:

        minimize   1/2 (alpha - alpha*)^T K (alpha - alpha*) + epsilon 1^T (alpha + alpha*) - y^T (alpha - alpha*)
        subject to 0 ≤ alpha_i ≤ C,
                   0 ≤ alpha*_i ≤ C,
                   1^T (alpha - alpha*) = 0.

    Here K is the kernel Gram matrix. The solution defines coefficients
    (alpha - alpha*) that weight support vectors in the prediction.

    Prediction
    ----------
    For a test point x,
        f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b.

    Notes
    -----
    * C > 0 controls the regularization strength (penalty for large alpha, alpha*).
    * epsilon ≥ 0 defines the “epsilon-insensitive” zone around targets y_i where
      deviations incur no loss.
    * Input normalization (scaling by max norm) can be enabled for
      numerical stability.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel=None, normalize=True):
        if kernel is None:
            raise ValueError("Provide an sklearn-compatible kernel instance (callable K(X, Y)).")
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.__sk_kernel = kernel
        self.__normalize = bool(normalize)

        # Learned params
        self.__a = None             # a
        self.__a_star = None        # a*
        self.__coef = None          # (a - a*)
        self.__bias = 0.0
        self.__training_X = None    # numpy, scaled if normalize=True
        self.__norm = 1.0
        self.__support_mask = None  # boolean mask over training samples

    # ---- Sklearn-kernel bridge ----
    def _kernel(self, X1_np, X2_np):
        return self.__sk_kernel(X1_np, X2_np)

    def fit(self, X, y):
        """
        Fit ε-SVR in the dual form using quadratic programming.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input vectors.
        y : array-like of shape (n_samples,)
            Training target values.

        Task
        ----
        * Optionally normalize X for stability.
        * Build the kernel Gram matrix K(X, X).
        * Formulate the dual quadratic program in variables z = [alpha; alpha*].
        * Solve with a QP solver (e.g. cvxopt).
        * Extract alpha, alpha*, coefficients (alpha - alpha*), and identify support vectors.
        * Compute bias b from KKT conditions using near-margin samples.

        Returns
        -------
        self : EpsilonSVR
            Fitted model with dual variables and bias stored.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n, _ = X.shape

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        small_num=1e-5
        #1. Optionally normalize X for stability.
        if self.__normalize:
            self.__norm=np.max(np.linalg.norm(X,axis=1))
            
            if self.__norm<small_num:
                self.__norm=1.0
            X_normed=X/self.__norm 
        else:
                self.__norm=1.0
                X_normed=X.copy()
        self.__training_X=X_normed

        #2. Build the kernel Gram matrix K(X,X')
        K=self._kernel(X_normed, X_normed)
        
        #3. Formulate the dual quadratic program in variables z = [alpha; alpha*].
        #P Matrix (Quadratic part) =>(alpha - alpha*)^T K (alpha - alpha*)
        P_block=np.block([[K, -K], [-K, K]])
        P=cvxopt.matrix(P_block)
        #q vector (Linear part) q_a=alpha, q_b=alpha*
        ones=np.ones(n)
        q_a=self.epsilon*ones-y
        q_b=self.epsilon*ones+y
        #make them column vector
        q=cvxopt.matrix(np.hstack([q_a, q_b]), (2*n,1))
        #Set up equality constraint Az=b A=[1,-1]*z=0
        A_array=np.hstack([ones, -ones])
        b=cvxopt.matrix(0.0)
        A=cvxopt.matrix(A_array,(1,2*n))
        #Set up inequality constraint Gz<h 0<=a_i<=C, 0<=b_i<=C
        #G matrix -->(4n x 2n)
        Is=np.eye(n)
        zeros=np.zeros((n,n))
        G_array=[np.hstack([-Is, zeros]),np.hstack([ Is, zeros]),np.hstack([zeros, -Is]),np.hstack([zeros,  Is])]
        G=cvxopt.matrix(np.vstack(G_array))
        #set up the h vector (4nby1)
        h_c=self.C*ones
        h_rows=np.hstack([np.zeros(n), h_c, np.zeros(n), h_c])
        h=cvxopt.matrix(h_rows,(4*n,1))

        #4. Solve with a QP solver (e.g. cvxopt).
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        #5. Extract alpha, alpha*, coefficients (alpha - alpha*), and identify support vectors.
        dual_var=np.array(solution['x']).ravel()
        self.__a = dual_var[:n] # a
        self.__a_star = dual_var[n:] # b

        #Clipping noise near the boundaries
        self.__a = np.clip(self.__a, 0, self.C)
        self.__a_star = np.clip(self.__a_star, 0, self.C)
        self.__a[self.__a < small_num] = 0.0
        self.__a_star[self.__a_star < small_num] = 0.0
        self.__a[self.__a > self.C - small_num] = self.C
        self.__a_star[self.__a_star > self.C - small_num] = self.C

        # Calculate coefficients + support mask
        self.__coef = self.__a - self.__a_star
        self.__support_mask = (np.abs(self.__coef) > small_num)

        # 6. Compute bias b from conditions
         # Free Support Vectors (fsvs) where 0 < a_i < C or 0 < b_i < C
        fsv_a = (self.__a > small_num) & (self.__a < self.C - small_num)
        fsv_b = (self.__a_star > small_num) & (self.__a_star < self.C - small_num)
        
        # A sample is an fsv if one of its multipliers is strictly between 0 and C
        fsv_idxs = np.where(fsv_a | fsv_b)[0]
        
        if len(fsv_idxs) > 0:
            # Calculate K * self.__coef (K * (a - b))
            K_w = K @ self.__coef
            
            bestimates = []
            
            for k in fsv_idxs:
                # If 0 < a_i < C: y_i - f(x_i) = epsilon  -->  biaas = y_i - epsilon - (K * w)_i
                if fsv_a[k]:
                    bestimates.append(y[k] - self.epsilon - K_w[k])
                # If 0 < b_i < C: f(x_i) - y_i = epsilon  -->  bias = y_i + epsilon - (K * w)_i
                elif fsv_b[k]:
                    bestimates.append(y[k] + self.epsilon - K_w[k])
            
            if bestimates:
                self.__bias = np.mean(bestimates)
            else:
                self.__bias = 0.0 
        else:
            # Fallback if no Free SVs found 
            print("No Free Support Vectors found!!!")
            # If no boundary than bias is zero
            self.__bias = 0.0 




        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        
        return self

    def _k_train_test(self, Xtest_scaled):
        return self._kernel(self.__training_X, Xtest_scaled)  # (n_train, n_test)

    def predict(self, X):
        """
        Predict regression outputs for new data using the dual form.

        For each test point x, compute:
            f(x) = Σ_i (alpha_i - alpha*_i) K(x_i, x) + b,
        where the sum runs over support vectors.

        Parameters
        ----------
        X : array-like of shape (m, n_features)
            Test input vectors.

        Returns
        -------
        y_pred : np.ndarray of shape (m,)
            Predicted target values.
        """

        if self.__coef is None:
            raise RuntimeError("Model is not fit yet.")

        X = np.asarray(X, dtype=np.float64)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        #Checking the normalization factor 
        if self.__norm != 1.0:
            X_scaled = X / self.__norm
        else:
            X_scaled = X.copy()
            
        # 2. Compute test Kernel matrix (K(X_train, X_test))
        K_test = self._k_train_test(X_scaled) 
        
        # 3. Compute predictiom
        
        # Filter K_test to only include rows corresponding to SVs
        K_sv = K_test[self.__support_mask, :]
        
        # Filter coefficients to only include non-zero ones
        coef_sv = self.__coef[self.__support_mask]
        
        # Calculate the weihgted sum of kernels
        y_pred = np.dot(coef_sv, K_sv) + self.__bias

        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        return y_pred
    

