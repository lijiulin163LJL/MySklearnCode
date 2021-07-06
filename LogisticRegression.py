import numpy as np
from .metrics import r2_score

class LinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit(self,X_train,y_train):
        """计算参数theta"""
        Xb = np.hstack([np.ones((len(X_train),1)),X_train])  #计算Xb
        self._theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self

    def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):
        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i*(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta,n_iters,t0=5,t1=50):
            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)

            for cur_iter in range(n_iters):
                indexs = np.random.permutation(m)
                X_b_new = X_b[indexs]
                y_new = y[indexs]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter*m+i) * gradient
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters,t0,t1)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
        """计算损失函数"""
        def J(theta,X_b,y):
            try:
                return np.sum((y-X_b.dot(theta))**2)/len(X_b)
            except:
                return float("inf")

        def dJ(theta,X_b,y):
            return X_b.T.dot(X_b.dot(theta)-y)*2/len(X_b)

        def gradient_descent(X_b,y,inital_theta,eta,n_iters=1e4,epsilon=1e-8):
            theta = inital_theta
            cur_iter = 0
            while cur_iter<n_iters:
                dj = dJ(theta,X_b,y)
                last_theta = theta
                theta = theta-eta*dj
                if (abs(J(theta,X_b,y)-J(last_theta,X_b,y))) < epsilon:
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b,y_train,initial_theta,eta,n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self,X_test):
        Xtb = np.hstack([np.ones((len(X_test),1)),X_test])
        return Xtb.dot(self._theta)

    def score(self,X_test,y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"