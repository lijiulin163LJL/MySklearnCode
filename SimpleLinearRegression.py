import numpy as np
from .metrics import r2_score

class SimpleLinearRegression:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        """根据训练集数据x_trian和y_trian进行训练"""
        assert x_train.ndim == 1,"训练数据维度必须为1"
        assert x_train.dtype == "float64","数据需要为float类型"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        p = (x_train-x_mean).dot(y_train-y_mean)
        q = (x_train-x_mean).dot(x_train-x_mean)

        self.a_ = p/q
        self.b_ = y_mean-self.a_*x_mean
        return self

    def predict(self,x_test):
        """根据a,b预测数据"""
        return np.array([self._predict(i) for i in x_test])

    def _predict(self,x_test):
        return self.a_*x_test + self.b_

    def score(self,x_test,y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "SimpleLinearRegression"