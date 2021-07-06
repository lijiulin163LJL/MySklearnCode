import numpy as np

def accuracy_score(y_true,y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0],\
        "y_true的维度必须和y_predict 保持一致"
    return np.sum(y_predict==y_true)/len(y_true)


def mean_squre_error(y_true,y_predict):
    """均方误差"""
    return np.sum((y_true-y_predict)**2)/len(y_true)

def r_mean_squre_error(y_true,y_predict):
    """均方根误差"""
    return np.sqrt(mean_squre_error(y_true,y_predict))

def mean_absolute_error(y_true,y_predict):
    """绝对值误差"""
    return np.sum(np.absolute(y_true-y_predict))/len(y_true)

def r2_score(y_true,y_predict):
    return 1-mean_squre_error(y_true,y_predict)/np.var(y_true)