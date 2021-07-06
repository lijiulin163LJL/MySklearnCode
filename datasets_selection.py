import numpy as np

def train_test_split(X,y,test_ratio=0.2,seed=None):
    """将数据x和标签y进行切分为训练集和测试集"""
    if seed:
        np.random.seed(seed)
    shuffled_index = np.random.permutation(len(X))

    test_size = int(len(X)*test_ratio)
    test_index = shuffled_index[:test_size]
    train_index = shuffled_index[test_size:]

    X_train = X[train_index]
    X_test= X[test_index]

    y_train = y[train_index]
    y_test = y[test_index]
    return X_train,X_test,y_train,y_test
