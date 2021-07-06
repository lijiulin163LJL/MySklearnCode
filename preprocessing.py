import numpy as np

class StandScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self,x):
        """根据训练数据x获得数据的均值和方差"""

        self.mean_ = np.array([np.mean(x[:,i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:,i]) for i in range(x.shape[1])])
        return self
    def transform(self,x):
        """将x根据标准化处理"""
        resx = np.empty(x.shape,dtype=float)
        for col in range(x.shape[1]):
            resx[:,col] = (x[:,col]-self.mean_[col])/self.scale_[col]
        return resx