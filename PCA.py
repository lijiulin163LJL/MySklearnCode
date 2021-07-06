import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.scale_ = None

    def fit(self,X,eta,n_iters=1e4):
        def demean(X):
            pass
        def f(w,X):
            pass

        def df(w,X):
            return w/np.linalg.norm(w)

        def first_component(X,initial_w,eta=0.01,n_iters=1e4,epsilon=1e-8):
            pass

    def __repr__(self):
        return "PCA(n_components={})".format(self.n_components)
