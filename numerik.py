from numbers import Number

import numpy as np


class Regression:

    def phi(self, j, x):
        if callable(self.model[j]):
            return self.model[j](x)
        else:
            return x ** self.model[j]

    def build_function(self, x):
        return sum(self.alpha[j] * self.phi(j, x) for j in range(self.m))

    def f(self, x):
        if isinstance(x, Number):
            return self.build_function(x)
        else:
            return np.fromiter((self.build_function(xi) for xi in x),
                               dtype=np.float64)

    def __init__(self, x, y, model, stats=False):
        idx = np.argsort(x)
        self.x = np.array(x)[idx]
        self.y = np.array(y)[idx]
        self.n = len(self.x)

        if isinstance(model, Number):
            self.model = list(range(model))
        else:
            self.model = list(model)
        self.m = len(self.model)

        self.A = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                self.A[i][j] = self.phi(j, self.x[i])
        AT = self.A.T

        self.B = AT @ self.A
        self.b = AT @ self.y

        if stats:
            self.beta = np.linalg.inv(self.B)
            self.alpha = self.beta @ self.b
            self.fx = self.f(self.x)
            self.r = self.A @ self.alpha - self.y
            self.var_fit = sum((self.y - self.fx) ** 2 / (self.n - self.m))
            self.cov = self.beta * self.var_fit
            self.var = np.fromiter((self.cov[k][k] for k in range(self.m)),
                                   dtype=np.float64)
            self.R2 = 1 - (sum((self.y - self.fx) ** 2) /
                           sum((self.y - np.mean(self.y)) ** 2))

        else:
            self.alpha = np.linalg.solve(self.B, self.b)
            self.fx = self.f(self.x)
