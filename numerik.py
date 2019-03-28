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


class CubicSpline:

    def auto_k(self, x, k=0):
        if x < self.x[0]:
            return 0
        elif k == self.n-1:
            return k
        elif self.x[k] <= x <= self.x[k+1]:
            return k
        else:
            k += 1
            return self.auto_k(x, k)

    def build_S(self, x, k):
        dx = x - self.x[k]
        return self.a[k] + self.b[k]*dx + self.c[k]*dx**2 + self.d[k]*dx**3

    def S(self, x):
        if isinstance(x, Number):
            return self.build_S(x, self.auto_k(x))
        else:
            return np.fromiter(
                (self.build_S(xi, self.auto_k(xi)) for xi in x),
                dtype=np.float64
            )

    def build_Sp(self, x, k):
        dx = x - self.x[k]
        return self.b[k] + 2*self.c[k]*dx + 3*self.d[k]*dx**2

    def Sp(self, x):
        if isinstance(x, Number):
            return self.build_Sp(x, self.auto_k(x))
        else:
            return np.fromiter(
                (self.build_Sp(xi, self.auto_k(xi)) for xi in x),
                dtype=np.float64
            )

    def build_Spp(self, x, k):
        dx = x - self.x[k]
        return 2*self.c[k] + 6*self.d[k]*dx

    def Spp(self, x):
        if isinstance(x, Number):
            return self.build_Spp(x, self.auto_k(x))
        else:
            return np.fromiter(
                (self.build_Spp(xi, self.auto_k(xi)) for xi in x),
                dtype=np.float64
            )

    def __init__(self, x, y, natural=None, clamped=None):

        idx = np.argsort(x)
        self.x = np.array(x)[idx]
        self.y = np.array(y)[idx]
        self.n = len(self.x) - 1
        self.h = np.fromiter(
            (self.x[k+1] - self.x[k] for k in range(self.n)),
            dtype=np.float64
        )

        self.A = np.zeros((self.n-1, self.n-1))
        self.ypp = np.zeros(self.n+1)
        self.vb = np.fromiter(
            (6 * (((self.y[k+2] - self.y[k+1]) / self.h[k+1])
                  - ((self.y[k+1] - self.y[k]) / self.h[k]))
             for k in range(self.n-1)),
            dtype=np.float64
        )

        if natural is not None:
            for i in range(self.n-1):
                if i > 0:
                    self.A[i-1][i] = self.A[i][i-1] = self.h[i]
                self.A[i][i] = 2 * (self.h[i] + self.h[i+1])
            self.AI = np.linalg.inv(self.A)
            self.ypp = np.zeros(self.n+1)
            if isinstance(natural, Number):
                self.ypp[0] = self.ypp[-1] = natural
            else:
                self.ypp[0] = natural[0]
                self.ypp[-1] = natural[1]
            self.vb[0] -= self.h[0] * self.ypp[0]
            self.vb[-1] -= self.h[-1] * self.ypp[-1]
            self.vz = self.AI @ self.vb
            self.ypp[1:-1] = self.vz

        elif clamped is not None:
            self.A[0][0] = (3 / 2) * self.h[0] + 2 * self.h[1]
            self.A[-1][-1] = 2 * self.h[self.n-2] + (3 / 2) * self.h[self.n-1]
            for i in range(1, self.n-1):
                self.A[i-1][i] = self.A[i][i-1] = self.h[i]
                if i < self.n-2:
                    self.A[i][i] = 2 * (self.h[i] + self.h[i+1])
            self.AI = np.linalg.inv(self.A)
            if isinstance(clamped, Number):
                yp0 = ypn = clamped
            else:
                yp0 = clamped[0]
                ypn = clamped[1]
            self.vb[0] -= 3 * ((self.y[1] - self.y[0]) / self.h[0] - yp0)
            self.vb[-1] -= 3 * (ypn - (self.y[self.n] -
                                       self.y[self.n-1]) / self.h[self.n-1])
            self.vz = self.AI @ self.vb
            self.ypp[1:-1] = self.vz
            self.ypp[0] = (
                (3 / self.h[0])
                * ((self.y[1] - self.y[0]) / self.h[0] - yp0)
                - 0.5 * self.ypp[1]
            )
            self.ypp[-1] = (
                (3 / self.h[self.n-1])
                * (ypn - (self.y[self.n] - self.y[self.n-1])
                   / self.h[self.n-1])
                - 0.5 * self.ypp[self.n-1]
            )

        self.a = self.y[:self.n]
        self.b = np.fromiter(
            (((self.y[k+1] - self.y[k]) / self.h[k])
             - ((2*self.ypp[k] + self.ypp[k+1]) / 6) * self.h[k]
             for k in range(self.n)),
            dtype=np.float64
        )
        self.c = self.ypp[:self.n] / 2
        self.d = np.fromiter(
            ((self.ypp[k+1] - self.ypp[k]) / (6 * self.h[k])
             for k in range(self.n)),
            dtype=np.float64
        )
