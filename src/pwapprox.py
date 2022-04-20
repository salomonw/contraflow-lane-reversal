from gurobipy import *
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import random as random

class pwapprox:
    def __init__(self, x,y, k=2):
        self.x = x
        self.y = y
        self.k = k

    def fit_convex_pw(self):
        model = Model("fit_convex_pw")
        model.setParam('OutputFlag', 1)
        m = len(self.x)
        n = 5
        k = self.k
        y = [model.addVar(name='y_' + str(i)) for i in range(m)]
        g = [model.addVar(name='g_' + str(i)) for i in range(m)]
        model.update()
        obj = sum([(self.y[i] - y[i])*(self.y[i] - y[i]) for i in range(m)])
        model.update()
        for i in range(m):
            for j in range(m):
                model.addConstr(y[j] >= y[i] + g[i]*(self.x[j]-self.x[i]))
        model.update()
        model.setObjective(obj)
        model.update()
        model.optimize()

        self.y_hat = []
        self.g_hat = []
        for i in range(m):
            self.y_hat.append(y[i].X)
            self.g_hat.append(g[i].X)
        return self.y_hat, self.g_hat

    def random_initial_partition(self):
        random.seed(8)#6
        np.random.seed(8)#6
        non_empty = True
        while non_empty == True:
            non_empty = True
            K = range(self.k)
            X = range(len(self.x))
            p = [random.uniform(min(self.x), max(self.x)) for j in K]
            P = {}
            for j in K:
                P[j] = []
            for i in X:
                for j in K:
                    a = [(self.x[i]-p[s])**2 for s in K]
                    a.remove((self.x[i]-p[j])**2)
                    if all((self.x[i]-p[j])**2 < r for r in a) == True:
                        P[j].append(i)

                    if len(P[j])==0:
                        non_empty=False
        self.P = P
        return P

    def update_P(self):
        K = range(self.k)
        f = []
        for j in K:
            for i in range(len(self.x)):
                f = max([self.a[s]*self.x[i] + self.b[s] for s in K])
                if f == self.a[j]*self.x[i] + self.b[j]:
                    self.P[j].append(i)

    def get_intersections(self):
        indx = np.argsort(self.a)
        a_sort = [self.a[i] for i in indx]
        b_sort = [self.b[i] for i in indx]
        self.x_part = []
        for i in range(self.k-1):
            self.x_part.append((b_sort[i]-b_sort[i+1])/(a_sort[i+1]-a_sort[i]))

    def run_least_sq_partition_alg(self, L=400):
        K = range(self.k)
        L = range(L)
        flag = True
        for l in L:
            a = []
            b = []
            for j in K:
                x = [self.x[i] for i in self.P[j]]
                y = [self.y[i] for i in self.P[j]]
                if len(x)>1:
                    aj , bj, r_value, p_value, std_err = stats.linregress(x,y)
                else:
                    return False
                a.append(aj)
                b.append(bj)
            self.a = a
            self.b = b
            self.update_P()
        self.y_hat = [max([a[j]*i+b[j] for j in range(self.k)]) for i in self.x]
        self.x_part = [self.x[i] for i in range(len(self.x)-1) if self.y_hat[i+1]-self.y_hat[i] ]
        self.get_intersections()
        self.get_coeffs()
        return flag


    def get_coeffs(self):
        indx = np.argsort(self.a)
        self.a_sort = [self.a[i] for i in indx]
        self.b_sort = [self.b[i] for i in indx]
        #self.coeffs = a_sort

    def get_RMS(self):
        m = len(self.y)
        return np.sqrt(1/m * sum([(self.y[i] - self.y_hat[i])**2 for i in range(m)]))

    def fit_convex_boyd(self, N, L=400):
        self.rms_vec = []
        self.a_list = []
        self.b_list = []
        self.thetas = []
        for i in range(N):
            a = False
            while a==False:
                self.random_initial_partition()
                a = self.run_least_sq_partition_alg(L=L)
                if a!=False:
                    self.rms_vec.append(self.get_RMS())
                    self.get_coeffs()
                    self.a_list.append(self.a_sort)
                    self.b_list.append(self.b_sort)
                    self.thetas.append(self.x_part)

    def plot_rms(self, ax):
        ax.hist(self.rms_vec)


    def f_hat(self, x):
        return max([self.y_hat[i] + self.g_hat[i]*(x-self.x[i]) for i in range(len(self.x))])

    def plot_fit(self, ax):
        ax.scatter(self.x, self.y, marker='.')
        y_hat = self.y_hat
        ax.plot(self.x, y_hat, color='red')
        for i in self.x_part:
            ax.axvline(i, color='k', linestyle=':')


def get_theta(fun):
    return fun.fit_breaks[1:-1]

def get_beta(fun):
    return fun.slopes[1:]


def eval_travel_time(x, fcoeffs):
    return sum([fcoeffs[i]*x**(i+1) for i in range(len(fcoeffs))])


def eval_pw(a, b, theta, x):
    theta2 = theta.copy()
    theta2.append(1000)
    for i in range(len(theta2)-1):
        if (theta2[i] <= x) and (theta2[i+1] > x):
            y = b[i] + a[i]*x
    return y

def get_approx_fun(fcoeffs, range_=[0,2], nlines=3, plot=2, ax=None):
    # Generate data
    x = [i  for i in list(np.linspace(range_[0], range_[1], 100))]
    y = [eval_travel_time(i, fcoeffs) for i in x]
    pws = pwapprox(x, y, k=nlines)
    pws.fit_convex_boyd(N=30, L=30)
    rms = min(pws.rms_vec)
    i = pws.rms_vec.index(rms)
    a = pws.a_list[i]
    b = pws.b_list[i]
    theta = pws.thetas[i]
    theta.insert(0,0)
    theta.append(range_[1])
    if plot == 1 :
        if nlines ==2 :
            ax.plot(x, y , label = '$t(x)$')#, color='k')
        ypws = [eval_pw(a,b, theta[0:-1], i) for i in x]
        p = ax.plot(x, ypws, '-', label='$\hat{t}(x)$, $n=$'+str(nlines))
        color = p[0].get_color()
        j=0
        for th in theta:
            if th > range_[0] and th< range_[1]:
                j+=1
                ax.text(th+0.02, 2.8 ,'$\\theta^{('+str(j)+')}$' ,color=color)
                ax.axvline(x=th, linestyle=':', color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('t(x)')
        ax.set_xlim((range_[0],range_[1]))
        plt.legend()
        plt.tight_layout()
    if plot == 2:
        fig, ax = plt.subplots(2)
        ax[0].plot(x, y , label = 'Original', color='k')
        ypws = [eval_pw(a,b, theta[0:-1], i) for i in x]
        ax[0].plot(x, ypws, label='pwlinear', color='red')
        for th in theta:
            ax[0].axvline(x=th, linestyle=':')
        plt.grid()
        plt.xlabel('$x$')
        plt.ylabel('$t(x)$')
        plt.legend()
        plt.tight_layout()
        pws.plot_rms(ax=ax[1])
        plt.show()
    return  theta, a, rms

