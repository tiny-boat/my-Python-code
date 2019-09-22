#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from warnings import filterwarnings


filterwarnings('ignore')


def prinTime(func):
    '''wraaper used for printing the running time of function'''
    from time import time

    def wrapper(*args, **kargs):
        startTime = time()
        funcResult = func(*args, **kargs)
        endTime = time()
        print('    Running time:      %.4f s\n' % (endTime - startTime))
        return funcResult

    return wrapper


def goldenSection(x0, d0, func, start=0, end=2, epsilon=1e-4):
    '''Golden Section Method for exact line search'''

    # find two insertion points using fixed ratio
    from math import sqrt
    ratio = sqrt(5) / 2 - 0.5
    intervalLen = end - start
    middleL = start + (1 - ratio) * intervalLen
    middleR = start + ratio * intervalLen

    while intervalLen >= epsilon:
        # update start or end point and two insertion points
        if func(x0 + middleL * d0) > func(x0 + middleR * d0):
            start = middleL
            intervalLen = end - start
            middleL = middleR
            middleR = start + ratio * intervalLen
        else:
            end = middleR
            intervalLen = end - start
            middleR = middleL
            middleL = start + (1 - ratio) * intervalLen

    return (start + end) / 2


def inexactLineSearch(x0, d0, func, grad, start=0, end=1e10, rho=0.1,
                      sigma=0.4, criterion='Wolfe Powell'):
    '''Inexact Line Search Method with four available criterion:
    1.Armijo Goldstein
    2.Wolfe Powell
    3.Strong Wolfe Powell
    4.Simple'''

    # reduce unnecessary caculations in loop
    f0, df0 = func(x0), np.dot(grad(x0), d0)
    rhoDf0 = rho * df0
    boundary3 = sigma * df0
    boundary4 = sigma * abs(df0)

    while True:
        alpha = (start + end) / 2
        boundary1 = f0 + rhoDf0 * alpha
        boundary2 = f0 + boundary3 * alpha
        fAlpha = func(x0 + alpha * d0)
        dfAlpha = np.dot(grad(x0 + alpha * d0), d0)

        # different criterions have same condition1
        # to avoid too large alpha
        condition1 = (fAlpha <= boundary1)
        # different criterions have different condition2
        # to avoid too small alpha
        if criterion == 'Armijo Goldstein':
            condition2 = (fAlpha >= boundary2)
        elif criterion == 'Wolfe Powell':
            condition2 = (dfAlpha >= boundary3)
        elif criterion == 'Strong Wolfe Powell':
            condition2 = (abs(dfAlpha) <= boundary4)
        else:
            condition2 = True

        # update start or end point or stop iteration
        if condition1 is False:
            end = alpha
        elif condition2 is False:
            start = alpha
        else:
            minPoint = alpha
            minValue = fAlpha
            break

    return minPoint


def stopCondition(x0, x1, f0, f1, grad1, eps1, eps2, gradneed=True):
    '''create stop condition controlled by eps1 and eps2
    using informations including x0, x1, f(x0),f(x1) and gradient(x1)'''

    normX0 = np.linalg.norm(x0)
    absF0 = abs(f0)
    if normX0 > eps2 and absF0 > eps2:
        diffF = abs(f1-f0)/absF0
        diffX = np.linalg.norm(x1-x0)/normX0
    else:
        diffF = abs(f1-f0)
        diffX = np.linalg.norm(x1-x0)
    if gradneed is True:
        return diffF > eps2 and diffX > eps2 \
               and np.linalg.norm(grad1) > eps1
    else:
        return diffF > eps2 and diffX > eps2


@prinTime
def gradDescent(x0, func, grad, eps1=1e-4, eps2=1e-5, search=True,
                exactsearch=False, disp=False):

    if disp is True:   # save initial point
        start = x0

    f0 = func(x0)
    grad0 = grad(x0)

    if np.linalg.norm(grad0) > eps1:
        iterNum, d0, stopCon = 0, - grad0, True
        while stopCon:

            # create new iteration point
            if search is True:
                if exactsearch is False:
                    alpha = inexactLineSearch(x0, d0, func, grad,
                                              criterion='Simple')
                else:
                    alpha = goldenSection(x0, d0, func)
                x1 = x0 + alpha * d0
            else:
                x1 = x0 + 0.001 * d0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0, x1, f0, f1, grad1, eps1, eps2)

            # update x0, d0 for computing next iteration point
            # update f0, grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, - grad1

            iterNum += 1

    if disp is True:
        if search is True:
            if exactsearch is True:
                print('\nMethod: Gradient Descent (exact line-search)\n')
            else:
                print('\nMethod: Gradient Descent (unexact line-search)\n')
        else:
            print('\nMethod: Gradient Descent (no line-search)\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(x0))
        print('    Function value:    {0}'.format(f0))
        print('    Gradient:          {0}'.format(grad0))
        print('    Iteration number:  {0}'.format(iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0


@prinTime
def newton(x0, func, grad, hess, eps1=1e-4, eps2=1e-5, search=True,
           exactsearch=False, disp=False):

    if disp is True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        '''
        if hess(x0) is not positive definite
        replace hess(x0) by hess(x0) + mu * I
        mu is a number larger than absolute value of
        minimum eigenvalue of hess(x0)
        '''
        dimX = np.size(x0)
        hess0 = hess(x0)
        eigHess0 = np.linalg.eig(hess0)[0]
        if np.any(eigHess0 <= eps1):
            hess0 = hess0 + (abs(min(eigHess0)) + 1) * np.identity(dimX)

        iterNum, d0, stopCon = 0, np.linalg.solve(hess0, -grad0), True
        while stopCon:

            # create new iteration point
            if search is True:
                if exactsearch is False:
                    alpha = inexactLineSearch(x0, d0, func, grad,
                                              criterion='Simple')
                else:
                    alpha = goldenSection(x0, d0, func)
                x1 = x0 + alpha * d0
            else:
                x1 = x0 + d0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0, x1, f0, f1, grad1, eps1, eps2)

            # if hess is not positive definite
            # replace it by a positive definite matrix
            hess1 = hess(x1)
            eigHess1 = np.linalg.eig(hess1)[0]
            if np.any(eigHess1 <= eps1):
                hess1 = hess1 + (abs(min(eigHess1)) + 1) * np.identity(dimX)

            # update x0, d0 for computing next iteration point
            # update f0, grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, np.linalg.solve(hess1, -grad1)

            iterNum += 1

    if disp is True:
        if search is True:
            if exactsearch is True:
                print('\nMethod: Newton-Raphson (exact line-search)')
            else:
                print('\nMethod: Newton-Raphson (unexact line-search)')
        else:
            print('\nMethod: Newton-Raphson (no line-search)')
        print('Remark: we use modified Newton-Raphson'
        	  + ', when Hessian is not positive definite，modify it\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(x0))
        print('    Function value:    {0}'.format(f0))
        print('    Gradient:          {0}'.format(grad0))
        print('    Iteration number:  {0}'.format(iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0


@prinTime
def trustNewton(x0, func, grad, hess, mu=1, eps1=1e-4,
                eps2=1e-5, disp=False):

    if disp is True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        # if H0=(hess0 + mu*I) is not positive definite, replace mu by 4mu
        hess0 = hess(x0)
        idenMatrix = np.identity(np.size(x0))
        H0 = hess0 + mu * idenMatrix
        eigH0 = np.linalg.eig(H0)[0]
        if np.any(eigH0 <= eps1):
            mu = 4 * mu
            H0 = hess0 + mu * idenMatrix

        iterNum, d0, stopCon = 0, np.linalg.solve(H0, -grad0), True
        while stopCon:

            x1 = x0 + d0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0, x1, f0, f1, grad1, eps1, eps2)

            # update mu equaling to update trust region
            # if mu is larger, then size of trust region is smaller
            detaq = np.dot(grad0, d0) + 0.5 * np.dot(np.dot(d0, hess0), d0)
            r0 = (f1 - f0) / detaq
            if r0 < 0.25:
                mu = 4 * mu
            elif r0 > 0.75:
                mu = 0.5 * mu

            # if H1 is not positive definite
            # replace it by a positive definite matrix
            hess1 = hess(x1)
            H1 = hess1 + mu * np.identity(np.size(x1))
            eigH1 = np.linalg.eig(H1)[0]
            if np.any(eigH1 <= eps1):
                mu = 4 * mu
                H1 = hess1 + mu * np.identity(np.size(x1))

            # update x0, d0 for computing next iteration point
            # update f0, grad0 for computing next stop condition
            # update d0, grad0, hess0 for compute next mu
            x0, f0, grad0, hess0 = x1, f1, grad1, hess1
            d0 = np.linalg.solve(H1, -grad1)

            iterNum += 1

    if disp is True:
        print('\nMethod: Trust-region Newton Method')
        print('Remark: use l2-norm, namely, Levenberg-Marquardt algorithm\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(x0))
        print('    Function value:    {0}'.format(f0))
        print('    Gradient:          {0}'.format(grad0))
        print('    Iteration number:  {0}'.format(iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0


@prinTime
def quasiNewton(x0, func, grad, eps1=1e-4, eps2=1e-5, search=True,
                exactsearch=False, disp=False):

    if disp is True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        dimX = np.size(x0)
        idenMat = np.identity(dimX)
        hessI0 = idenMat

        iterNum, d0, stopCon = 0, -grad0, True
        while stopCon:

            # create new iteration point
            if search is True:
                if exactsearch is False:
                    alpha = inexactLineSearch(x0, d0, func, grad,
                                              criterion='Simple')
                else:
                    alpha = goldenSection(x0, d0, func)
                s0 = alpha * d0
                x1 = x0 + s0
            else:
                s0 = d0
                x1 = x0 + s0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0, x1, f0, f1, grad1, eps1, eps2)

            # update approximate inverse of Hessian
            y0 = grad1 - grad0
            s0M = s0.reshape(dimX, 1)
            y0M = y0.reshape(dimX, 1)
            s0y0T = np.dot(s0M, y0M.T)
            y0s0T = np.dot(y0M, s0M.T)
            s0Ty0 = np.dot(s0, y0)
            matA = idenMat - s0y0T / s0Ty0
            matB = idenMat - y0s0T / s0Ty0
            matC = np.dot(s0M, s0M.T) / s0Ty0
            hessI1 = np.dot(np.dot(matA, hessI0), matB) + matC

            # update x0, d0 for computing next iteration point
            # update f0, grad0 for computing next stop condition
            x0, f0, grad0, hessI0 = x1, f1, grad1, hessI1
            d0 = np.dot(hessI1, -grad1)

            iterNum += 1

    if disp is True:
        if search is True:
            if exactsearch is True:
                print('\nMethod: Quasi Newton (exact line-search）')
            else:
                print('\nMethod: Quasi Newton (unexact line-search)')
        else:
            print('\nMethod: Quasi Newton (no line-search)')
        print('Remark: we use BFGS formula\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(x0))
        print('    Function value:    {0}'.format(f0))
        print('    Gradient:          {0}'.format(grad0))
        print('    Iteration number:  {0}'.format(iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0


@prinTime
def conjuGrad(x0, func, grad, eps1=1e-4, eps2=1e-5, method='DY',
              search=True, exactsearch=False, disp=False):

    if disp is True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        iterNum, d0, stopCon = 0, -grad0, True
        while stopCon:

            # create new iteration point
            if search is True:
                if exactsearch is False:
                    alpha = inexactLineSearch(x0, d0, func, grad,
                                              criterion='Wolfe Powell')
                else:
                    alpha = goldenSection(x0, d0, func)
                s0 = alpha * d0
                x1 = x0 + s0
            else:
                s0 = d0
                x1 = x0 + s0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0, x1, f0, f1, grad1, eps1, eps2)

            # update approximate inverse of Hessian
            y0 = grad1 - grad0
            if method == 'HS':
                # Hestenes-Stiefel (1952)
                beta1 = np.dot(grad1, y0) / np.dot(d0, y0)
            elif method == 'LS':
                # Liu-Storey (1964)
                beta1 = np.dot(grad1, y0) / np.dot(-d0, grad0)
            elif method == 'PR':
                # Polak-Ribiere (1964)
                beta1 = np.dot(grad1, y0) / np.dot(grad0, grad0)
            elif method == 'FR':
                # Fletcher-Reeves (1969)
                beta1 = np.dot(grad1, grad1) / np.dot(grad0, grad0)
            elif method == 'CD':
                # Fletcher (1988)
                beta1 = np.dot(grad1, grad1) / np.dot(-d0, grad0)
            elif method == 'DY':
                # Dai-Yuan (2000)
                beta1 = np.dot(grad1, grad1) / np.dot(d0, y0)
            else:
                beta1 = 0

            # update x0, d0 for computing next iteration point
            # update f0, grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, -grad1 + beta1 * d0

            iterNum += 1

    if disp is True:
        if search is True:
            if exactsearch is True:
                print('\nMethod: Conjugate Method (exact line-search)\n')
            else:
                print('\nMethod: Conjugate Method (unexact line-search)\n')
        else:
            print('\nMethod: Conjugate Method (no line-search)\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(x0))
        print('    Function value:    {0}'.format(f0))
        print('    Gradient:          {0}'.format(grad0))
        print('    Iteration number:  {0}'.format(iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0


@prinTime
def nelderMead(x0, func, k=2, rho=1, sigma=2, gamma=0.5,
               eps1=1e-4, eps2=1e-5, disp=True):

    if disp is True:
        start = x0

    dimX = np.size(x0)
    A = np.array([x0] * dimX) + k * np.identity(dimX)
    # set of vertexes of simplex
    simplexMat = np.append(x0, A).reshape(dimX+1, dimX)
    # function values of vertexes
    fMat = list(map(func, simplexMat))

    fMax, fMin, iterNum = 1, 0, 0
    while abs(fMax-fMin) >= eps2:

        fMax, fSecMax, fMin = max(fMat), sorted(fMat)[-2], min(fMat)
        maxIndex, minIndex = fMat.index(fMax), fMat.index(fMin)
        secMaxIndex = fMat.index(fSecMax)

        simplexMax = simplexMat[maxIndex]
        # centroid
        centroid = (simplexMat.cumsum(axis=0)[dimX] - simplexMax) / dimX
        # reflection
        vertexNew = centroid + rho * (centroid - simplexMax)
        fNew = func(vertexNew)

        if fNew < fSecMax and fNew >= fMin:
            simplexMat[maxIndex] = vertexNew
            fMat[maxIndex] = fNew
        elif fNew < fMin:
            # expansion
            vertexNew2 = centroid + sigma * (vertexNew - centroid)
            fNew2 = func(vertexNew2)
            if fNew2 < fNew:
                # expansion is successful
                simplexMat[maxIndex] = vertexNew2
                fMat[maxIndex] = fNew2
            else:
                # expansion is unsuccessful
                simplexMat[maxIndex] = vertexNew
                fMat[maxIndex] = fNew
        else:
            if fNew < fMax:
                # inside contraction
                vertexNew2 = centroid + gamma * (vertexNew - centroid)
            else:
                # outside contraction
                vertexNew2 = centroid + gamma * (simplexMax - centroid)
            fNew2 = func(vertexNew2)
            if fNew2 < fMax:
                # contraction is successful
                simplexMat[maxIndex] = vertexNew2
                fMat[maxIndex] = fNew2
            else:
                for i, value in enumerate(simplexMat):
                    # contraction is unsuccessful
                    simplexMat[i] = (value + simplexMat[minIndex]) / 2
                fMat = list(map(func, simplexMat))

        iterNum += 1

    minPoint = simplexMat[minIndex]
    if disp is True:
        print('\nMethod: Nelder-Mead\n')
        print('    Initial point:     {0}'.format(start))
        print('    Termination point: {0}'.format(minPoint))
        print('    Function value:    {0}'.format(fMin))
        print('    Iteration number:  {0}'.format(iterNum))
        return minPoint, fMin, iterNum
    else:
        return minPoint


def simulatedAnnealing(x0, func, k=2, rho=1, sigma=2, gamma=0.5,
               eps1=1e-4, eps2=1e-5, disp=True)


def func(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2


def grad(x):
    temp = 200 * (x[0] ** 2 - x[1])
    return np.array([2 * temp * x[0] + 2 * (x[0] - 1), - temp])


def hess(x):
    h12 = - 400 * x[0]
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, h12], [h12, 200]])


x0 = np.array([11, 10])

gradDescent(x0, func, grad, search=False, disp=True)
gradDescent(x0, func, grad, exactsearch=False, disp=True)
gradDescent(x0, func, grad, exactsearch=True, disp=True)

newton(x0, func, grad, hess, search=False, disp=True)
newton(x0, func, grad, hess, exactsearch=False, disp=True)
newton(x0, func, grad, hess, exactsearch=True, disp=True)

trustNewton(x0, func, grad, hess, disp=True)

quasiNewton(x0, func, grad, search=False, disp=True)
quasiNewton(x0, func, grad, exactsearch=False, disp=True)
quasiNewton(x0, func, grad, exactsearch=True, disp=True)

conjuGrad(x0, func, grad, search=False, disp=True)
conjuGrad(x0, func, grad, exactsearch=False, disp=True)
conjuGrad(x0, func, grad, exactsearch=True, disp=True)

nelderMead(x0, func)

# a = list(map(lambda x:x/max(x),data))
# b = np.array([])
# for i in a:
#     b = np.append(b, i)
# data = b.reshape(20, 20)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import numpy as np

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
            -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

bounds = [(-512, 512), (-512, 512)]

x = np.arange(-512, 513)
y = np.arange(-512, 513)
xgrid, ygrid = np.meshgrid(x, y)
xy = np.stack([xgrid, ygrid])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='tessrain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eggholder(x, y)')
plt.show()

results = dict()
results['shgo'] = optimize.shgo(eggholder, bounds)
results['shgo']
results['DA'] = optimize.dual_annealing(eggholder, bounds)
results['DA']
results['DE'] = optimize.differential_evolution(eggholder, bounds)
results['BH'] = optimize.basinhopping(eggholder, bounds)
results['shgo_sobol'] = optimize.shgo(eggholder, bounds, n=200, iters=5,
                                      sampling_method='sobol')

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(eggholder(xy), interpolation='bilinear', origin='lower',
               cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')

def plot_point(res, marker='o', color=None):
    ax.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10)

plot_point(results['BH'], color='y')  # basinhopping           - yellow
plot_point(results['DE'], color='c')  # differential_evolution - cyan
plot_point(results['DA'], color='w')  # dual_annealing.        - white

# SHGO produces multiple minima, plot them all (with a smaller marker size)
plot_point(results['shgo'], color='r', marker='+')
plot_point(results['shgo_sobol'], color='r', marker='x')
for i in range(results['shgo_sobol'].xl.shape[0]):
    ax.plot(512 + results['shgo_sobol'].xl[i, 0],
            512 + results['shgo_sobol'].xl[i, 1],
            'ro', ms=2)

ax.set_xlim([-4, 514*2])
ax.set_ylim([-4, 514*2])
plt.show()

x = np.arange(-5, 5, 1)
y = np.arange(-5, 5, 1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
