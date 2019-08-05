#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def prinTime(func):
    '''wraaper used for printing the running time of function'''
    from time import time
    def wrapper(*args,**kargs):
        startTime = time()
        funcResult = func(*args,**kargs)
        endTime = time()
        print("运行时间：%.4f s\n" % (endTime - startTime))
        return funcResult
    return wrapper

def goldenSection(x0,d0,func,start=0,end=2,epsilon=1e-4):
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

def inexactLineSearch(x0,d0,func,grad,start=0,end=1e10,rho=0.1,sigma=0.4,criterion='Wolfe Powell'):
    '''Inexact Line Search Method with four available criterion:
    1.Armijo Goldstein
    2.Wolfe Powell
    3.Strong Wolfe Powell
    4.Simple'''

    # reduce unnecessary caculations in loop
    f0, df0 = func(x0), np.dot(grad(x0),d0)
    rhoDf0 = rho * df0
    boundary3 = sigma * df0
    boundary4 = sigma * abs(df0)

    while True:
        alpha = (start + end) / 2
        boundary1 = f0 + rhoDf0 * alpha
        boundary2 = f0 + boundary3 * alpha
        fAlpha, dfAlpha = func(x0 + alpha * d0), np.dot(grad(x0 + alpha * d0), d0)

        # different criterions have same condition1 to avoid too large alpha
        condition1 = (fAlpha <= boundary1)
        # different criterions have different condition2 to avoid too small alpha
        if criterion == 'Armijo Goldstein':
            condition2 = (fAlpha >= boundary2)
        elif criterion == 'Wolfe Powell':
            condition2 = (dfAlpha >= boundary3)
        elif criterion == 'Strong Wolfe Powell':
            condition2 = (abs(dfAlpha) <= boundary4)
        else:
            condition2 = True

        # update start or end point or stop iteration
        if condition1 == False:
            end = alpha
        elif condition2 == False:
            start = alpha
        else:
            minPoint = alpha
            minValue = fAlpha
            break

    return minPoint

def stopCondition(x0,x1,f0,f1,grad1,eps1,eps2,gradneed=True):
    '''create stop condition controlled by eps1 and eps2
    using informations including x0,x1,f(x0),f(x1) and gradient(x1)'''

    normX0 = np.linalg.norm(x0)
    absF0 = abs(f0)
    if normX0 > eps2 and absF0 > eps2:
        diffF = abs(f1-f0)/absF0
        diffX = np.linalg.norm(x1-x0)/normX0
    else:
        diffF = abs(f1-f0)
        diffX = np.linalg.norm(x1-x0)
    if gradneed == True:
        return diffF > eps2 and diffX > eps2 and np.linalg.norm(grad1) > eps1
    else:
        return diffF > eps2 and diffX > eps2

@prinTime
def gradDescent(x0,func,grad,eps1=1e-4,eps2=1e-5,search=True,exactsearch=False,appendix=False):

    if appendix == True:   # save initial point
        start = x0

    f0 = func(x0)
    grad0 = grad(x0)

    if np.linalg.norm(grad0) > eps1:
        iterNum, d0, stopCon = 0, - grad0, True
        while stopCon:

            # create new iteration point
            if search == True:
                if exactsearch == False:
                    alpha = inexactLineSearch(x0,d0,func,grad,criterion='Simple')
                else:
                    alpha = goldenSection(x0,d0,func)
                x1 = x0 + alpha * d0
            else:
                x1 = x0 + 0.001 * d0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0,x1,f0,f1,grad1,eps1,eps2)

            # update x0,d0 for computing next iteration point
            # update f0,grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, - grad1

            iterNum += 1

    if appendix == True:
        if search == True:
            if exactsearch == True:
                print("方法：梯度下降法（精确线搜索）")
            else:
                print("方法：梯度下降法（非精确线搜索）")
        else:
            print("方法：梯度下降法（无线搜索）")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(x0, f0))
        print("停止点梯度：{0}; 迭代次数：{1}".format(grad0, iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0

@prinTime
def newton(x0,func,grad,hess,eps1=1e-4,eps2=1e-5,search=True,exactsearch=False,appendix=False):

    if appendix == True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        # if hess(x0) is not positive definite, replace hess(x0) by hess(x0) + mu * I
        # mu is a number larger than absolute value of minimum eigenvalue of hess(x0)
        dimX = np.size(x0)
        hess0 = hess(x0)
        eigHess0 = np.linalg.eig(hess0)[0]
        if np.any(eigHess0 <= eps1):
            hess0 = hess0 + (abs(min(eigHess0)) + 1) * np.identity(dimX)

        iterNum, d0, stopCon = 0, np.linalg.solve(hess0, -grad0), True
        while stopCon:

            # create new iteration point
            if search == True:
                if exactsearch == False:
                    alpha = inexactLineSearch(x0,d0,func,grad,criterion='Simple')
                else:
                    alpha = goldenSection(x0,d0,func)
                x1 = x0 + alpha * d0
            else:
                x1 = x0 + d0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0,x1,f0,f1,grad1,eps1,eps2)

            # if hess is not positive definite, replace it by a positive definite matrix
            hess1 = hess(x1)
            eigHess1 = np.linalg.eig(hess1)[0]
            if np.any(eigHess1 <= eps1):
                hess1 = hess1 + (abs(min(eigHess1)) + 1) * np.identity(dimX)

            # update x0,d0 for computing next iteration point
            # update f0,grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, np.linalg.solve(hess1, -grad1)

            iterNum += 1

    if appendix == True:
        if search == True:
            if exactsearch == True:
                print("方法：牛顿法（精确线搜索）")
            else:
                print("方法：牛顿法（非精确线搜索）")
        else:
            print("方法：牛顿法（无线搜索）")
        print("说明：对牛顿法作了修正，当 Hessian 矩阵不正定时，用正定矩阵替换之")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(x0, f0))
        print("停止点梯度：{0}; 迭代次数：{1}".format(grad0, iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0

@prinTime
def trustNewton(x0,func,grad,hess,mu=1,eps1=1e-4,eps2=1e-5,appendix=False):

    if appendix == True:   # save initial point
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
            stopCon = stopCondition(x0,x1,f0,f1,grad1,eps1,eps2)

            # update mu equaling to update trust region
            # if mu is larger, then size of trust region is smaller
            detaq = np.dot(grad0,d0) + 0.5 * np.dot(np.dot(d0,hess0),d0)
            r0 = (f1 - f0) / detaq
            if r0 < 0.25:
                mu = 4 * mu
            elif r0 > 0.75:
                mu = 0.5 * mu

            # if H1 is not positive definite, replace it by a positive definite matrix
            hess1 = hess(x1)
            H1 = hess1 + mu * np.identity(np.size(x1))
            eigH1 = np.linalg.eig(H1)[0]
            if np.any(eigH1 <= eps1):
                mu = 4 * mu
                H1 = hess1 + mu * np.identity(np.size(x1))

            # update x0,d0 for computing next iteration point
            # update f0,grad0 for computing next stop condition
            # update d0,grad0,hess0 for compute next mu
            x0, f0, grad0, hess0, d0 = x1, f1, grad1, hess1, np.linalg.solve(H1, -grad1)

            iterNum += 1

    if appendix == True:
        print("方法：信頼域方法")
        print("说明：采用 Levenberg-Marquardt 方法")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(x0, f0))
        print("停止点梯度：{0}; 迭代次数：{1}".format(grad0, iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0

@prinTime
def quasiNewton(x0,func,grad,eps1=1e-4,eps2=1e-5,search=True,exactsearch=False,appendix=False):

    if appendix == True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        dimX = np.size(x0)
        idenMat = np.identity(dimX)
        hessI0 = idenMat

        iterNum, d0, stopCon = 0, -grad0, True
        while stopCon:

            # create new iteration point
            if search == True:
                if exactsearch == False:
                    alpha = inexactLineSearch(x0,d0,func,grad,criterion='Simple')
                else:
                    alpha = goldenSection(x0,d0,func)
                s0 = alpha * d0
                x1 = x0 + s0
            else:
                s0 = d0
                x1 = x0 + s0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0,x1,f0,f1,grad1,eps1,eps2)

            # update approximate inverse of Hessian
            y0 = grad1 - grad0
            s0M = s0.reshape(dimX,1)
            y0M = y0.reshape(dimX,1)
            s0y0T = np.dot(s0M,y0M.T)
            y0s0T = np.dot(y0M,s0M.T)
            s0Ty0 = np.dot(s0,y0)
            matA = idenMat - s0y0T / s0Ty0
            matB = idenMat - y0s0T / s0Ty0
            matC = np.dot(s0M,s0M.T) / s0Ty0
            hessI1 = np.dot(np.dot(matA, hessI0), matB) + matC

            # update x0,d0 for computing next iteration point
            # update f0,grad0 for computing next stop condition
            x0, f0, grad0, hessI0, d0 = x1, f1, grad1, hessI1, np.dot(hessI1, -grad1)

            iterNum += 1

    if appendix == True:
        if search == True:
            if exactsearch == True:
                print("方法：拟牛顿法（精确线搜索）")
            else:
                print("方法：拟牛顿法（非精确线搜索）")
        else:
            print("方法：拟牛顿法（无线搜索）")
        print("说明：采用 BFGS 算法")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(x0, f0))
        print("停止点梯度：{0}; 迭代次数：{1}".format(grad0, iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0

@prinTime
def conjuGrad(x0,func,grad,eps1=1e-4,eps2=1e-5,method='DY',search=True,exactsearch=False,appendix=False):

    if appendix == True:   # save initial point
        start = x0

    f0, grad0 = func(x0), grad(x0)

    if np.linalg.norm(grad0) > eps1:

        iterNum, d0, stopCon = 0, -grad0, True
        while stopCon:

            # create new iteration point
            if search == True:
                if exactsearch == False:
                    alpha = inexactLineSearch(x0,d0,func,grad,criterion='Wolfe Powell')
                else:
                    alpha = goldenSection(x0,d0,func)
                s0 = alpha * d0
                x1 = x0 + s0
            else:
                s0 = d0
                x1 = x0 + s0

            # compute stop condition
            f1, grad1 = func(x1), grad(x1)
            stopCon = stopCondition(x0,x1,f0,f1,grad1,eps1,eps2)

            # update approximate inverse of Hessian
            y0 = grad1 - grad0
            if method == 'HS':
                beta1 = np.dot(grad1,y0) / np.dot(d0,y0)   # Hestenes-Stiefel (1952)
            elif method == 'LS':
                beta1 = np.dot(grad1,y0) / np.dot(-d0,grad0)   # Liu-Storey (1964)
            elif method == 'PR':
                beta1 = np.dot(grad1,y0) / np.dot(grad0,grad0)   # Polak-Ribiere (1964)
            elif method == 'FR':
                beta1 = np.dot(grad1,grad1) / np.dot(grad0,grad0)   # Fletcher-Reeves (1969)
            elif method == 'CD':
                beta1 = np.dot(grad1,grad1) / np.dot(-d0,grad0)   # Fletcher (1988)
            elif method == 'DY':
                beta1 = np.dot(grad1,grad1) / np.dot(d0,y0)   # Dai-Yuan (2000)
            else:
                beta1 = 0

            # update x0,d0 for computing next iteration point
            # update f0,grad0 for computing next stop condition
            x0, f0, grad0, d0 = x1, f1, grad1, -grad1 + beta1 * d0

            iterNum += 1

    if appendix == True:
        if search == True:
            if exactsearch == True:
                print("方法：共轭梯度法（精确线搜索）")
            else:
                print("方法：共轭梯度法（非精确线搜索）")
        else:
            print("方法：共轭梯度法（无线搜索）")
        print("说明：采用 Hestenes-Stiefel 公式")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(x0, f0))
        print("停止点梯度：{0}; 迭代次数：{1}".format(grad0, iterNum))
        return x0, f0, grad0, iterNum
    else:
        return x0

@prinTime
def nelderMead(x0,func,k=2,rho=1,sigma=2,gamma=0.5,eps1=1e-4,eps2=1e-5,appendix=True):

    if appendix == True:
        start = x0

    dimX = np.size(x0)
    A = np.array([x0] * dimX) + k * np.identity(dimX)
    simplexMat = np.append(x0,A).reshape(dimX+1,dimX)   # set of vertexes of simplex
    fMat = list(map(func,simplexMat))   # function values of vertexes

    fMax, fMin, iterNum = 1, 0, 0
    while abs(fMax-fMin) >= eps2:

        fMax, fSecMax, fMin = max(fMat), sorted(fMat)[-2], min(fMat)
        maxIndex, secMaxIndex, minIndex = fMat.index(fMax), fMat.index(fSecMax), fMat.index(fMin)

        simplexMax = simplexMat[maxIndex]
        centroid = (simplexMat.cumsum(axis=0)[dimX] - simplexMax) / dimX   # centroid
        vertexNew = centroid + rho * (centroid - simplexMax)   # reflection
        fNew = func(vertexNew)

        if fNew < fSecMax and fNew >= fMin:
            simplexMat[maxIndex] = vertexNew
            fMat[maxIndex] = fNew
        elif fNew < fMin:
            vertexNew2 = centroid + sigma * (vertexNew - centroid)   # expansion
            fNew2 = func(vertexNew2)
            if fNew2 < fNew:
                simplexMat[maxIndex] = vertexNew2   # expansion is successful
                fMat[maxIndex] = fNew2
            else:
                simplexMat[maxIndex] = vertexNew   # expansion is unsuccessful
                fMat[maxIndex] = fNew
        else:
            if fNew < fMax:
                vertexNew2 = centroid + gamma * (vertexNew - centroid)   # inside contraction
            else:
                vertexNew2 = centroid + gamma * (simplexMax - centroid)   # outside contraction
            fNew2 = func(vertexNew2)
            if fNew2 < fMax:
                simplexMat[maxIndex] = vertexNew2   # contraction is successful
                fMat[maxIndex] = fNew2
            else:
                for i,value in enumerate(simplexMat):
                    simplexMat[i] = (value + simplexMat[minIndex]) / 2   # contraction is unsuccessful
                fMat = list(map(func,simplexMat))

        iterNum += 1

    minPoint = simplexMat[minIndex]
    if appendix == True:
        print("方法：Nelder-Mead 单纯形法")
        print("初始点：{0}".format(start))
        print("停止点：{0}; 停止点函数值：{1}".format(minPoint, fMin))
        print("迭代次数：{0}".format(iterNum))
        return minPoint, fMin, iterNum
    else:
        return minPoint


def func(x):
    return 100 * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

def grad(x):
    temp = 200 * (x[0] ** 2 - x[1])
    return np.array([2 * temp * x[0] + 2 * (x[0] - 1), - temp])

def hess(x):
    h12 = - 400 * x[0]
    return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, h12],[h12, 200]])

x0 = np.array([1000,100])

# gradDescent(x0,func,grad,search=False,appendix=True)
# gradDescent(x0,func,grad,exactsearch=False,appendix=True)
# gradDescent(x0,func,grad,exactsearch=True,appendix=True)

# newton(x0,func,grad,hess,search=False,appendix=True)
# newton(x0,func,grad,hess,exactsearch=False,appendix=True)
# newton(x0,func,grad,hess,exactsearch=True,appendix=True)

# trustNewton(x0,func,grad,hess,appendix=True)

# quasiNewton(x0,func,grad,search=False,appendix=True)
# quasiNewton(x0,func,grad,exactsearch=False,appendix=True)
# quasiNewton(x0,func,grad,exactsearch=True,appendix=True)

# conjuGrad(x0,func,grad,search=False,appendix=True)
# conjuGrad(x0,func,grad,exactsearch=False,appendix=True)
# conjuGrad(x0,func,grad,exactsearch=True,appendix=True)

nelderMead(x0,func)

# a = list(map(lambda x:x/max(x),data))
# b = np.array([])
# for i in a:
#     b = np.append(b,i)
# data = b.reshape(20,20)