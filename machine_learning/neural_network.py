#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# -----------------------------------------
# set of activate function and its derivate
# -----------------------------------------

def relu(x):
    if x > 0:
        return x
    else:
        return 0.0

def sigmod(x):
    return 1 / (1 + exp(-x))

def linear(x):
    return x

relu = np.vectorize(relu)
sigmod = np.vectorize(sigmod)
linear = np.vectorize(linear)

def drelu(x):
    if x >= 0:
        return 1.0
    else:
        return 0.0

def dsigmod(x):
    y = sigmod(x)
    return y * (1 - y)

def dlinear(x):
    return 1

drelu = np.vectorize(drelu)
dsigmod = np.vectorize(dsigmod)
dlinear = np.vectorize(dlinear)

# ------------------------------------------------------
# set of loss function, cost function and its derivative
# ------------------------------------------------------

def loss(truey, predy):
    return np.linalg.norm(truey - predy)

def cost(trueY, predY):
    cost = 0
    for i in (trueY - predY).T:
        cost += np.linalg.norm(i)
    cost = cost / trueY.shape[1]
    return cost

def dloss(truey, predy):
    return predy - truey

# --------------------------
# data and model information
# --------------------------

# data
X = np.random.rand(10000, 500)
Label = np.random.rand(10000, 1)

# n: sample number  mx: variable number of x  my: variable number of y
n, mx, my = X.shape[0], X.shape[1], Label.shape[1]

# number of nodes of six hidden layers: 2, 3, 4, 5, 6, 7
node = np.array([mx, 2, 3, 4, 5, 6, 7, my])
nlayer = len(node) - 1

# activiate function
actifunc = ('', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear')
dactifunc = ('', 'drelu', 'drelu', 'drelu', 'drelu', \
             'drelu', 'drelu', 'dlinear')

# learning rate
alpha = 0.01

# stochasitc initialization
for i in range(1, nlayer + 1):
    stri = str(i)
    Wi = np.random.randn(node[i], node[i - 1])
    bi = np.random.randn(node[i], 1)
    exec('W' + stri + '= Wi')   # n[i] * n[i-1]
    exec('b' + stri + '= bi')   # n[i] * 1


# ------------------------------------
# optimiztion method: gradient descent
# ------------------------------------

A0, Y = X.T, Label.T
iterNum, costdiff, cost1, gradNorm = 0, 1, 0, 1
while costdiff >= 1e-6 and iterNum <= 10000 and gradNorm >= 1e-4:
    # forward propogation
    for i in range(1, nlayer + 1):
        stri = str(i)
        Ais1 = eval('A' + str(i - 1))   # n[i-1] * n
        Wi = eval('W' + stri)
        bi = eval('b' + stri)
        Zi = np.dot(Wi, Ais1) + bi
        Ai = eval(actifunc[i] + '(Zi)')
        exec('Z' + stri + '= Zi')   # n[i] * n
        exec('A' + stri + '= Ai')   # n[i] * n

    # compute cost descent for stop condition
    cost0 = cost1
    cost1 = eval('cost(Y, A' + str(nlayer) + ')')
    costdiff = abs(cost1 - cost0)

    # backward propogation
    exec('dA' + str(nlayer) + '= dloss(Y, A' + str(nlayer) + ')')
    for i in range(nlayer, 0, -1):
        stri, stris1 = str(i), str(i - 1)
        Wi = eval('W' + stri)
        Zi = eval('Z' + stri)
        Ais1 = eval('A' + stris1)
        dAi = eval('dA' + stri)   # n[i] * n
        dZi = dAi * eval(dactifunc[i] + '(Zi)')   # n[i] * n
        dWi = np.dot(dZi, Ais1.T) / n   # n[i] * n[i-1]
        dbi = np.sum(dZi, axis=1, keepdims=True) / n   # n[i] * 1
        dAis1 = np.dot(Wi.T, dZi)
        exec('dW' + stri + '= dWi')
        exec('db' + stri + '= dbi')
        exec('dA' + stris1 + '= dAis1')

    # compute gradNorm for stop condition
    gradNorm = 0
    for i in range(1, nlayer + 1):
        exec('gradNorm += np.linalg.norm(dW' + str(i) + ') +'
             + 'np.linalg.norm(db' + str(i) + ')')
    print('第 %d 次迭代\n损失：%.4f，梯度范数：%.4f\n' % (iterNum, cost1, gradNorm))

    # gradient descent
    for i in range(1, nlayer + 1):
        stri = str(i)
        exec('W' + stri + '-= alpha * dW' + stri)
        exec('b' + stri + '-= alpha * db' + stri)
    iterNum += 1

print('----参数----：\n')
for i in range(1, nlayer + 1):
    strWi, strbi = 'W' + str(i), 'b' + str(i)
    print(strWi + ' = {0}'.format(eval(strWi)))
    print(strbi + ' = {0}\n'.format(eval(strbi)))

# ----------------------------------------------
# optimiztion method: minibatch gradient descent
# ----------------------------------------------

iterNum, batchNum, nbatch, costdiff, cost1, gradNorm = 0, 0, 10, 1, 0, 1

while iterNum <= (n // nbatch) * 10 and gradNorm >= 1e-4:

    start = nbatch * batchNum
    end = start + nbatch
    if end >= n:
        A0 = X[start:].T
        Y = Label[start:].T
        batchNum = 0
    else:
        A0 = X[start:end].T
        Y = Label[start:end].T
        batchNum += 1

    # forward propogation
    for i in range(1, nlayer + 1):
        stri = str(i)
        Ais1 = eval('A' + str(i - 1))   # n[i-1] * n
        Wi = eval('W' + stri)
        bi = eval('b' + stri)
        Zi = np.dot(Wi, Ais1) + bi
        Ai = eval(actifunc[i] + '(Zi)')
        exec('Z' + stri + '= Zi')   # n[i] * n
        exec('A' + stri + '= Ai')   # n[i] * n

    cost1 = eval('cost(Y, A' + str(nlayer) + ')')

    # backward propogation
    exec('dA' + str(nlayer) + '= dloss(Y, A' + str(nlayer) + ')')
    for i in range(nlayer, 0, -1):
        stri, stris1 = str(i), str(i - 1)
        Wi = eval('W' + stri)
        Zi = eval('Z' + stri)
        Ais1 = eval('A' + stris1)
        dAi = eval('dA' + stri)   # n[i] * n
        dZi = dAi * eval(dactifunc[i] + '(Zi)')   # n[i] * n
        dWi = np.dot(dZi, Ais1.T) / nbatch   # n[i] * n[i-1]
        dbi = np.sum(dZi, axis=1, keepdims=True) / nbatch   # n[i] * 1
        dAis1 = np.dot(Wi.T, dZi)
        exec('dW' + stri + '= dWi')
        exec('db' + stri + '= dbi')
        exec('dA' + stris1 + '= dAis1')

    # compute gradNorm for stop condition
    gradNorm = 0
    for i in range(1, nlayer + 1):
        exec('gradNorm += np.linalg.norm(dW' + str(i) + ') +'
             + 'np.linalg.norm(db' + str(i) + ')')
    print('第 %d 次迭代\n第 %d 批样本损失：%.4f，梯度范数：%.4f\n' % (iterNum, batchNum+1, cost1, gradNorm))

    # gradient descent
    for i in range(1, nlayer + 1):
        stri = str(i)
        exec('W' + stri + '-= alpha * dW' + stri)
        exec('b' + stri + '-= alpha * db' + stri)
    iterNum += 1

print('----参数----：\n')
for i in range(1, nlayer + 1):
    strWi, strbi = 'W' + str(i), 'b' + str(i)
    print(strWi + ' = {0}'.format(eval(strWi)))
    print(strbi + ' = {0}\n'.format(eval(strbi)))

