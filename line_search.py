#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

@prinTime
def forwardBackward(func,x0=10,stepsize=1,multiplier=2,appendix=False):
    '''Forward-Backward Method for determining search inteval of exact line search'''
    if appendix == True:
        start0 = x0

    x1 = x0 + stepsize
    f0, f1 = func(x0), func(x1)
    iterNum = 0
    while True:
        if f1 > f0:
            if iterNum == 0:
                stepsize = - stepsize
                x1, x2 = x0 + stepsize, x1
            else:
                start, end = min(x1,x2), max(x1,x2)
                break
        else:
            stepsize = multiplier * stepsize
            x2, x0, x1 = x0, x1, x1 + stepsize
        f0, f1 = func(x0), func(x1)
        iterNum += 1

    if appendix == True:
        print("方法：进退法")
        print("初始点：%.2f" % (start0))
        print("最终区间：[%f, %f]; 迭代次数：%d" % (start,end,iterNum))

    return start, end, iterNum

@prinTime
def dichotomy(start,end,func,dfunc,epsilon=1e-4,appendix=False):
    '''Dichotomy Method for exact line search'''

    if appendix == True:
        start0, end0 = start, end   # save initial search interval

    iterNum = 0
    while True:
        # compute middle point and its derivation
        middle = (start + end) / 2
        dfMiddle = dfunc(middle)
        if abs(dfMiddle) < epsilon or end - start < epsilon:
            break
        # update start or end point
        elif dfMiddle > 0:
            end = middle
        else:
            start = middle
        iterNum += 1

    minPoint = middle
    minValue = func(minPoint)
    if appendix == True:
        print("方法：二分法")
        print("初始区间：[%.2f, %.2f]; 最终区间：[%f, %f]" % (start0,end0,start,end))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def goldenSection(start,end,func,epsilon=1e-4,appendix=False):
    '''Golden Section Method for exact line search'''

    if appendix == True:
        start0, end0 = start, end   # save initial search interval

    # find two insertion points using fixed ratio
    from math import sqrt
    ratio = sqrt(5) / 2 - 0.5
    intervalLen = end - start
    middleL = start + (1 - ratio) * intervalLen
    middleR = start + ratio * intervalLen

    iterNum = 0
    while intervalLen >= epsilon:
        # update start or end point and two insertion points
        if func(middleL) > func(middleR):
            start = middleL
            intervalLen = end - start
            middleL = middleR
            middleR = start + ratio * intervalLen
        else:
            end = middleR
            intervalLen = end - start
            middleR = middleL
            middleL = start + (1 - ratio) * intervalLen
        iterNum += 1

    minPoint = (start + end) / 2
    minValue = func(minPoint)
    if appendix == True:
        print("方法：黄金分割法")
        print("初始区间：[%.2f, %.2f]; 最终区间：[%f, %f]" % (start0,end0,start,end))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def fibonacci(start,end,func,n=100,epsilon=1e-4,appendix=False):
    '''Fibonacci Method for exact line search'''

    if appendix == True:
        start0, end0 = start, end   # save initial search interval

    # create list fibL for storing Fibonacci sequence
    fib0, fib1, fibL = 0, 1, []
    for i in range(n):
        fib0, fib1 = fib1, fib0 + fib1
        fibL.append(fib0)

    # find two insertion points using unfixed ratio
    denominator, numerator = fibL.pop(), fibL.pop()
    ratio = numerator / denominator
    intervalLen = end - start
    middleL = start + (1-ratio) * intervalLen
    middleR = start + ratio * intervalLen

    iterNum = 0
    while iterNum < n and intervalLen >= epsilon:
        # update start or end point and two insertion points
        if func(middleL) > func(middleR):
            start = middleL
            intervalLen = end - start
            middleL = middleR
            middleR = start + ratio * intervalLen
        else:
            end = middleR
            intervalLen = end - start
            middleR = middleL
            middleL = start + (1 - ratio) * intervalLen
        # update unfixed ratio
        denominator, numerator = numerator, fibL.pop()
        ratio = numerator / denominator
        if iterNum == n-1:
           ratio = ratio + epsilon
        iterNum += 1

    minPoint = (start + end) / 2
    minValue = func(minPoint)
    if appendix == True:
        print("方法：斐波那契法")
        print("初始区间：[%.2f, %.2f]; 最终区间：[%f, %f]" % (start0,end0,start,end))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def newton(x0,func,dfunc,ddfunc,epsilon=1e-4,appendix=False):
    '''Newton Method for exact line search'''

    if appendix == True:
        initial = x0   # save initial point

    # make sure that conditions of loop are available
    # also make sure no loop if df(x0) = 0
    x1 = x0
    diffX = epsilon + 1

    iterNum = 0
    while diffX >= epsilon and abs(dfunc(x1)) >= epsilon:
        x1 = x0 - dfunc(x0)/ddfunc(x0)
        diffX = abs(x1 - x0)
        x0 = x1
        iterNum += 1

    minPoint = x0
    minValue = func(x0)
    if appendix == True:
        print("方法：一点二次插值法（牛顿法）")
        print("初始点：%.2f" % (initial))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def quadInterpo2(x0,x1,func,dfunc,epsilon=1e-4,appendix=False,method='secant'):
    '''Quadratic Interpolation Method with Two-Points for exact line search'''

    if appendix == True:
        initial1 = x0   # save initial point
        initial2 = x1

    if abs(dfunc(x0)) >= epsilon:
        # make sure that conditions of loop are available
        # also make sure no loop if df(x1) = 0
        x2 = x1
        distance = epsilon + 1

        iterNum = 0
        while distance >= epsilon and abs(dfunc(x2)) >= epsilon:
            if method == 'secant':
                x2 = x1 - dfunc(x1) * (x1 - x0) / (dfunc(x1) - dfunc(x0))
            else:
                diffX1X0 = x1 - x0
                x2 = x1 + 0.5 * diffX1X0 / ((func(x1) - func(x0)) / (dfunc(x1) * diffX1X0) - 1)
            distance = abs(x2 - x1)
            x0, x1 = x1, x2
            iterNum += 1
    else:
        x1 = x0   # make sure that the minPoint is x0 if df(x0) = 0

    minPoint = x1
    minValue = func(x1)
    if appendix == True:
        if method == 'secant':
            print("方法：二点二次插值法（割线法）")
        else:
            print("方法：二点二次插值法（非割线法）")
        print("初始点：%.2f, %.2f" % (initial1,initial2))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def quadInterpo3(start,middle,end,func,epsilon=1e-4,appendix=False):
    '''Quadratic Interpolation Method with Three-Points for exact line search'''

    if appendix == True:
        start0, middle0, end0 = start, middle, end   # save initial point

    iterNum = 0
    while True:
        # compute quadMinPoint (minimum point of quadratic interpolation)
        fStart, fMiddle, fEnd = func(start), func(middle), func(end)
        diffME, diffES, diffSM = middle - end, end - start, start - middle
        numerator = (fStart - fMiddle) * diffME * diffES
        denominator = diffME * fStart + diffES * fMiddle + diffSM * fEnd
        quadMinPoint = 0.5 * (start + middle + numerator / denominator)

        # update start, middle and end point
        # stop iteration if quadMinPoint approaches middle point
        if abs(quadMinPoint - middle) < epsilon:
            break
        elif func(quadMinPoint) > fMiddle:
            if quadMinPoint > middle:
                end = quadMinPoint
            else:
                start = quadMinPoint
        else:
            if quadMinPoint > middle:
                start = middle
                middle = quadMinPoint
            else:
                end = middle
                middle = quadMinPoint
        iterNum += 1

    minPoint = quadMinPoint
    minValue = func(quadMinPoint)
    if appendix == True:
        print("方法：三点二次插值法（抛物线法）")
        print("初始点：%.2f, %.2f, %.2f" % (start0,middle0,end0))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def cubInterpo2(start,end,func,dfunc,epsilon=1e-4,appendix=False):
    '''Cubic Interpolation Method with Two-Points for exact line search'''
    from math import sqrt

    if appendix == True:
        start0, end0 = start, end   # save initial point

    iterNum = 0
    while True:
        # compute cubMinPoint (minimum point of cubic interpolation)
        intervalLen = end - start
        fStart, fEnd = func(start), func(end)
        dfStart, dfEnd = dfunc(start), dfunc(end)
        dfSE = dfStart * dfEnd
        omega = 3 * (fEnd - fStart) / intervalLen - dfSE
        eta = sqrt(omega ** 2 - dfSE)
        cubMinPoint = start + (eta - dfStart - omega) * intervalLen / (2 * eta - dfStart + dfEnd)
        dfCubMinPoint = dfunc(cubMinPoint)

        # update start or end point
        # stop iteration if df(cubMinPoint) approaches 0
        if abs(dfCubMinPoint) < epsilon:
            break
        elif dfCubMinPoint > 0:
            end = cubMinPoint
        else:
            start = cubMinPoint
        iterNum += 1

    minPoint = cubMinPoint
    minValue = func(minPoint)
    if appendix == True:
        print("方法：二点三次插值法")
        print("初始点：%.2f, %.2f" % (start0,end0))
        print("极小值点：%.4f; 极小值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))

    return minPoint, minValue, iterNum


@prinTime
def inexactLineSearch(func,dfunc,start=0,end=1e10,rho=0.1,sigma=0.4,criterion='Wolfe Powell',appendix=False):
    '''Inexact Line Search Method with four available criterion:
    1.Armijo Goldstein
    2.Wolfe Powell
    3.Strong Wolfe Powell
    4.Simple'''

    if appendix == True:
        alpha0 = (start + end) / 2   # save initial point

    # reduce unnecessary caculations in loop
    f0, df0 = func(0), dfunc(0)
    rhoDf0 = rho * df0
    boundary3 = sigma * df0
    boundary4 = sigma * abs(df0)

    iterNum = 0
    while True:
        alpha = (start + end) / 2
        boundary1 = f0 + rhoDf0 * alpha
        boundary2 = f0 + boundary3 * alpha
        fAlpha, dfAlpha = func(alpha), dfunc(alpha)

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
        iterNum += 1

    if appendix == True:
        print("方法：非精确线搜索；准则：%s" % criterion)
        print("初始点：%.2f" % (alpha0))
        print("停止点：%.4f; 停止点函数值：%.4f; 迭代次数：%d" % (minPoint,minValue,iterNum))
        return minPoint, minValue, iterNum
    else:
        return minPoint


from math import sin,cos

def func(x):
    return x ** 2 - sin(x)

def dfunc(x):
    return 2 * x - cos(x)

def ddfunc(x):
    return 2 + sin(x)

forwardBackward(func,appendix=True)   # 进退法

dichotomy(0,1,func,dfunc,1e-5,True)   # 二分法
goldenSection(0,1,func,1e-5,True)   # 黄金分割法
fibonacci(0,1,func,100,1e-5,True)   # 斐波那契法

newton(0,func,dfunc,ddfunc,1e-5,True)   # 一点二次插值法（牛顿法）
quadInterpo2(0,1,func,dfunc,1e-5,True)   # 二点二次插值法（割线法）
quadInterpo2(0,1,func,dfunc,1e-5,True,'')   # 二点二次插值法
quadInterpo3(0,0.5,1,func,1e-5,True)   # 三点二次插值法（抛物线法）
cubInterpo2(0,1,func,dfunc,1e-5,True)   # 二点三次插值法

inexactLineSearch(func,dfunc,criterion='Armijo Goldstein',appendix=True) # 非精确线搜索