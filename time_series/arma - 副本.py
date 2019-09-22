#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import filterwarnings
from math import sqrt
import numpy as np
from statsmodels.tsa import arima_process, stattools, ar_model, arima_model

'''
warnings 是 Python 的标准库，由于使用 statsmodels.tsa 中
函数进行极大似然估计时会弹出警告信息（通常来源于优化算法）
这里导入了该库中的一个函数进行设置以避免程序运行过程中显示警告信息

运行该程序文件前请确保安装好了 numpy 和 statsmodels
statsmodels.tsa 是用于时间序列分析的 Python 第三方库
其官方帮助文档及源代码见：https://www.statsmodels.org/stable/tsa.html
其源代码托管在 Github：https://github.com/statsmodels/statsmodels

statsmodels 的安装在联网情况下使用命令 pip install statsmodels 即可
'''

'''
函数定义下的注释 """……""" 是该函数的帮助文档，当把该 py 文件当作模块导入时
使用 help(armaME) 命令，显示的就是该注释部分
一个帮助文档应至少包括该函数基本功能的介绍、参数类型及含义、返回值类型及含义这三个部分
'''


def armaME(p, q, acovf, appendix=True):
    """
    get moment esitimation of parameters of ARMA(p,q)
    using its estimation of autocovariance function acovf(k)

    Parameters
    ----------
    p : int
        the order of AR
    q : int
        the order of MA
    acovf: tuple, list or 1-d numpy.ndarray
        esitimation of autocovariance of ARMA:
        acovf(0), acovf(1), …, acovf(p+q)
    appendix: bool
        if appendix=True, print informations of armaPara

    Returns
    -------
    armaPara: tuple
        esitimation of parameters of ARMA(p,q):
        a1, a2, …, ap, b1, b2, …, bq, sigma^2

    """

    if p != 0:
        '''
        get the coefficents of AR: (a1, …, ap)' = A^-1 * b
        b = (acovf(q+1), …, acovf(q+p+1))
        the (i,j) element of A is acovf(q+i-j)
        '''
        A = np.zeros([p, p])
        for i in range(p):
            for j in range(p):
                index = q + i - j
                if index < 0:
                    index = -index
                A[i, j] = acovf[index]
        b = np.array(acovf[q+1:q+p+1])
        coeAR = np.linalg.solve(A, b)

        if q != 0:
            '''
            get the autocovariance function of MA: acovfMA(k) = a'B(k)a
            a = (-1,a1,a2,…,ap)', the (i,j) element of B(k) is acovf(k-i+j)
            acovfMA(k) will be used for computing the coefficients of MA
            '''
            a = np.append(-1, coeAR)
            Bk = np.zeros([p+1, p+1])
            acovfMA = np.zeros(q+1)
            for k in range(q+1):
                for i in range(p+1):
                    for j in range(p+1):
                        index = k - i + j
                        if index < 0:
                            index = -index
                        Bk[i, j] = acovf[index]
                acovfMA[k] = np.dot(a, np.dot(Bk, a))
        else:
            # sigma2 of AR(p)
            sigma2 = acovf[0] - np.dot(b, coeAR)

    else:
        acovfMA = acovf

    if q != 0:
        '''
        get the coefficents of MA:
        (b1, …, bq)' = [(acovfMA(1), …, acovfMA(q))' - ABC] / sigma^2
        B = lim(k→∞) omigak * acovfk^-1 * omigak'
        sigma^2 = acovfMA(0) - C'BC
        '''
        k = (p + q) * 10
        omigak = np.zeros([q, k])
        for i in range(q):
            for j in range(k):
                index = 1 + i + j
                if index <= q:
                    omigak[i, j] = acovfMA[index]
        acovfk = np.zeros([k, k])
        for i in range(k):
            for j in range(k):
                index = j - i
                if abs(index) <= q:
                    if index < 0:
                        index = -index
                    acovfk[i, j] = acovfMA[index]

        B = np.dot(omigak, np.dot(np.linalg.inv(acovfk), omigak.T))

        C = np.append(1, np.zeros(q-1))
        # sigma2 of ARMA(p,q) or MA(q)
        sigma2 = acovfMA[0] - np.dot(C.T, np.dot(B, C))

        A = np.zeros([q, q])
        for i in range(q-1):
            A[i, i+1] = 1
        coeMA = (acovfMA[1:q+1] - np.dot(A, np.dot(B, C))) / sigma2

    if q == 0:
        armaPara = tuple(np.append(coeAR, sigma2))
    elif p == 0:
        armaPara = tuple(np.append(coeMA, sigma2))
    else:
        armaPara = tuple(np.append(np.append(coeAR, coeMA), sigma2))

    if appendix:
        print('\nParameters of ARMA(%d,%d)\n' % (p, q))
        for i, value in enumerate(armaPara):
            if i < p:
                print('    a%d = %.4f' % (i+1, value))
            else:
                if i < p + q:
                    print('    b%d = %.4f' % (i-p+1, value))
                else:
                    print('    sigma^2 = %.4f' % value)
        print('\nMethod: Moment Estimation\n')

    return armaPara


'''
_rejRate() 使用的白噪声检验统计量为 Ljung–Box Q 统计量
这一统计量是白噪声检验更常用的统计量，与书上的统计量略有不同
该函数使用了 arima_process.arma_generate_sample() 生成样本
使用了 stattools.acf() 计算样本自相关系数
'''


def _rejRate(a1, a2, b1, b2, sigma2=1, seriesLen=400,
             maxLag=6, alpha=0.05, testNum=500):

    a1, a2, b1, b2 = c1 + c2, - c1 * c2, - (d1 + d2), d1 * d2
    ar, ma = (1, -a1, -a2), (1, b1, b2)
    rejNum = 0
    for j in range(testNum):
        series = arima_process.arma_generate_sample(ar, ma, seriesLen)
        '''
        autoCorr_chisq 包含自相关系数, 延迟 1 至 maxLag 期
        Ljung–Box Q 统计量及其 P 值
        '''
        autoCorr_chisq = stattools.acf(series, nlags=maxLag,
                                       qstat=True, fft=False)
        chisqPValue = autoCorr_chisq[2][-1]
        if chisqPValue < alpha:
            rejNum += 1

    rejRate = rejNum / testNum
    return rejRate


if __name__ == '__main__':

    '''
    ### 第二章习题 1.5, 2.4
    已知自协方差函的前提下，使用上面定义的 armaME() 计算 ARMA(p, q) 的参数
    '''
    print('\n*********************\n    chapter2: 1.5\n*********************')
    p, q, acovf = 0, 2, (12.4168, -4.7520, 5.2)
    maPara = armaME(p, q, acovf)

    print('\n*********************\n    chapter2: 2.4\n*********************')
    p, q, acovf = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    armaPara = armaME(p, q, acovf)

    '''
    ### 第二章习题 2.5
    使用 statsmodels.tsa 库中的函数 arima_process.arma_acovf()
    计算 ARMA(p,q) 的总体自协方差函数
    '''
    ar, ma, sigma2 = (1, -0.0894, 0.6265), (1, -0.3334, 0.8158), 4.0119
    armaAcovf = arima_process.arma_acovf(ar, ma, nobs=11, sigma2=sigma2)

    print('\n*********************\n    chapter2: 2.5\n*********************')
    for i, value in enumerate(armaAcovf):
        if i > 4:
            print('    γ%d = %.4f' % (i, value))
    print('\n')

    '''
    ### 第四章习题 3.4
    固定 c1, d1 和 d2，使用自定义函数 _rejRate()
    计算不同 c2 下白噪声检验的拒绝率
    c2 从 -1 到 1 每隔 0.1 取一个值
    '''
    print('\n*********************\n    chapter4: 3.4\n*********************')
    print('\nCannot get all availabe (c1,c2,d1,d2)'
          + ' that make the rejection rate > 90%')
    print('since the soulutions of (c1,c2) is depend on values of (d1,d2)')
    print('and no significantly quantitative laws can be found')
    print('\nFollowing are two exaples:\n')

    # 例 1

    c1, d1, d2 = 0.3, 0.2, 0.4
    c2L, rejrateL = [], []   # 存储 c2 或拒绝率的列表
    for c2 in range(-9, 10):
        c2 = c2 / 10
        rejrate = round(_rejRate(c1, c2, d1, d2) * 100, 2)
        c2L.append(c2)
        rejrateL.append(rejrate)

    print('----c1 = 0.3, d1 = 0.2, d2 = 0.4----\n')
    print('     c2     rejection rates')
    for j in range(len(c2L)):
        c2, rejrate = c2L[j], rejrateL[j]
        if c2 >= 0:
            if int(rejrate) < 10:
                print('     {0}          {1} %'.format(c2, rejrate))
            elif int(rejrate) < 100:
                print('     {0}         {1} %'.format(c2, rejrate))
            else:
                print('     {0}        {1} %'.format(c2, rejrate))
        else:
            if int(rejrate) < 10:
                print('    {0}          {1} %'.format(c2, rejrate))
            elif int(rejrate) < 100:
                print('    {0}         {1} %'.format(c2, rejrate))
            else:
                print('    {0}        {1} %'.format(c2, rejrate))

    # 例 2

    c1, d1, d2 = 0.4, 0.5, -0.3
    c2L, rejrateL = [], []
    for c2 in range(-9, 10):
        c2 = c2 / 10
        rejrate = round(_rejRate(c1, c2, d1, d2) * 100, 2)
        c2L.append(c2)
        rejrateL.append(rejrate)

    print('\n----c1 = 0.4, d1 = 0.5, d2 = -0.3----\n')
    print('     c2     rejection rates')
    for j in range(len(c2L)):
        c2, rejrate = c2L[j], rejrateL[j]
        intRej = int(rejrate)
        if c2 >= 0:
            if intRej < 10:
                print('     {0}          {1} %'.format(c2, rejrate))
            elif intRej < 100:
                print('     {0}         {1} %'.format(c2, rejrate))
            else:
                print('     {0}        {1} %'.format(c2, rejrate))
        else:
            if intRej < 10:
                print('    {0}          {1} %'.format(c2, rejrate))
            elif intRej < 100:
                print('    {0}         {1} %'.format(c2, rejrate))
            else:
                print('    {0}        {1} %'.format(c2, rejrate))
    print('\n')

    '''
    ### 第六章习题 1.5
    使用 arima_process.arma_generate_sample() 产生 ARMA 样本
    使用 ar_model.AR() 从样本构建 AR 模型
    使用 ar_model.AR.select_order() 给 AR 模型定阶，定阶准则为 AIC 或 BIC
    使用 stattools.acovf() 计算序列样本自协方差函数
    利用样本自协方差函数，使用 armaME() 得到 AR 模型矩估计 (Yule-Walker 估计)
    以上函数，除了 armaME()，均来自于第三方库 statsmodel.tsa
    其使用方法请参考官方帮助文档，或在导入库后使用 help 命令查询
    '''
    print('\n*********************\n    chapter6: 1.5\n*********************')

    def dist(x):
        '''从均匀分布 U(-4,4) 中产生随机数'''
        return 8 * np.random.random_sample(x) - 4

    ar, ma = (1, 0.9, 1.4, 0.7, 0.6), (1, )
    series = arima_process.arma_generate_sample(ar, ma, 500, distrvs=dist)
    ARmodel = ar_model.AR(series)
    maxlag = 12

    print('\n----order selection using AIC----\n')
    print('upper bound of order: %d' % maxlag)
    ARorder_aic = ARmodel.select_order(maxlag, 'aic', trend='nc')
    print('order: %d' % ARorder_aic)
    armaAcovf = stattools.acovf(series, nlag=ARorder_aic, fft=False)
    armaYW = armaME(ARorder_aic, 0, armaAcovf)

    print('----order selection using BIC----\n')
    print('upper bound of order: %d' % maxlag)
    ARorder_bic = ARmodel.select_order(maxlag, 'bic', trend='nc')
    print('order: %d' % ARorder_bic)
    armaAcovf = stattools.acovf(series, nlag=ARorder_bic, fft=False)
    armaYW = armaME(ARorder_bic, 0, armaAcovf)

    '''
    ### 第六章习题 3.1
    '''
    print('\n*********************\n    chapter6: 3.1\n*********************')

    '''
    ## (1)
    使用自定义函数 armaME() 得到 AR 部分的参数估计
    使用逆相关函数法得到 MA 部分的参数估计
    '''
    ar, ma, sigma2 = (1, -1.16, 0.37, 0.19, -0.18), (1, 0.5, -0.4), 4
    p, q = len(ar) - 1, len(ma) - 1
    series = arima_process.arma_generate_sample(ar, ma, 100, sigma=2)

    seriesAcovf = stattools.acovf(series, nlag=p+q, fft=False)
    paraAR = armaME(p, q, seriesAcovf, appendix=False)[0:p]

    # 逆相关函数法
    seriesMA = series[0:-p]
    for j in range(p):
        if j == p - 1:
            seriesMA = seriesMA - paraAR[j] * series[j+1:]
        else:
            seriesMA = seriesMA - paraAR[j] * series[j+1:-p+j+1]
    tempARmodel = ar_model.AR(seriesMA)
    tempARorder = tempARmodel.select_order(12, 'aic', trend='nc')
    tempARAcovf = stattools.acovf(seriesMA, nlag=tempARorder, fft=False)
    ywCoe = np.append(-1, armaME(tempARorder, 0, tempARAcovf, appendix=False))
    inverseAcov = np.zeros(q+1)

    for k in range(q+1):
        inverseAcovk = 0
        for j in range(p):
            if j + k >= p:
                break
            else:
                inverseAcovk = inverseAcovk + ywCoe[j] * ywCoe[j + k]
        inverseAcov[k] = inverseAcovk / ywCoe[-1]

    paraMA = armaME(q, 0, inverseAcov, appendix=False)
    paraSigma2 = 1 / paraMA[-1]
    paraMA = list(map(lambda x: -x, paraMA[0:-1]))

    def formatIter(x):
        '''
        格式化可迭代对象中的元素，每个元素保留 4 位小数
        '''
        return [round(i, 4) for i in x]

    print('\n----3.1 (1)----\n')
    print('    Coefficents of AR: {0}\n'.format(formatIter(paraAR)))
    print('    Coefficents of MA: {0}\n'.format(formatIter(paraMA)))
    print('    Sigma^2: %.4f\n' % paraSigma2)
    print('    Method：Moment Estimation / Inverse Correlation Function Method')

    '''
    ## (2)
    使用 arima_model.ARMA.fit() 得到 ARMA 的极大似然估计
    该函数默认的优化算法为使用 BFGS 公式的有限存储拟牛顿法（LBFGS）
    其他可供选择的优化算法有牛顿法、共轭梯度法、单纯形法等
    使用该函数优化似然函数时可能产生警告信息，但这里将之忽视
    此外这里没有按习题要求将（1）中的结果作为优化算法的初始点
    这是因为这些初始点对应的模型可能不满足平稳性和可逆性条件，而这将导致函数运行报错
    这里的初始点的选取是由 arima_model.ARMA.fit() 调用库中一个私有函数 _fit_start_params
    自动完成的，详情参见帮助文档
    '''
    filterwarnings('ignore')   # ignore warnings
    while True:
        series = arima_process.arma_generate_sample(ar, ma, 100, sigma=2)
        armaModel = arima_model.ARMA(series, (4, 2))
        try:
            armaResult = armaModel.fit(method='mle', trend='nc',
                                       disp=0, maxiter=10000)
        except ValueError as e:
            continue
        else:
            paraAR_mle, paraMA_mle = armaResult.arparams, armaResult.maparams
            paraSigma2_mle = armaResult.sigma2
            break

    print('\n----3.1 (2)----\n')
    print('    Coefficents of AR: {0}\n'.format(formatIter(paraAR_mle)))
    print('    Coefficents of MA: {0}\n'.format(formatIter(paraMA_mle)))
    print('    Sigma^2: %.4f\n' % paraSigma2_mle)
    print('    Method：Maximum Likelihood Estimation')

    '''
    ## (3)
    使用 arima_process.ArmaProcess 类中的属性 isstationary 和 isinvertible
    判断 ARMA 模型是否满足平稳性条件和可逆性条件
    '''
    armaProcess = arima_process.ArmaProcess(np.append(1, -paraAR_mle),
                                            np.append(1, paraMA_mle))
    stationary = armaProcess.isstationary
    invertible = armaProcess.isinvertible

    print('\n----3.1 (3)----\n')
    print('    Is stationary: {0}'.format(stationary))
    print('    Is invertible: {0}'.format(invertible))

    '''
    ## (4)
    对该题第 (2) 问中代码做一百次重复计算
    将 100 次计算结果存储于变量 data 中
    然后计算每个参数估计的均值、方差和均方误差
    实验显示结果不是很稳定，这可能与数据量较小有关
    '''
    data = np.array([np.append(np.append(paraAR_mle, paraMA_mle), paraSigma2)])
    i = 1
    while i < 100:
        series = arima_process.arma_generate_sample(ar, ma, 100, sigma=2)
        armaModel = arima_model.ARMA(series, (4, 2))
        try:
            armaResult = armaModel.fit(method='mle', trend='nc',
                                       disp=0, maxiter=10000)
        except ValueError as e:
            continue
        else:
            paraAR_mle, paraMA_mle = armaResult.arparams, armaResult.maparams
            paraSigma2_mle = armaResult.sigma2
            data = np.r_[data, np.array([np.append(np.append(paraAR_mle,
                                         paraMA_mle), paraSigma2_mle)])]
            i += 1

    varArray = np.var(data, 0)
    meanArray = np.mean(data, 0)
    biasArray = meanArray - np.array([1.16, -0.37, -0.19, 0.18, 0.5, -0.4, 4])
    mseArray = varArray + biasArray ** 2

    print('\n----3.1 (4)----\n')
    print('    100 estimation of parameters')
    print('    which include coefficents of AR, coefficents of MA'
          + ' and sigma^2\n')
    print('    Following are Mean, Var and MSE of'
          + ' each parameters estimation:\n')
    print('    Mean:              {0}\n'.format(formatIter(meanArray)))
    print('    Variance:          {0}\n'.format(formatIter(varArray)))
    print('    Mean Square Error: {0}\n'.format(formatIter(mseArray)))
