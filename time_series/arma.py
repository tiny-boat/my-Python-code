#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import filterwarnings
from math import sqrt
import numpy as np
from statsmodels.tsa import arima_process, stattools, ar_model, arima_model

'''
please make sure that you have installed numpy and statsmodels
if not, use 'pip install numpy' install numpy
use 'pip install statsmodels' install statsmodels
statsmodels.tsa is an Python third party library
using for time_series_analysis
reference: https://www.statsmodels.org/stable/tsa.html
'''


def armaME(p, q, gamma, appendix=True):
    """
    get moment esitimation of parameters of ARMA(p,q)
    using its estimation of autocovariance function gamma(k)

    Parameters
    ----------
    p : int
        the order of AR
    q : int
        the order of MA
    gamma: tuple, list or 1-d numpy.ndarray
        esitimation of autocovariance of ARMA:
        gamma(0), gamma(1), …, gamma(p+q)
    appendix: bool
        if appendix=True, print informations of armaPara

    Returns
    -------
    armaPara: tuple
        esitimation of parameters of ARMA(p,q):
        a1, a2, …, ap, b1, b2, …, bq, sigma^2

    Examples
    --------
    >>> from time_series.arma import armaME
    >>> p, q, gamma = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    >>> armaPara = armaME(p, q, gamma)

    Parameters of ARMA(2,2)

        a1 = -0.0211
        a2 = -0.3953
        b1 = -0.1758
        b2 = 0.4540
        sigma^2 = 5.3405

    Method: Moment Estimation

    """

    if p != 0:
        '''
        get the coefficents of AR: (a1, …, ap)' = A^-1 * b
        b = (gamma(q+1), …, gamma(q+p+1))
        the (i,j) element of A is gamma(q+i-j)
        '''
        A = np.zeros([p, p])
        for i in range(p):
            for j in range(p):
                index = q + i - j
                if index < 0:
                    index = -index
                A[i, j] = gamma[index]
        b = np.array(gamma[q+1:q+p+1])
        coeAR = np.linalg.solve(A, b)

        if q != 0:
            '''
            get the autocovariance function of MA: gammaMA(k) = a'B(k)a
            a = (-1,a1,a2,…,ap)', the (i,j) element of B(k) is gamma(k-i+j)
            gammaMA(k) will be used for computing the coefficients of MA
            '''
            a = np.append(-1, coeAR)
            Bk = np.zeros([p+1, p+1])
            gammaMA = np.zeros(q+1)
            for k in range(q+1):
                for i in range(p+1):
                    for j in range(p+1):
                        index = k - i + j
                        if index < 0:
                            index = -index
                        Bk[i, j] = gamma[index]
                gammaMA[k] = np.dot(a, np.dot(Bk, a))
        else:
            # sigma2 of AR(p)
            sigma2 = gamma[0] - np.dot(b, coeAR)

    else:
        gammaMA = gamma

    if q != 0:
        '''
        get the coefficents of MA:
        (b1, …, bq)' = [(gammaMA(1), …, gammaMA(q))' - ABC] / sigma^2
        B = lim(k→∞) omigak * gammak^-1 * omigak'
        sigma^2 = gammaMA(0) - C'BC
        '''
        k = (p + q) * 10
        omigak = np.zeros([q, k])
        for i in range(q):
            for j in range(k):
                index = 1 + i + j
                if index <= q:
                    omigak[i, j] = gammaMA[index]
        gammak = np.zeros([k, k])
        for i in range(k):
            for j in range(k):
                index = j - i
                if abs(index) <= q:
                    if index < 0:
                        index = -index
                    gammak[i, j] = gammaMA[index]

        B = np.dot(omigak, np.dot(np.linalg.inv(gammak), omigak.T))

        C = np.append(1, np.zeros(q-1))
        # sigma2 of ARMA(p,q) or MA(q)
        sigma2 = gammaMA[0] - np.dot(C.T, np.dot(B, C))

        A = np.zeros([q, q])
        for i in range(q-1):
            A[i, i+1] = 1
        coeMA = (gammaMA[1:q+1] - np.dot(A, np.dot(B, C))) / sigma2

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


def _rejRate(a1, a2, b1, b2, sigma2=1, seriesLen=400,
             maxLag=6, alpha=0.05, testNum=500):

    a1, a2, b1, b2 = c1 + c2, - c1 * c2, - (d1 + d2), d1 * d2
    ar, ma = (1, -a1, -a2), (1, b1, b2)
    rejNum = 0
    for j in range(testNum):
        series = arima_process.arma_generate_sample(ar, ma, seriesLen)
        # autoCorr_chisq include acovf, Ljung–Box Q statistics and p-value
        autoCorr_chisq = stattools.acf(series, nlags=maxLag,
                                       qstat=True, fft=False)
        chisqPValue = autoCorr_chisq[2][-1]
        if chisqPValue < alpha:
            rejNum += 1

    rejRate = rejNum / testNum
    return rejRate


if __name__ == '__main__':

    '''
    ### chapter 3: 1.5, 2.4
    use custom function armaME() defined above
    for computing parameters of ARMA process with acovf
    acovf: auto correlation variance function
    '''
    print('\n*********************\n    chapter2: 1.5\n*********************')
    p, q, acovf = 0, 2, (12.4168, -4.7520, 5.2)
    maPara = armaME(p, q, acovf)

    print('\n*********************\n    chapter2: 2.4\n*********************')
    p, q, acovf = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    armaPara = armaME(p, q, acovf)

    '''
    ### chapter 3: 2.5
    use function arima_process.arma_acovf() in statsmodels.tsa
    for computing theoretical autocovariance function of ARMA process
    '''
    ar, ma, sigma2 = (1, -0.0894, 0.6265), (1, -0.3334, 0.8158), 4.0119
    armaAcovf = arima_process.arma_acovf(ar, ma, nobs=11, sigma2=sigma2)

    print('\n*********************\n    chapter2: 2.5\n*********************')
    for i, value in enumerate(armaAcovf):
        if i > 4:
            print('    γ%d = %.4f' % (i, value))
    print('\n')

    '''
    ### chapter 4: 3.4
    use custom function _rejRate()
    for computing rejection rate of White Noise hypothesis test
    with different c2 for fixed c1, d1 and d2
    '''
    print('\n*********************\n    chapter4: 3.4\n*********************')
    print('\nCannot get all availabe (c1,c2,d1,d2)'
          + ' that make the rejection rate > 90%')
    print('since the soulutions of (c1,c2) is depend on values of (d1,d2)')
    print('and no significantly quantitative laws can be found')
    print('\nFollowing are two exaples:\n')

    '''
    example 1
    '''
    c1, d1, d2 = 0.3, 0.2, 0.4
    c2L, rejrateL = [], []   # list storing c2 or rejection rate
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

    '''
    example 2
    '''
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
    ### chapter 6: 1.5
    use arima_process.arma_generate_sample() for generating sample of ARMA
    use ar_model.AR() for create AR model from sample
    use ar_model.AR.select_order() for determing order of AR by AIC or BIC
    use stattools.acovf() for computing sample acovf of series
    use armaME() for getting moment (Yule-Walker) estimation with sample acovf
    these functions are all in statsmodel.tsa except custom function armaME()
    '''
    print('\n*********************\n    chapter6: 1.5\n*********************')

    def dist(x):
        '''generate number from U(-4,4)'''
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
    ### chapter6: 3.1
    '''
    print('\n*********************\n    chapter6: 3.1\n*********************')

    '''
    ## (1)
    compute parameters of AR using custom funciton armaME()
    compute parameters of MA using Inverse Correlation Function Method
    '''
    ar, ma, sigma2 = (1, -1.16, 0.37, 0.19, -0.18), (1, 0.5, -0.4), 4
    p, q = len(ar) - 1, len(ma) - 1
    series = arima_process.arma_generate_sample(ar, ma, 100, sigma=2)

    seriesAcovf = stattools.acovf(series, nlag=p+q, fft=False)
    paraAR = armaME(p, q, seriesAcovf, appendix=False)[0:p]

    # Inverse Correlation Function Method
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
        formatted output of element in iterable
        keep four decimal places
        '''
        return [round(i, 4) for i in x]

    print('\n----3.1 (1)----\n')
    print('    Coefficents of AR: {0}\n'.format(formatIter(paraAR)))
    print('    Coefficents of MA: {0}\n'.format(formatIter(paraMA)))
    print('    Sigma^2: %.4f\n' % paraSigma2)
    print('    Method：Moment Estimation / Inverse Correlation Function Method')

    '''
    ## (2)
    use arima_model.ARMA.fit() to compute MLE
    we may encounter warnings, i.e. non-positive definite Hessian
    ，but ignore it
    we select start_param of optimization algorithm by arima_model.ARMA.fit()
    instead of results of (1)
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
    # (3)
    use attributes isstationary and isinvertible
    of Class arima_process.ArmaProcess
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
    100 repetition of (2)
    100 results are stored in a variable data
    so that Mean, Var, MSE of results can be computed
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
