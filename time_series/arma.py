#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from statsmodels.tsa import arima_process, stattools
'''
statsmodels.tsa is an Python third party library
using for time_series_analysis
'''


def armaME(p, q, gamma):
    """
    get moment esitimation of parameters of ARMA(p,q)
    using its estimation of autocovariance function gamma(k)

    Parameters
    ----------
    p : int
        the order of AR
    q : int
        the order of MA
    gamma: tuple or list
        esitimation of autocovariance of ARMA:
        gamma(0), gamma(1), …, gamma(p+q)

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

    if not (isinstance(p, int) and isinstance(q, int) and
            (isinstance(gamma, tuple) or isinstance(gamma, list))):
        print('\nError: p and/or q are not integers, '
              + 'or gamma is not tuple or list!\n')
    else:
        if len(gamma) <= p+q:
            print('\nError: length of gamma <= p+q !\n')
        else:
            if not all(map(lambda x: isinstance(x, int)
                       or isinstance(x, float), gamma)):
                print('\nError: one or more elements of '
                      + 'gamma are not numbers!\n')

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


def armaAutocov(coeAR, coeMA, sigma2, kmax=10):
    """
    get autocovariance function gamma(k) of ARMA(p,q)

    Parameters
    ----------
    coeAR : tuple or list
        the coefficents of AR, set coeAR = () if model is MA(q)
    coeMA : tuple or list
        the coefficents of MA, set coeMA = () if model is AR(p)
    sigma2: float or int
        variance of White Noise
    kmax: int, 10 is the default
        max time lag of autocovariance function

    Returns
    -------
    gamma: tuple
        autocovariance function of ARMA(p,q):
        gamma(0), gamma(1), …, gamma(kmax)

    Examples
    --------
    >>> from time_series.arma import armaAutocov
    >>> coeAR, coeMA, sigma2 = (-0.0211, -0.3953), (-0.1758, 0.4540), 5.3405
    >>> armaGamma = armaAutocov(coeAR, coeMA, sigma2)

    Autocovariance of ARMA(2,2) with
    Coefficents of AR: (-0.0211, -0.3953)
    Coefficents of MA：(-0.1758, 0.454)
    Variance of White Noise：5.3405

        γ0 = 5.6100
        γ1 = -1.0999
        γ2 = 0.2302
        γ3 = 0.4299
        γ4 = -0.1001
        γ5 = -0.1678
        γ6 = 0.0431
        γ7 = 0.0654
        γ8 = -0.0184
        γ9 = -0.0255
        γ10 = 0.0078

    """
    p, q = len(coeAR), len(coeMA)
    psiLen = (p + q) * 100 + 1
    psi = np.zeros(psiLen)
    gammaLen = kmax + 1
    gamma = np.zeros(gammaLen)

    psi[0] = 1
    for j in range(1, psiLen):
        if q != 0:
            if j <= q:
                bj = coeMA[j - 1]
            else:
                bj = 0
            psi[j] = bj
        if p != 0:
            for k in range(p):
                if j - k - 1 >= 0:
                    psijk = psi[j - k - 1]
                else:
                    psijk = 0
                psi[j] = psi[j] + coeAR[k] * psijk

    for k in range(gammaLen):
        psijjk = 0
        for j in range(psiLen):
            if j + k >= psiLen:
                break
            else:
                psijjk = psijjk + psi[j] * psi[j + k]
        gamma[k] = sigma2 * psijjk

    print('\nAutocovariance of ARMA(%d,%d) with' % (p, q))
    print('Coefficents of AR: {0} \nCoefficents of MA：{1} \n'
          .format(coeAR, coeMA)
          + 'Variance of White Noise：{0}\n'
          .format(sigma2))

    for i, value in enumerate(gamma):
        print('    γ%d = %.4f' % (i, value))
    print('\n')

    return gamma


def _rejRate(a1, a2, b1, b2, sigma2=1, seriesLen=400,
            maxLag=6, alpha=0.05, testNum=500):

    a1, a2, b1, b2 = c1 + c2, - c1 * c2, - (d1 + d2), d1 * d2
    ar, ma = (1, -a1, -a2), (1, b1, b2)
    rejNum = 0
    for j in range(testNum):
        series = arima_process.arma_generate_sample(ar, ma, seriesLen)
        # autoCorr_chisq include acovf, Ljung–Box Q statistics and p-value
        autoCorr_chisq = stattools.acf(series, nlags=maxLag, qstat=True, fft=False)
        chisqPValue = autoCorr_chisq[2][-1]
        if chisqPValue < alpha:
            rejNum += 1

    rejRate = rejNum / testNum
    return rejRate


if __name__ == '__main__':

    # 1.5
    print('\n***********\n    1.5\n***********')
    p, q, gamma = 0, 2, (12.4168, -4.7520, 5.2)
    maPara = armaME(p, q, gamma)

    # 2.4
    print('\n***********\n    2.4\n***********')
    p, q, gamma = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    armaPara = armaME(p, q, gamma)

    # 2.5
    print('\n***********\n    2.5\n***********')

    print('\n# method 1: use custom function armaAutocov()')
    coeAR, coeMA, sigma2 = (0.0894, -0.6265), (-0.3334, 0.8158), 4.0119
    armaGamma = armaAutocov(coeAR, coeMA, sigma2)

    print('# method 2: use function arima_process.arma_acovf()'
          + ' in statsmodels.tsa\n')
    ar, ma, sigma2 = (1, -0.0894, 0.6265), (1, -0.3334, 0.8158), 4.0119
    armaAcovf = arima_process.arma_acovf(ar, ma, nobs=11, sigma2=sigma2)
    for i, value in enumerate(armaAcovf):
        if i > 4:
            print('    γ%d = %.4f' % (i, value))
    print('\n')

    # 3.4
    print('\n***********\n    3.4\n***********')
    c1, d1, d2 = 0.3, 0.2, 0.4
    c2L, rejrateL = [], []
    for c2 in range(-9, 10):
        c2 = c2 / 10
        rejrate = round(_rejRate(c1, c2, d1, d2) * 100, 2)
        c2L.append(c2)
        rejrateL.append(rejrate)

    print('\nCannot get all availabe (c1,c2,d1,d2)'
          + ' that make the rejection rate > 90%')
    print('since the soulutions of (c1,c2) is depend on values of (d1,d2)')
    print('and no significantly quantitative laws can be found')
    print('\nFollowing are two exaples:\n')
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
