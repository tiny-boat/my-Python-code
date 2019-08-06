import numpy as np
from math import sqrt
from scipy.stats import chi2
# from scipy.optimize import Bounds, minimize, LinearConstraint


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
            if i < p+q:
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


def normaliz(x):
    return (x - x.mean()) / x.std()


def rejRate(c1, c2, d1, d2, sigma2=1, seriesLen=400,
             maxLag=5, alpha=0.05, testNum=500):

    a1, a2, b1, b2 = c1 + c2, - c1 * c2, - (d1 + d2), d1 * d2
    rejNum = 0
    for j in range(testNum):
        series = np.random.random_sample(2) * 10 - 10
        noises = np.random.randn(3) * sqrt(sigma2)
        # generate series of ARMA(2, 2)
        for i in range(seriesLen):
            series = np.append(series, a1 * series[-1] + a2 * series[-2]
                               + noises[-1] + b1 * noises[-2] + b2 * noises[-3])
            noises = np.append(noises[-2:], np.random.randn(1))
        series = series[-seriesLen:]
        # compute sample autocorrelations of series of ARMA(2, 2)
        autoCorr = np.zeros(maxLag + 1)
        for k in range(1, maxLag + 1):
            autoCorr[k] = np.dot(normaliz(series[0:seriesLen-k]), normaliz(series[k:])) / (seriesLen-k-1)
        # compute chisquare statistics and its p value
        chisq = seriesLen * np.dot(autoCorr, autoCorr)
        chisqPValue = 1 - chi2.cdf(chisq, maxLag)
        if chisqPValue < alpha:
            rejNum += 1

    rejRate = rejNum / testNum
    # print('c1 = %f, c2 = %f, d1 = %f, d2 = %f, sigam^2 = %f'
    #        % (c1, c2, d1, d2, sigma2))
    # print('a1 = %f, a2 = %f, b1 = %f, b2 = %f, sigam^2 = %f'
    #        % (a1, a2, b1, b2, sigma2))
    # print('    reject rate: %.2f %%\n' % (rejRate * 100))
    return rejRate


if __name__ == '__main__':

    # p, q, gamma = 0, 2, (12.4168, -4.7520, 5.2)
    # maPara = armaME(p, q, gamma)

    # p, q, gamma = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    # armaPara = armaME(p, q, gamma)

    # coeAR, coeMA, sigma2 = (0.0894, -0.6265), (-0.3334, 0.8158), 4.0119
    # armaGamma = armaAutocov(coeAR, coeMA, sigma2)

    c1, d1, d2 = 0.2, 0.5, 0.5
    record = []
    for c2 in range(-10, 10, 1):
        c2 = c2 / 10
        rejrate = rejRate(c1, c2, d1, d2)
        if rejrate < 0.9:
            record.append((c2, rejrate*100))
    print(record)
