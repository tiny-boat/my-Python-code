import numpy as np
# from scipy.optimize import Bounds, minimize, LinearConstraint


def armaME(p, q, *gamma):
    """
    get moment esitimation of parameters of ARMA(p,q)
    using its estimation of autocovariance function gamma(k)

    Parameters
    ----------
    p : int
        the order of AR
    q : int
        the order of MA
    *gamma: tuple
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
    >>> armaPara = armaME(p, q, *gamma)

    Parameters of ARMA(2,2)

        a1 = -0.0211
        a2 = -0.3953
        b1 = -0.1758
        b2 = 0.4540
        sigma^2 = 5.3405

    Method: Moment Estimation

    """

    if not (isinstance(p, int) and isinstance(q, int)):
        print('\nError: p and/or q are not integers!\n')
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
        (b1, …, bq)' = [(gammaMA(1), …, gammaMA(q))' - ABC] / sigam^2
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


if __name__ == '__main__':

    p, q, gamma = 2, 2, (5.61, -1.1, 0.23, 0.43, -0.1)
    armaME(p, q, *gamma)

    p, q, gamma = 0, 2, (12.4168, -4.7520, 5.2)
    armaME(p, q, *gamma)
