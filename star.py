#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1.65, 1.65, .01)
y = np.arange(-1.4, 1.9, .01)
x, y = np.meshgrid(x, y)

f = x ** 2 + (y - (x ** 2) ** (1/3)) ** 2 - 1
plt.figure()
plt.contour(x, y, f, 0, colors='red')
plt.title(r'$x^2 + \left(y-\sqrt[3]{x^2} \right)^2 = 1$', pad = 20)
plt.show()

