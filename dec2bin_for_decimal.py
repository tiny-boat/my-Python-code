#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 化十进制小数位二进制
def dec2bin_for_decimal(x,n):
    a = []
    i = 0
    actn = 0
    trun = x
    while i < n:
        xt2 = x * 2
        if xt2 == 0:
            break
        elif xt2 >=1:
            a = a + [1,]
            x = xt2 - 1
        else:
            a = a + [0,]
            x = xt2
        i += 1
    for i, value in enumerate(a):
        if value == 1:
            b = pow(2,-(i+1))
            actn = b + actn
        else:
            continue
    err = actn - trun
    return a, err

dec2bin_for_decimal(0.18,19)