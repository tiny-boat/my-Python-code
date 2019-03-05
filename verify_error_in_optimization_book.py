#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
本程序验证了：孙文瑜 袁亚湘 著《最优化理论与方法》1.2.4 节（秩一校正）中
关于 ”P 的 2 范数 = y^T * x / (||x|| * ||y||)“ 的结论是错误的
这里 P = I - x * y^T / (||x|| * ||y||)，I 为单位矩阵

'''

import numpy as np
import math
import random

n = 2
I = np.mat(np.diag([1] * n))			  		# n * n 单位阵 I
x = np.mat(np.random.randint(0,10,size=(n,1)))  # n 维随机整数向量 x
y = np.mat(np.random.randint(0,10,size=(n,1)))  # n 维随机整数向量 y
xNorm = np.linalg.norm(x,ord = 2)				# x 的 2 范数
yNorm = np.linalg.norm(y,ord = 2)               # y 的 2 范数
fact = 1 / (xNorm * yNorm)
P = I - fact * (x * y.T)						# n * n 矩阵 P
PtP = P.T * P                                   # n * n 矩阵 P^T * P

fytx = float(fact * (y.T * x))
oSfytx = 1 - fytx                         	    # P 的不等于 1 的特征值
oSfytx2 = oSfytx ** 2                     	    # P^T * P 的行列式

PEig = np.linalg.eig(P)[0]                		# P 的特征值
PtPEig = np.linalg.eig(PtP)[0] 			  	    # P^T * P 的所有特征值
PtPDet = np.linalg.det(PtP) 			  		# P^T * P 的行列式
PNorm2 = np.linalg.norm(P,ord = 2) 		  		# P 的 2 范数 / 谱范数
PNorm1 = np.linalg.norm(P,ord = 1) 		  		# P 的 1 范数 / 列和范数
PNormInf = np.linalg.norm(P,ord = np.inf) 		# P 的无穷范数 / 行和范数
PNormFro = np.linalg.norm(P,ord = 'fro')  		# P 的 Frobenius 范数

print("\n----------------------------------------------------------------")
print("本程序验证了：")
print("孙文瑜 袁亚湘 著《最优化理论与方法》1.2.4 节（秩一校正）中")
print("关于 ”P 的 2 范数 = y^T * x / (||x|| * ||y||)“ 的结论是错误的")
print("这里 P = I - x * y^T / (||x|| * ||y||)，I 为单位矩阵")
print("----------------------------------------------------------------")

print("\n1.向量 x：{0}；向量 y：{1}".format(x.T.tolist()[0],y.T.tolist()[0]))
print("2.矩阵 P：\n{0}".format(P))
print("3.矩阵 P^T * P：\n{0}\n".format(PtP))

print("4.P 的特征值：{0}".format(PEig))
print("5.P 的不为 1 的特征值 = 1 - y^T * x / (||x|| * ||y||) = {0}\n".format(oSfytx))

print("6.P^T * P 的特征值：{0}".format(PtPEig))
print("7.P^T * P 的行列式：{0}".format(PtPDet))
print("8.P^T * P 的行列式 = (1 - y^T * x / (||x|| * ||y||) )^2 = {0}\n".format(oSfytx2))

print("9.P 的谱范数、列和范数、行和范数、Frobenius范数：\n{0}".format([PNorm2,PNorm1,PNormInf,PNormFro]))
print("10.P 的 2 范数（谱范数）≠ y^T * x / (||x|| * ||y||) = {0}\n\n".format(fytx))