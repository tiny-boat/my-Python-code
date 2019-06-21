#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import time

def Two_norm(x):
	# x 为 n*1 矩阵
	return math.sqrt(x.T*x)

def S_alpha_beta(x,y,z1,z2):
	# x,y,z 为 n*1 矩阵
	xT = x.T
	yT = y.T
	s1 = (xT * x).item(0)                                  # 数 s1
	s2 = (yT * y).item(0)                                  # 数 s2
	s3 = (xT * y).item(0)                                  # 数 s3
	s = s1 * s2 - s3**2                                    # 数 s
	c1 = (xT * z1).item(0)                                  # 数 c1
	c2 = (yT * z2).item(0)                                  # 数 c2
	alpha = (c1*s2 - c2*s3)/s                              # 数 alpha
	beta = (s1*c2 - c1*s3)/s                               # 数 beta
	return s,alpha,beta

'''
def Subspace_method(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r0 = b - A * x                                            # n*1 余量矩阵 r0
	q1 = A * r0                                               # n*1 矩阵 q1
	p = b - r0                                                # n*1 矩阵 p

	s,alpha,beta = S_alpha_beta(p,q1,b,b)                       # 数 s,alpha,beta

	r1 = (1-alpha)*b + alpha*r0 - beta*q1                     # n*1 余量矩阵 r1
	r1_norm = Two_norm(r1)                                    # r1 的 2-范数
	iter_num = 0										      # 迭代次数

	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r1_norm >= epsilon and Two_norm(x1-x0) >= epsilon:

		if iter_num > 0:
			x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

		q2 = A * r1
		s,alpha,beta = S_alpha_beta(q1,q2,r0,r0)

		x1 = x0 + alpha*r0 + beta*r1
		t = r0
		r0 = r1
		r1 = t - alpha*q1 - beta*q2
		q1 = q2

		r1_norm = Two_norm(r1)                                   # 更新 r 的 2-范数
		iter_num += 1                                          # 更新迭代次数

		print("第 {0} 次迭代解为：{1}\n".format(iter_num,x1.T)) # 打印迭代解
		print("余量为：{0}\n".format(r1.T))					   # 打印余量
		print("余量范数为：%f\n" % (r1_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r1.T, r1_norm, iter_num
'''

def Subspace_method(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r = b - A * x                                             # n*1 余量矩阵 r
	r_norm = Two_norm(r)                                      # r 的 2-范数
	iter_num = 0										      # 迭代次数
	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r_norm >= epsilon and Two_norm(x1-x0) >= epsilon:

		if iter_num > 0:
			x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

		p = b - r                                              # n*1 矩阵 p
		pT = p.T                                               # 1*n 矩阵 p^T
		q = A * r                                              # n*1 矩阵 q
		qT = q.T                                               # 1*n 矩阵 q^T

		s1 = (pT * p).item(0)                                  # 数 s1
		s2 = (qT * q).item(0)                                  # 数 s2
		s3 = (pT * q).item(0)                                  # 数 s3
		s = s1 * s2 - s3**2                                    # 数 s
		c1 = (pT * b).item(0)                                  # 数 c1
		c2 = (qT * b).item(0)                                  # 数 c2

		alpha = (c1*s2 - c2*s3)/s                              # 数 alpha
		beta = (s1*c2 - c1*s3)/s                               # 数 beta

		x1 = alpha*x0 + beta*r                                 # 更新 n*1 当前解矩阵 x1
		r = (1-alpha)*b + alpha*r - beta*q                     # 更新 n*1 余量矩阵 r

		r_norm = Two_norm(r)                                   # 更新 r 的 2-范数
		iter_num += 1                                          # 更新迭代次数

		print("第 {0} 次迭代解为：{1}\n".format(iter_num,x1.T)) # 打印迭代解
		print("余量为：{0}\n".format(r.T))					   # 打印余量
		print("余量范数为：%f\n" % (r_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r.T, r_norm, iter_num


A = np.mat([[1,2,3],[2,7,4],[3,4,9]])
b = np.mat([[14],[28],[38]])
x = np.mat([[1000],[-400],[75]])
epsilon = 1e-4
Subspace_method(A,b,x,epsilon)