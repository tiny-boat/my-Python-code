#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import time

def Two_norm(x):
	# x 为 n*1 矩阵
	return math.sqrt(x.T*x)

def Alpha_beta(x,y,z):
	# x,y,z 为 n*1 矩阵
	xT = x.T
	yT = y.T
	zT = z.T
	s1 = (xT * x).item(0)                                  # 数 s1
	s2 = (yT * y).item(0)                                  # 数 s2
	s3 = ((xT * y + yT * x)/2).item(0)                     # 数 s3
	s = s1 * s2 - s3**2                                    # 数 s
	c1 = ((zT * x + xT * z)/2).item(0)                     # 数 c1
	c2 = ((zT * y + yT * z)/2).item(0)                     # 数 c2
	alpha = (c1*s2 - c2*s3)/s                              # 数 alpha
	beta = (s1*c2 - c1*s3)/s                               # 数 beta
	return alpha,beta

def Subspace_method_1(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r = b - A * x                                             # n*1 余量矩阵 r
	r_norm = Two_norm(r)                                      # r 的 2-范数
	iter_num = 0										      # 迭代次数
	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r_norm >= epsilon and Two_norm(x1-x0) >= epsilon and iter_num < 1000:

		if iter_num > 0:
			x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

		p1 = A * x0                                            # n*1 矩阵 p1
		p2 = A * r
		alpha,beta = Alpha_beta(p1,p2,r)

		x1 = x0 + alpha*x0 + beta*r                                 # 更新 n*1 当前解矩阵 x1
		r = r - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

		r_norm = Two_norm(r)                                   # 更新 r 的 2-范数
		iter_num += 1                                          # 更新迭代次数

	print("\n方法一：u=x_k,v=r_k\n")
	print("第 {0} 次迭代解：{1}".format(iter_num,x1.T)) # 打印迭代解
	print("余量：{0}".format(r.T))					   # 打印余量
	print("余量范数：%f" % (r_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r.T, r_norm, iter_num

def Subspace_method_2(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r0 = b - A * x                                             # n*1 余量矩阵 r
	p1 = A * x                                            # n*1 矩阵 p1
	p2 = A * r0
	alpha,beta = Alpha_beta(p1,p2,r0)
	r1 = r0 - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

	r1_norm = Two_norm(r1)                                      # r 的 2-范数
	iter_num = 0										      # 迭代次数
	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r1_norm >= epsilon and Two_norm(x1-x0) >= epsilon and iter_num < 1000:

		if iter_num > 0:
			x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

		p1 = A * r1                                            # n*1 矩阵 p1
		p2 = A * r0
		alpha,beta = Alpha_beta(p1,p2,r0)

		x1 = x0 + alpha*r1 + beta*r0                                 # 更新 n*1 当前解矩阵 x1
		t = r0
		r0 = r1
		if iter_num == 0:
			r1 = t - alpha*p1 - beta*p2
		else:
			r1 = r0 - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

		r1_norm = Two_norm(r1)                                   # 更新 r 的 2-范数
		iter_num += 1                                          # 更新迭代次数

	print("方法二：u=r_k,v=r_k-1\n")
	print("第 {0} 次迭代解：{1}".format(iter_num,x1.T)) # 打印迭代解
	print("余量：{0}".format(r1.T))					   # 打印余量
	print("余量范数：%f" % (r1_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r1.T, r1_norm, iter_num

def Subspace_method_3(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r0 = b - A * x                                             # n*1 余量矩阵 r
	p1 = A * x                                            # n*1 矩阵 p1
	p2 = A * r0
	alpha,beta = Alpha_beta(p1,p2,r0)
	r1 = r0 - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

	r1_norm = Two_norm(r1)                                      # r 的 2-范数
	iter_num = 0										      # 迭代次数
	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r1_norm >= epsilon and Two_norm(x1-x0) >= epsilon and iter_num < 1000:

		if iter_num > 0:
			x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

		u = r1
		v = (r0 - ((r1.T*r0).item(0)/(r1.T*r1).item(0)) * r1)
		p1 = A * u                                            # n*1 矩阵 p1
		p2 = A * v
		alpha,beta = Alpha_beta(p1,p2,r0)

		x1 = x0 + alpha*u + beta*v                                 # 更新 n*1 当前解矩阵 x1
		t = r0
		r0 = r1
		if iter_num == 0:
			r1 = t - alpha*p1 - beta*p2
		else:
			r1 = r0 - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

		r1_norm = Two_norm(r1)                                   # 更新 r 的 2-范数
		iter_num += 1                                          # 更新迭代次数

	print("方法三：u=r_k,v=(r_k-1)-[(r_k^T*r_k-1)/norm^2(r_k)]*r_k\n")
	print("第 {0} 次迭代解：{1}".format(iter_num,x1.T)) # 打印迭代解
	print("余量：{0}".format(r1.T))					   # 打印余量
	print("余量范数：%f" % (r1_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r1.T, r1_norm, iter_num

'''
def Subspace_method_4(A,b,x,epsilon):

	# 求解 Ax^T = b^T
	# A 为 n*n 非奇异矩阵，x 为 n*1 矩阵，b 为 n*1 矩阵

	StartTime = time.time()

	r = b - A * x                                             # n*1 余量矩阵 r
	r_norm = Two_norm(r)                                      # r 的 2-范数
	iter_num = 0										      # 迭代次数
	x0 = x                                                    # 初始化上一步解矩阵
	x1 = 2*x                                                  # 初始化当前解矩阵

	while r_norm >= epsilon and Two_norm(x1-x0) >= epsilon and iter_num <= 1000:

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

	print("方法四：u=x_k,v=r_k\n")
	print("第 {0} 次迭代解为：{1}".format(iter_num,x1.T)) # 打印迭代解
	print("余量为：{0}".format(r.T))					   # 打印余量
	print("余量范数为：%f\n" % (r_norm))                    # 打印余量范数

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime
	print("执行时间：%f\n" % (ExecuteTime))                     # 打印执行时间

	return x1.T, r.T, r_norm, iter_num
'''

A = np.mat([[1,2,3],[2,7,4],[4,5,9]])
b = np.mat([[14],[28],[41]])
x = np.mat([[199],[200],[555]])
epsilon = 1e-4
print("\n迭代停止条件：||r_(k+1)|| < 1e-4 或 ||x_(k+1) - x_k|| < 1e-4 或 迭代次数 > 1000")
print("矩阵A：\n{0}".format(A))
print("向量b：{0}".format(b.T))
print("初值x：{0}".format(x.T))
print("精确解：{0}".format(np.mat([[1,2,3]])))
Subspace_method_1(A,b,x,epsilon)
Subspace_method_2(A,b,x,epsilon)
Subspace_method_3(A,b,x,epsilon)
#Subspace_method_4(A,b,x,epsilon)
'''
A = np.mat([[1,2,3],[2,7,4],[4,5,9]])
b = np.mat([[14],[28],[41]])
x = np.mat([[199],[0],[555]])
r0 = b - A * x                                             # n*1 余量矩阵 r
p1 = A * x                                            # n*1 矩阵 p1
p2 = A * r0
alpha,beta = Alpha_beta(p1,p2,r0)
r1 = r0 - alpha*p1 - beta*p2                     # 更新 n*1 余量矩阵 r

r1_norm = Two_norm(r1)                                      # r 的 2-范数
iter_num = 0										      # 迭代次数
x0 = x                                                    # 初始化上一步解矩阵
x1 = 2*x                                                  # 初始化当前解矩阵

while iter_num < 2:

	if iter_num > 0:
		x0 = x1			                                   # 更新 n*1 上一步解矩阵 x1

	print("第 {0} 次循环：".format(iter_num + 1))
	p1 = A * r1                                            # n*1 矩阵 p1
	print("p1：{0}".format(p1.T))
	p2 = A * r0
	print("p2：{0}".format(p2.T))
	alpha,beta = Alpha_beta(p1,p2,r0)
	print("alpha：{0}".format(alpha))
	print("beta：{0}".format(beta))

	x1 = x0 + alpha*r1 + beta*r0                                 # 更新 n*1 当前解矩阵 x1
	print("旧的x1：{0}".format(x0.T))
	print("新的x1：{0}".format(x1.T))
	t = r0
	print("旧的r0(t)：{0}".format(t.T))
	print("旧的r1：{0}".format(r1.T))
	r0 = r1
	print("新的r0：{0}".format(r0.T))
	if iter_num == 0:
		r1 = t - alpha*p1 - beta*p2
	else:
		r1 = r0 - alpha*p1 - beta*p2
	print("b-A*x0：{0}".format((b-A*x1).T))
	print("新的r1：{0}".format(r1.T))

	r1_norm = Two_norm(r1)                                   # 更新 r 的 2-范数
	iter_num += 1                                          # 更新迭代次数

#	print("第 {0} 次迭代解为：{1}\n".format(iter_num,x1.T)) # 打印迭代解
#	print("余量为：{0}\n".format(r1.T))					   # 打印余量
	print("r1范数为：%f\n" % (r1_norm))                    # 打印余量范数


<<<<<<< HEAD
A = np.mat([[1,2,3],[2,7,4],[3,4,9]])
b = np.mat([[14],[28],[38]])
x = np.mat([[1000],[-400],[75]])
epsilon = 1e-4
Subspace_method(A,b,x,epsilon)
=======
A = np.mat([[1,2,3],[2,7,4],[4,5,9]])
x = np.mat([[418.17159384],[-1419.5998351],[1753.10645053]])
b = np.mat([[14],[28],[41]])
r = b - A * x
r_n = math.sqrt(r.T * r)
print(r,r_n)
'''
>>>>>>> 2c515191b6eb0d9d3762ec8aa52e7a0bc3d400ef
