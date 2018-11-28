#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----命名规则说明-----
# 所有自定义函数的名称均以大写字母开头，并用下划线 _ 区分名称中的不同单词
# 所有自定义变量的名称均以大写字母开头，并用首字母大写的方式区分名称中的不同单词
# 为区分局部变量与全局变量，所有全局变量的名称都以 _g 结尾
# ---------------------

# 二分类 logistic 回归的对数似然函数为 ∑_i { y_i * (θ·x_i) - log[1 + exp(θ·x_i)] }
# 对数似然函数可表示为矩阵形式：Y*X*θ-1'log(1+exp(X*θ))
# 其中，Y = (y_1,……,y_n)，X = (x_i,……,x_n)', 1' = (1,1,……,1)
# θ = (θ_1,θ_2,……,θ_p,θ_p+1)'：p+1 个参数构成的向量
# x_i = (x_i1,x_i2,……,x_ip,1)'：由第 i 个样本的 p 个属性（自变量）与 1 构成的向量
# y_i：第 i 个样本的标签（因变量）
# θ·x_i = θ_1*x_i1 + … + θ_p*x_ip + θ_p+1：θ 与 x_i 的内积

# 对数似然函数的梯度向量第 j 个元素为：∑_i { x_ij * [y_i - exp(θ·x_i) / (1+exp(θ·x_i))] }
# 对数似然函数的梯度向量的转置（雅可比矩阵）可表示为矩阵形式：[Y' - 1/(1+exp(-X*θ)]' * X
# 对数似然函数的海森矩阵第 (j,k) 个元素为：∑_i { -x_ij * x_ik * [exp(θ·x_i) / (1+exp(θ·x_i))^2] }
# 对数似然函数的海森矩阵可表示为矩阵形式：X'*H*X
# 其中 H 为由矩阵 1/(1+exp(-X*θ)) * [1/(1+exp(X*θ))]' 的对角元构成的对角矩阵

import pandas as pd
import numpy as np
import math
import random

def LR_Log_likelihood(df,isgradient = False,ishessian = False,*theta):

	# 参数说明：df 为数据框,*theta 为 θ = (θ_1,θ_2,……,θ_p,θ_p+1)

	# 为随机梯度下降法设置
	if len(df.shape) == 1:
		df = pd.DataFrame(df).T

	RowNum = df.shape[0]  # 数据框 (DataFrame) 行数
	ColNum = df.shape[1]  # 数据框 (DataFrame) 列数

	X = np.column_stack((df.iloc[:,0:ColNum-1],np.mat([1]*RowNum).T))  # RowNum*ColNum 矩阵 X
	Y = np.mat(df.iloc[:,-1])	# 1*RowNum 矩阵 Y
	Theta = np.mat(theta).T  # ColNum*1 矩阵 θ

	XTimesTheta = X * Theta # RowNum*1 矩阵 X*θ
	# RowNum*1 矩阵 1+exp(X*θ)

	ExpXttAndOne,LogExao = [],[]
	j = 0
	for element in XTimesTheta:
		if element > 709.7:
			ExpXttAndOne.append(np.inf)
			LogExao.append(XTimesTheta.T.tolist()[0][j])
		else:
			ExpXttAndOne.append(1 + np.exp(XTimesTheta.T.tolist()[0][j]))
			LogExao.append(np.log(ExpXttAndOne[j]))
		j += 1
	ExpXttAndOne = np.mat(ExpXttAndOne).T
	LogExao = np.mat(LogExao).T
	# RowNum*1 矩阵 1/(1+exp(X*θ))，该矩阵第 i 个元素也代表给定 x_i 后 y_i = 0 的条件概率的估计值
	P0 = 1 / ExpXttAndOne
	# RowNum*1 矩阵 1/(1+exp(-X*θ))，，该矩阵第 i 个元素也代表给定 x_i 后 y_i = 1 的条件概率的估计值
	P1 = 1 - P0

	# 1*1 对数似然损失 -Y'*X*θ+1'log(1+exp(X*θ))
	LogLikelihoodCost = np.sum(LogExao) - Y * XTimesTheta

	# 1*ColNum 对数似然损失梯度的转置（雅可比矩阵）-[Y' - 1/(1+exp(-X*θ)]' * X
	if isgradient == True and ishessian == False:
		LogLikelihoodCostGradient = (P1.T - Y) * X
		return LogLikelihoodCost, LogLikelihoodCostGradient
	# ColNum*ColNum 对数似然损失海森矩阵 -X'*H*X
	elif ishessian == True:
		LogLikelihoodCostGradient = (P1.T - Y) * X
		H = np.diag(np.array(np.multiply(P1,P0).T)[0])
		LogLikelihoodCostHessian = - X.T * H * X
		return LogLikelihoodCost, LogLikelihoodCostGradient, LogLikelihoodCostHessian
	else:
		return LogLikelihoodCost

def Optimization(df,method = 'Gradient_Descent',*theta0):

	if method == 'Gradient_Descent':
		Gradient = LR_Log_likelihood(df,True,False,theta0)[1]
		LogLikelihoodCost0 = LR_Log_likelihood(df,False,False,theta0)
		LogLikelihoodCost1 = 2*LogLikelihoodCost0

		print(theta0)
		print(Gradient * Gradient.T)
		print(LR_Log_likelihood(df,False,False,theta0))
		print("\n")

		theta = theta0
		i = 0
		while abs(LogLikelihoodCost1-LogLikelihoodCost0)/max(1,abs(LogLikelihoodCost0)) > 1e-8 and i < 501:
			if i != 0:
				LogLikelihoodCost0 = LogLikelihoodCost1

			def g(k):
				tkG = theta-k*Gradient
				tkG = tkG.tolist()[0]
				return LR_Log_likelihood(df,False,False,*tkG)

			def goldenopt(a,b,Theta_error):
				r=(math.sqrt(5)-1)/2
				a1=b-r*(b-a)
				a2=a+r*(b-a)
				while abs(b-a)>Theta_error:
					g1=g(a1)
					g2=g(a2)
					if g1>g2:
						a=a1
						g1=g2
						a1=a2
						a2=a+r*(b-a)
					else:
						b=a2
						a2=a1
						g2=g1
						a1=b-r*(b-a)
					x_opt=(a+b)/2
				return x_opt

			theta = theta - goldenopt(0,1,1e-5)*Gradient
			theta = theta.tolist()[0]
			LogLikelihoodCostAndGradient = LR_Log_likelihood(df,True,False,*theta)
			LogLikelihoodCost1 = LogLikelihoodCostAndGradient[0]
			Gradient = LogLikelihoodCostAndGradient[1]
			print(theta)
			print(Gradient * Gradient.T)
			print(LR_Log_likelihood(df,False,False,theta))
			print("\n")
			i += 1

		return theta

	elif method == 'Newton':

		Gradient = LR_Log_likelihood(df,True,True,theta0)[1]
		Hessian = LR_Log_likelihood(df,True,True,theta0)[2]
		LogLikelihoodCost0 = LR_Log_likelihood(df,True,True,theta0)[0]
		LogLikelihoodCost1 = 2*LogLikelihoodCost0

		theta = theta0
		i = 0
		while abs(LogLikelihoodCost1-LogLikelihoodCost0)/max(1,abs(LogLikelihoodCost0)) > 0.01 or i < 2000:
			if i != 0:
				LogLikelihoodCost0 = LogLikelihoodCost1

			theta = np.mat(theta).T - Hessian.I * Gradient.T
			theta = theta.T.tolist()[0]
			LogLikelihoodCostAndGradientHessian = LR_Log_likelihood(df,True,True,*theta)
			LogLikelihoodCost1 = LogLikelihoodCostAndGradientHessian[0]
			Gradient = LogLikelihoodCostAndGradientHessian[1]
			Hessian = LogLikelihoodCostAndGradientHessian[2]
			print(theta)
			print(Gradient * Gradient.T)
			print(LR_Log_likelihood(df,False,False,theta))
			print("\n")
			i += 1

		return theta

	elif method == 'Stochasitic_Gradient_Descent':

		df0 = df.iloc[int(random.uniform(0,df.shape[0])),:]
		Gradient_s = LR_Log_likelihood(df0,True,False,theta0)[1]
		LogLikelihoodCost0 = LR_Log_likelihood(df,False,False,theta0)
		LogLikelihoodCost1 = 2*LogLikelihoodCost0

		print(theta0)
		print(Gradient_s * Gradient_s.T)
		print(LogLikelihoodCost0)
		print("\n")

		theta = theta0
		i = 0
		while i < 1000:
			if i != 0:
				LogLikelihoodCost0 = LogLikelihoodCost1

			def g(k):
				tkG = theta-k*Gradient_s
				tkG = tkG.tolist()[0]
				return LR_Log_likelihood(df0,False,False,*tkG)

			def goldenopt(a,b,Theta_error):
				r=(math.sqrt(5)-1)/2
				a1=b-r*(b-a)
				a2=a+r*(b-a)
				while abs(b-a)>Theta_error:
					g1=g(a1)
					g2=g(a2)
					if g1>g2:
						a=a1
						g1=g2
						a1=a2
						a2=a+r*(b-a)
					else:
						b=a2
						a2=a1
						g2=g1
						a1=b-r*(b-a)
					x_opt=(a+b)/2
				return x_opt

			theta = theta - goldenopt(0,1,1e-5)*Gradient_s
			theta = theta.tolist()[0]
			LogLikelihoodCost1 = LR_Log_likelihood(df,False,False,*theta)
			Gradient_s = LR_Log_likelihood(df0,True,False,*theta)[1]
			print(theta)
			print(Gradient_s * Gradient_s.T)
			print(LogLikelihoodCost1)
			print("\n")
			i += 1
			df0 = df.iloc[int(random.uniform(0,df.shape[0])),:]

		return theta


df_g = pd.read_csv('F:/Data/testSet.csv',encoding='GB2312')
theta_g = list([-0.1,0.1,0,-0.5])
result = Optimization(df_g,'Gradient_Descent',*theta_g)
print(result)
