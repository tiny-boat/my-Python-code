#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----命名规则说明-----
# 所有自定义函数的名称均以大写字母开头，并用下划线 _ 区分名称中的不同单词
# 所有自定义变量的名称均以大写字母开头，并用首字母大写的方式区分名称中的不同单词
# 为区分局部变量与全局变量，所有全局变量的名称都以 _g 结尾
# ---------------------

# 二分类 logistic 回归的对数似然损失函数为 ∑_i { log[1 + exp(θ·x_i)] - y_i * (θ·x_i) }
# 对数似然函数可表示为矩阵形式：1'log(1+exp(X*θ)) - Y'*X*θ
# 其中，Y = (y_1,……,y_n)'，X = (x_i,……,x_n)', 1 = (1,1,……,1)'
# θ = (θ_1,θ_2,……,θ_p,θ_p+1)'：p+1 个参数构成的向量
# x_i = (x_i1,x_i2,……,x_ip,1)'：由第 i 个样本的 p 个属性（自变量）与 1 构成的向量
# y_i：第 i 个样本的标签（因变量）
# θ·x_i = θ_1*x_i1 + … + θ_p*x_ip + θ_p+1：θ 与 x_i 的内积
# 对数似然损失函数的梯度向量第 j 个元素为：∑_i { x_ij * [y_i - exp(θ·x_i) / (1+exp(θ·x_i))] }
# 对数似然损失函数的梯度向量的转置（雅可比矩阵）可表示为矩阵形式：[1/(1+exp(-X*θ) - Y]' * X

import pandas as pd
import numpy as np
import math
import random
import time

def LR_log_likelihood(df,*theta):

	# 参数说明：df 为数据框，*theta 为 θ = (θ_1,θ_2,……,θ_p,θ_p+1)

	# 为随机梯度下降法设置
	if len(df.shape) == 1:
		df = pd.DataFrame(df).T

	# 数据框 (DataFrame) 行数与列数
	RowNum, ColNum = df.shape

	# RowNum*ColNum 矩阵 X，RowNum*1 矩阵 Y，ColNum*1 矩阵 θ，RowNum*1 矩阵 X*θ
	X = np.column_stack((df.iloc[:,0:ColNum-1],np.mat([1]*RowNum).T))
	Y = np.mat(df.iloc[:,-1]).T
	Theta = np.mat(theta).T
	XTimesTheta = X * Theta

	# RowNum*1 矩阵 log(1+exp(X*θ)), RowNum*1 矩阵 1/(1+exp(-X*θ))
	P1, LogExao = np.mat(np.zeros([RowNum,1])), np.mat(np.zeros([RowNum,1]))
	j = 0
	for element in XTimesTheta:
		if element > 709.7:
			LogExao[j,0] = XTimesTheta[j,0]
			P1[j,0] = 1
		else:
			ExpXttAndOnej = 1 + np.exp(XTimesTheta[j,0])
			LogExao[j,0] = np.log(ExpXttAndOnej)
			P1[j,0] = 1 - 1 / ExpXttAndOnej
		j += 1

	# 1*1 对数似然损失 1'log(1+exp(X*θ))-Y'*X*θ
	LogLikelihoodCost = np.sum(LogExao) - Y.T * XTimesTheta

	# 1*ColNum 对数似然损失梯度的转置（雅可比矩阵）[1/(1+exp(-X*θ)-Y]' * X
	LogLikelihoodCostGradient = (P1 - Y).T * X

	return LogLikelihoodCost, LogLikelihoodCostGradient


def Optimization(df,method = 'Gradient_Descent',*theta):

	StartTime = time.time()

	Rel_dis = lambda x,y: abs(x-y)/max(1,abs(y))
	Vec_norm = lambda x: math.sqrt(x*x.T)

	if method =='Gradient_Descent':
		LogLikelihoodCost0, Gradient = LR_log_likelihood(df,theta)
		LogLikelihoodCost1 = LogLikelihoodCost0
		RelativeDistance = 1
		GradientNorm = Vec_norm(Gradient)
		IterationNum = 0

		print(theta)
		print(LogLikelihoodCost1)
		print(GradientNorm)
		print(RelativeDistance)
		print("\n")

		#while RelativeDistance > 1e-5 and IterationNum < 501 and GradientNorm > 1e-5:
		while GradientNorm > 1e-4:

			LogLikelihoodCost0 = LogLikelihoodCost1  # 更新上一次迭代点函数值
			theta = theta - 0.01*Gradient # 更新当前迭代点
			theta = theta.tolist()[0]  # 将矩阵转换为列表
			LogLikelihoodCost1, Gradient = LR_log_likelihood(df,*theta)  # 更新当前迭代点函数值与梯度
			GradientNorm = Vec_norm(Gradient)  # 更新当前迭代点的梯度范数
			RelativeDistance = Rel_dis(LogLikelihoodCost1,LogLikelihoodCost0)  # 更新当前迭代点函数值与上一迭代点函数值的相对距离
			IterationNum += 1 # 更新迭代次数

			print(theta)
			print(LogLikelihoodCost1)
			print(GradientNorm)
			print(RelativeDistance)
			print("\n")

	elif method == 'Steepest_Descent':
		LogLikelihoodCost0, Gradient = LR_log_likelihood(df,theta)
		LogLikelihoodCost1 = LogLikelihoodCost0
		RelativeDistance = 1
		GradientNorm = Vec_norm(Gradient)
		IterationNum = 0

		print(theta)
		print(LogLikelihoodCost1)
		print(GradientNorm)
		print(RelativeDistance)
		print("\n")

		#while RelativeDistance > 1e-5 and IterationNum < 501 and GradientNorm > 1e-5:
		while GradientNorm > 1e-4:

			# 搜索步长 k 的函数
			def g(k):
				tkG = theta-k*Gradient
				tkG = tkG.tolist()[0]  # 将矩阵转换为列表
				return LR_log_likelihood(df,*tkG)[0]

			# 黄金分割法
			def goldenopt(a,b,error):
				r=(math.sqrt(5)-1)/2
				a1=b-r*(b-a)
				a2=a+r*(b-a)
				while abs(b-a)>error:
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
					x=(a+b)/2
				return x

			LogLikelihoodCost0 = LogLikelihoodCost1  # 更新上一次迭代点函数值
			theta = theta - goldenopt(0,1,1e-5)*Gradient # 更新当前迭代点
			theta = theta.tolist()[0]  # 将矩阵转换为列表
			LogLikelihoodCost1, Gradient = LR_log_likelihood(df,*theta)  # 更新当前迭代点函数值与梯度
			GradientNorm = Vec_norm(Gradient)  # 更新当前迭代点的梯度范数
			RelativeDistance = Rel_dis(LogLikelihoodCost1,LogLikelihoodCost0)  # 更新当前迭代点函数值与上一迭代点函数值的相对距离
			IterationNum += 1 # 更新迭代次数

			print(theta)
			print(LogLikelihoodCost1)
			print(GradientNorm)
			print(RelativeDistance)
			print("\n")

	elif method == 'Stochasitic_Gradient_Descent':
		for j in range(50):
			RowNum = df.shape[0]
			Index = list(range(0,RowNum))
			random.shuffle(Index)
			LogLikelihoodCost0, Gradient = LR_log_likelihood(df.iloc[Index[0],:],theta)
			LogLikelihoodCost1 = LogLikelihoodCost0
			RelativeDistance = 1
			GradientNorm = Vec_norm(Gradient)

			print(theta)
			print(LogLikelihoodCost1)
			print(GradientNorm)
			print(RelativeDistance)
			print("\n")

			#while RelativeDistance > 1e-5 and IterationNum < 501 and GradientNorm > 1e-5:
			#while GradientNorm > 1e-4:
			IterationNum = 0
			while IterationNum < RowNum-1:

				IterationNum += 1 # 更新迭代次数
				LogLikelihoodCost0 = LogLikelihoodCost1  # 更新上一次迭代点函数值
				theta = theta - (4/(1+j+IterationNum)+0.01)*Gradient # 更新当前迭代点
				theta = theta.tolist()[0]  # 将矩阵转换为列表
				LogLikelihoodCost1, Gradient = LR_log_likelihood(df.iloc[Index[IterationNum],:],*theta)  # 更新当前迭代点函数值与梯度
				GradientNorm = Vec_norm(Gradient)  # 更新当前迭代点的梯度范数
				RelativeDistance = Rel_dis(LogLikelihoodCost1,LogLikelihoodCost0)  # 更新当前迭代点函数值与上一迭代点函数值的相对距离

				print(theta)
				print(LogLikelihoodCost1)
				print(GradientNorm)
				print(RelativeDistance)
				print("\n")

	EndTime = time.time()
	ExecuteTime = EndTime - StartTime

	print("优化算法：%s\n" % (method))
	print("对数似然损失近似极小值点为：{0}\n".format(theta))
	print("对数似然损失为：{0[0]}，对数似然损失梯度为：{0[1]}\n".format(LR_log_likelihood(df,*theta)))
	print("迭代次数：%d\n" % (IterationNum*50))
	print("执行时间：%f\n" % (ExecuteTime))

	return theta,IterationNum,ExecuteTime

df_g = pd.read_csv('F:/Data/testSet.csv',encoding='GB2312')
theta_g = [-0.1,0.1,0,0]
result = Optimization(df_g,'Stochasitic_Gradient_Descent',*theta_g)
print(result)