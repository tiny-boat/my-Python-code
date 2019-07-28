#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np

def F(x):
	F = 0
	# int() is necessary, because 'float' object cannot be interpreted as an integer
	for i in range(int(np.size(x)/4)):
		x1,x2,x3,x4 = x[(0+4*i):(4+4*i)]
		F0 = (x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2 + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4
		F = F + F0
	return F

def GradientF(x):
	grad = np.array([])
	for i in range(int(np.size(x)/4)):
		x1,x2,x3,x4 = x[(0+4*i):(4+4*i)]
		grad0 = np.array([2*x1 + 20*x2 + 40*(x1 - x4)**3,20*x1 + 200*x2 + 4*(x2 - 2*x3)**3,10*x3 - 10*x4 - 8*(x2 - 2*x3)**3,-10*x3 + 10*x4 - 40*(x1 - x4)**3])
		grad = np.append(grad,grad0)
	return grad

def WolfeSearch(rho,sigma,alphamax,x0,d0,g0):
	for alpha in np.arange(alphamax,0,-0.001):
		x1 = x0 + alpha * d0
		Fx0 = F(x0)
		Fx1 = F(x1)
		Fx1sFx0 = Fx1 - Fx0
		if Fx1sFx0 < 0:
			g1 = GradientF(x1)
			d0g0 = np.dot(d0,g0)
			d0g1 = np.dot(d0,g1)
			if Fx1sFx0 <= rho * alpha * d0g0 and d0g1 >= sigma * d0g0:
				break
			else:
				continue
		else:
			continue
	if abs(alpha - 0.001) < 1e-4:
		d0 = - g0
		alpha = 1
		x1 = x0 + alpha * d0
		Fx0 = F(x0)
		Fx1 = F(x1)
		g1 = GradientF(x1)
	return alpha,x1,Fx0,Fx1,g1

def ThreeTermSubspaceCG(x0,M,e1,e2):
	g0 = GradientF(x0)
	if np.linalg.norm(g0,ord=2) >= e1:
		d0 = - g0
		alpha0,x1,F0,F1,g1 = WolfeSearch(rho,sigma,alphamax,x0,d0,g0)
		d1 = - g1
		if np.linalg.norm(g1,ord=2) >= e1 and abs(F1-F0)/max(1,abs(F0)) >= e2:
			alpha1,x2,F1,F2,g2 = WolfeSearch(rho,sigma,alphamax,x1,d1,g1)
			while np.linalg.norm(g2,ord=2) >= e1 and abs(F2-F1)/max(1,abs(F1)) >= e2:
				s0 = x1 - x0
				s1 = x2 - x1
				y1 = g2 - g1 #
				p1 = np.dot(s1,y1) #
				q1 = np.dot(s0,y1) #
				u1 = np.dot(s1,g2) #
				v1 = np.dot(s0,g2) #
				w1 = np.dot(g2,y1) #
				l1 = 2 * M * (1 + alpha1 ** 2) #
				m1 = 2 * M * alpha1 #
				a11 = l1 * p1 ** 2 #
				a12 = l1 * p1 * q1 #
				b11 = l1 * p1 * w1 - u1 - m1 * p1 * u1 #

				coef = np.corrcoef([g2,s1,s0])
				s0s1coef = coef[1][2]
				s0g2coef = coef[0][2]
				s1g2coef = coef[0][1]
				if abs(s0g2coef - 1) <= 1e-4 or abs(s0s1coef - 1) <= 1e-4:
					a1 = b11/a11
					b1 = 0
					d1 = -g2 + a1 * s1
				elif abs(s1g2coef - 1) <= 1e-4:
					a1 = 0
					b1 = b11/a12
					d1 = - g2 + b1 * s0
				else:
					a22 = l1 * q1 ** 2
					b21 = l1 * q1 * w1 - v1 - m1 * q1 * u1
					deta = a11 * a22 - a12 * a12
					a1 = (b11 * a22 - a12 * b21) / deta
					b1 = (a11 * b21 - a12 * b11) / deta
					d1 = - g2 + a1 * s1 + b1 * s0

				g1 = g2
				x0 = x1
				x1 = x2
				alpha1,x2,F1,F2,g2 = WolfeSearch(rho,sigma,alphamax,x1,d1,g1)
				print(F2)
			return x2,F2,g2
		else:
			return x1,F1,g1
	else:
		return x0,F0,g0

x0 = np.array([3,-1,0,1]*10)
g0 = GradientF(x0)
d0 = - g0
rho = 0.36
sigma = 0.5
alphamax = 1

print(ThreeTermSubspaceCG(x0,1000,1e-4,1e-4))