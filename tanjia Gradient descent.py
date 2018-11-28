import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return 100*(x[1]-x[0]**2)**2+(1-x[0])**2

def gradient(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]),200*(x[1]-x[0]**2)])


X1=np.arange(-1.5,1.5,0.05)
X2=np.arange(-3.5,2,0.05)

[x1,x2]=np.meshgrid(X1,X2)
'''
f=100*(x2-x1**2)**2+(1-x1)**2 # 给定的函数
plt.contour(x1,x2,f,20) # 画出函数的20条轮廓线
'''
#这是程序的第 1 个关键错误
#更改说明：你在前面已经定义了名为 f 的函数，那么此处不应该将这个变量命名为 f，否则你之前定义的函数 f 将被覆盖
myfun=100*(x2-x1**2)**2+(1-x1)**2 # 给定的函数
plt.contour(x1,x2,myfun,20) # 画出函数的20条轮廓线


x0=np.array([-1.2,1])
stepnum=0
W=np.zeros((2,2000))
W[:,0] = x0


while sum((gradient(x0))**2)>10**(-5):

    def g(x):
        return f(x0-x*gradient(x0))

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
        '''
            alpha0=g(x_opt)
        return (alpha0)
        '''
        #这是程序的第2个关键错误
        #更改说明：你后面要将 alpha0 作为最佳搜索步长使用，故而不应该让其等于目标函数值
        return (x_opt)

    stepnum+=1
    '''
    alpha0=goldenopt(0,0.05,10**(-5))
    '''
    #更改说明，搜索步长取值范围不合适，这里如使用上述范围，迭代次数将超过你设定的迭代上限2000次
    #这个很诡异，我尝试了几个范围，0-0.8 是其中最少的一个范围，迭代次数为 462 次
    alpha0=goldenopt(0,0.8,10**(-5))
    x0=x0-alpha0*gradient(x0)
    W[:,stepnum] = x0

'''
print('函数的极小值为：',x0)
'''
print('函数的极小值点为：',x0) #更改说明：x0 是极值点不是极值
print('迭代次数为：',stepnum)
W=W[:,0:stepnum]

plt.plot(W[0,:],W[1,:],'g*',W[0,:],W[1,:]) # 画出迭代点收敛的轨迹
#plt.plot(W[0,:],W[1,:],'g*') # 画出迭代点收敛的轨迹
plt.show()