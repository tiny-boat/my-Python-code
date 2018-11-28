#黄金分割法确定单峰函数在区间的极小点
import numpy as np
import matplotlib.pyplot as plt
import math

#定义目标函数
def func(x):
    return pow(x,4)-14*pow(x,3)+60*pow(x,2)-70*x
#print(func(1)) #for test

#def func(x):
#    return pow(x,3)-2*x+1

#定义主函数 黄金分割法计算
def main(a0,b0,i):#区间和精度
    x_opt = []
    f_opt = []
    #初值区间[0,2] l为精度 注意：1-r为压缩比
    r = (math.sqrt(5)-1)/2
    k = 0
    a1 = a0+(1-r)*(b0-a0)
    b1 = a0+r*(b0-a0)
    #循环分割 if判断
    while abs(b0-a0) > i:
        k += 1
        fa = func(a1)
        fb = func(b1)
        if fa >= fb:
            a0 = a1
            a1 = b1
            fa = fb
            b1 = a0+r*(b0-a0)
        else:
            b0 = b1
            b1 = a1
            fb = fa
            a1 = a0+(1-r)*(b0-a0)
        x_opt.append((a0+b0)/2)
        f_opt.append(func((a0+b0)/2))
    return (x_opt,f_opt,k)
    #print(a0,b0)#for test



main_return = main(0,2,0.05)
print(main_return)
#main(0,3,0.15)



#x = np.linspace(0,2,100)
#y = [func(t) for t in x]
plt.plot(main_return[0],main_return[1],'r*')
plt.show()

x_opt = main_return[0][-1]
f_opt = main_return[1][-1]
k = main_return[2]
print("%.5f is the optimal point of the function and execute %d steps and the answer is %.5f"%(x_opt,k,f_opt))


