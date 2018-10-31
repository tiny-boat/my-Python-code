#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#name = input('请输入您的姓名')
#print('您好，',name,'先生')

# print(r'''line1\n
# line2\\
# lin3\t''') #r表示无需使用转义符，'''  '''用于换行

' a test module '

__author__ = 'tiny-boat'

#BMI指数计算器
height = input('\n这是一个BMI指数计算器\n\n请您在输入数据时不要加单位\n\n现在，请输入您的身高（单位：cm），然后按回车键：\n',)
weight = input('\n接下来，请输入您的体重（单位：kg），然后按回车键：\n',)
height = float(height)
weight = float(weight)
BMI = weight/pow(height*0.01,2)

if BMI < 18.5:
	TYPE='过轻'
elif 18.5 <= BMI < 24:
	TYPE='正常'
elif 23 <= BMI < 27:
	TYPE='过重'
elif 27 <= BMI <= 30:
	TYPE='肥胖'
else:
	TYPE='严重肥胖'

print("\n您的BMI指数 = %.2f，属 \"%s\" 类型" % (BMI,TYPE)) #切记不要加逗号！！

if __name__ == '__main__':
	print("loveyoumaster")