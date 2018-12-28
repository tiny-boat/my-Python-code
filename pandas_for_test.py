#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## -------------------
## 1 Creat DataFrame
## -------------------

# 通过 numpy 数组创建
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))
print(df)

# 通过字典创建
items = [('A',1.),('B',pd.Timestamp('20130102')),('C',pd.Series(1,index=list(range(4)),dtype='float32')),('D',np.array([3] * 4,dtype='int32')),('E',pd.Categorical(["test","train","test","train"])),('F','foo')]
df2 = pd.DataFrame(dict(items))
print(df2)

# 通过列表创建
data = DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.1]])

## ------------------------
## 2 Descriptive Statistics
## ------------------------

df.sum() # 求和(列)
df.sum(axis=1) # 求和(行)
df.describe() # 汇总统计
df.mean() # 均值
df.var() # 方差
df.std() # 标准差
df.cov() # 协方差
df.corr() # 相关系数
df.skew() # 偏度
df.kurt() # 峰度
df.cumsum() # 累积和
df.cumprod() # 累积积
df.cummax() # 累积最大
df.cummin() # 累积最小

## ---------
## 3 fillna
## ---------

data.fillna({0:0.5, 1:0, 2:-1}) # 第 1，2，3 列缺失值分别填补 0.5，0 和 -1