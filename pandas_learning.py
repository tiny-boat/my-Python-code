#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### ——————————————————————————————————
### Chapter Five: 10 Minutes to Pandas
### ——————————————————————————————————

## -------------------
## 5.1 Object Creation
## -------------------

print("\n---------------------------------------------------------------\n")
print("                      5.1 Object Creation                      \n")
print("---------------------------------------------------------------\n")

'''
5.1.1 Creat a Series by passing a list of values, with a default integer index
'''
s = pd.Series([1,3,5,np.nan,6,8])

print("A Series (s):")
print(s)
print("\n")

'''
5.1.2 Creat a DataFrame by passing a NumPy array, with a datetime index and labeled columns
'''
dates = pd.date_range('20180101',periods=7)

print("A datetime index (dates) used by following DataFrame:")
print(dates)
print("\n")

# "sigma * np.random.randn(m,n) + mu" will return a 6*4 array
# with the elements randomlly sampled from N(mu,sigma^2).
df = pd.DataFrame(np.random.randn(7,4), index=dates, columns=list('ABCD'))

print("A DataFrame (df) by passing a NumPy array:")
print(df)
print("\n")

'''
5.1.3 Creat a DataFrame by passing a dict of objects
'''
items = [('A',1.),('B',pd.Timestamp('20130102')),('C',pd.Series(1,index=list(range(4)),dtype='float32')),('D',np.array([3] * 4,dtype='int32')),('E',pd.Categorical(["test","train","test","train"])),('F','foo')]
df2 = pd.DataFrame(dict(items))

print("A DataFrame (df2) by passing a dict of objects:")
print(df2)
print("\n")
print("the dtypes of columns of above DataFrame (df2.dtypes):")
print(df2.dtypes)
print("\n")

## -----------------
## 5.2 Viewing Data
## -----------------

print("\n------------------------------------------------------------\n")
print("                      5.2 Viewing Data                      \n")
print("------------------------------------------------------------\n")

'''
5.2.1 View the top rows, bottom rows, index, colums and NumPy data of the DataFrame
'''
print("5 top rows of df:")
print(df.head())
print("\n")

print("3 bottom rows of df:")
print(df.tail(3))
print("\n")

print("index of df:")
print(df.index)
print("\n")

print("columns of df:")
print(df.columns)
print("\n")

print("values of df:")
print(df.values)
print("\n")

'''
5.2.2 Show a quick statistic summary of DataFrame
'''
print("a quick statistic summary of df:")
print(df.describe)
print("\n")

'''
5.2.3 Transpose the DataFrme
'''
print("the transposition of df:")
print(df.T)
print("\n")

'''
5.2.4 Sort the DataFrme
'''
print("sort of df by axis 1, namely by columns (sort_index, not ascending):")
print(df.sort_index(axis=1, ascending=False))
print("\n")

print("sort of df by values (sort_values, by 'B'):")
print(df.sort_values(by='B'))
print("\n")

## --------------
## 5.3 Selection
## --------------

print("\n---------------------------------------------------------\n")
print("                      5.3 Selection                      \n")
print("---------------------------------------------------------\n")