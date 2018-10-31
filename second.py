# -*- coding: utf-8 -*-

import math

def quadratic(a, b, c):
    d = math.sqrt(pow(b,2)-4*a*c)
    e = 2*a
    x = (-b + d)/e
    y = (-b - d)/e
    return x, y

'''
print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))

if quadratic(2, 3, 1) != (-0.5, -1.0):
    print('测试失败')
elif quadratic(1, 3, -4) != (1.0, -4.0):
    print('测试失败')
else:
    print('测试成功')
'''

def product(*x):
    pro = 1
    for i in x:
        pro = pro * i
    if pro == 1:
    	raise TypeError('no arguments are inputed')
    else:
    	return pro

'''
print('product(5) =', product(5))
print('product(5, 6) =', product(5, 6))
print('product(5, 6, 7) =', product(5, 6, 7))
print('product(5, 6, 7, 9) =', product(5, 6, 7, 9))
if product(5) != 5:
    print('测试失败!')
elif product(5, 6) != 30:
    print('测试失败!')
elif product(5, 6, 7) != 210:
    print('测试失败!')
elif product(5, 6, 7, 9) != 1890:
    print('测试失败!')
else:
    try:
        product()
        print('测试失败!')
    except TypeError:
        print('测试成功!')
'''

def hanoi(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        hanoi(n - 1, a, c, b)
        print(a, '-->', c)
        hanoi(n - 1, b, a, c)

'''
# 调用
hanoi(3, 'A', 'B', 'C')
hanoi(4, 'A', 'B', 'C')
'''

def trim(s):
    i = 0
    ls = len(s)
    j = ls - 1
    if ls > 0:
        while s[i]==' ':
            i += 1
            if i >= ls:
                break
        while s[j]==' ' and i < ls:
            j -= 1
    return s[i:j+1]

'''
if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')
'''

def findMinAndMax(li):
    if len(li) != 0:
        minL,maxL = li[0],li[0]
        for x in li[1:]:
            if x < minL:
                minL = x
            elif x > maxL:
                maxL = x
        return minL,maxL
    else:
        return (None,None)

'''
if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')
'''

L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [s.lower() for s in L1 if isinstance(s,str)]

'''
print(L2)
if L2 == ['hello', 'world', 'apple']:
    print('测试通过!')
else:
    print('测试失败!')
'''

def triangles():
    L = [1]
    while True:
        yield L
        L = [1] + [L[k] + L[k + 1] for k in range(len(L) - 1)] + [1]
    return 'done'
'''
    L = []
    LenL = 1
    while LenL:
        if LenL > 2:
            # 2th to (LenL-1)th elements should be updated
            OldL = L[:]
            for IndexL in range(1,LenL-1):
                L[IndexL] = OldL[IndexL-1] + OldL[IndexL]
        L.append(1) # the last element is always 1
        yield L[:]
        LenL += 1
    return 'done'
'''
'''
n = 0
results = []
for t in triangles():
    print(t)
    results.append(t)
    n = n + 1
    if n == 10:
        break
if results == [
    [1],
    [1, 1],
    [1, 2, 1],
    [1, 3, 3, 1],
    [1, 4, 6, 4, 1],
    [1, 5, 10, 10, 5, 1],
    [1, 6, 15, 20, 15, 6, 1],
    [1, 7, 21, 35, 35, 21, 7, 1],
    [1, 8, 28, 56, 70, 56, 28, 8, 1],
    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
]:
    print('测试通过!')
else:
    print('测试失败!')
'''

def normalize(name):
    return name.capitalize()

'''
# 测试:
L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)
'''

from functools import reduce
def prod(L):
    return reduce(lambda x, y: x * y, L)

'''
print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
if prod([3, 5, 7, 9]) == 945:
    print('测试成功!')
else:
    print('测试失败!')
'''

from functools import reduce

def str2float(st):

    def SplitStr(char):
        float_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '.': '.'}
        return float_dict[char]

    # list formed from characters splited from string 由从字符串中分离的字符组成的列表
    list_splited_from_str = list(map(SplitStr, st))
    # index of decimal point in list 小数点在列表中的索引
    decpoint_index = list_splited_from_str.index('.')

    # sublist formed from integer part of float 由浮点数整数部分构成的子列表
    lis_int = list_splited_from_str[0:decpoint_index]
    # sublist formed from decimal part of float 由浮点数小数部分构成的子列表
    lis_dec = list_splited_from_str[-1:decpoint_index:-1]

    return reduce(lambda x,y: x*10 + y, lis_int) + reduce(lambda x,y: x/10 + y, lis_dec) / 10

'''
print('str2float(\'123.456\') =', str2float('123.456'))
if abs(str2float('123.456') - 123.456) < 0.00001:
    print('测试成功!')
else:
    print('测试失败!')
'''

def is_palindrome(num):
    str_num = str(num)
    reverse_num = int(str_num[-1::-1])
    return reverse_num == num

'''
output = filter(is_palindrome, range(1, 1000))
print('1~1000:', list(output))
if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]:
    print('测试成功!')
else:
    print('测试失败!')
'''

# 对列表按姓名正序排列
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]
def by_name(t):
    return t[0]
L1 = sorted(L, key=by_name)
print(L1)

# 对列表按分数倒序排列
def by_score(t):
    return t[1]
L2 = sorted(L, key=by_score, reverse = True)
print(L2)

# 利用闭包返回一个计数器函数，每次调用它返回递增整数
def createCounter():
    L = [0]
    def counter():
        L[0] += 1
        return L[0]
    return counter

'''
# 测试:
counterA = createCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA()) # 1 2 3 4 5
counterB = createCounter()
if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
    print('测试通过!')
else:
    print('测试失败!')

L = list(filter(lambda x: x % 2 == 1, range(1, 20)))
print(L)
'''

#装饰器：在代码运行期间动态增加功能

import functools

# 支持无参数的装饰器
def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper

@log
def now():
    print('2015-4-25')

# 调用 now 时会在其前面打印日志：call now():
now()
# 如果装饰器定义中缺少命令 @functools.wraps(func), now 的名称将不是 now 而是 wrapper
print(now.__name__)


# 同时支持有参数和无参数的装饰器
def log(text):
    if isinstance(text,str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kw):
                print('%s %s():' % (text, func.__name__))
                return func(*args, **kw)
            return wrapper
        return decorator
    else:
        @functools.wraps(text)
        def wrapper(*args, **kw):
            print('call %s():' % text.__name__)
            return text(*args, **kw)
        return wrapper

@log
def now():
    print('2018-10-01')

now()

@log('hello')
def now():
    print('2019-10-01')

# 调用 now 时会在其前面打印日志：hello now():
now()


import time, functools

# 计算函数执行时间的装饰器
def metric(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        start_time = time.time()
        fn(*args, **kw)
        end_time = time.time()
        print('%s executed in %s ms' % (fn.__name__, end_time - start_time))
        return fn(*args, **kw)
    return wrapper

# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;

@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

f = fast(11, 22)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')


class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.__gender = gender

    def get_gender(self):
        return self.__gender

    def set_gender(self, gender):
        if gender != 'male' and gender != 'female':
            raise ValueError("your gender must be male or female!")
        else:
            self.__gender = gender

# 测试:
bart = Student('Bart', 'male')
if bart.get_gender() != 'male':
    print('测试失败!')
else:
    bart.set_gender('female')
    if bart.get_gender() != 'female':
        print('测试失败!')
    else:
        print('测试成功!')

#bart.set_gender('男')

class Animal(object):
    # self 代表类的实例
    def run(self):
        print("animal is running")

    def run_twice(self,animal):
        animal.run()
        animal.run()

class Dog(Animal):
    def run(self):
        print("dog is running")

class Cat(Animal):
    def run(self):
        print("cat is running")

class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly...')

class Timer(object):
    def run(self):
        print('Start...')

t = Animal()
t.run_twice(Dog())
t.run_twice(Timer())

Animal.run_twice(Animal(),Dog())
Animal.run_twice(Animal(),Timer())