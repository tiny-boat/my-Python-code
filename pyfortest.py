#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
1.1 python 是怎样一种语言

python 是一门跨平台、开源的解释型高级动态编程语言，除了解释执行外，通过 py2exe、pyinstaller 或 cx_Freeze 等工具可将源代码编译为可执行文件，此外，python 还支持将源代码伪编译为字节码以提高程序执行速度。Python 支持命令式编程和函数式编程两种方式，完全支持面向对象程序设计。Python 语法简洁清晰、功能强大，拥有大量的几乎支持所有领域应用开发的成熟扩展库。Python 可以把多种不同语言编写的程序融合到一起实现无缝拼接，因而被称为“胶水语言”

1.3 python 编程规范与代码优化建议

a.严格使用缩进来体现代码的逻辑从属关系（4个空格键，不要使用tab键，它们对应的ASCII码值是不一样的）；
b.每个 import 语句只导入一个模块，按标准库、扩展库、自定义库的顺序依次导入，尽量避免导入整个库；
c.在每个类、函数定义和一段完整的功能代码之后增加空行，在运算符两侧各增加一个空格，逗号后面增加一个空格；
d.尽量不要写过长的语句；
e.对复杂的表达式建议在适当位置使用括号以增强可读性；
f.对关键代码和重要业务逻辑进行注释（“#” 后加一个空格，行内注释时，空两格以上）；
g.在开发和运行速度间尽量取得最佳平衡；
h.根据运算特点选择最合适的数据类型；（优先考虑集合和字典，元组次之，最后考虑列表和字符串）
i.充分利用生成器对象或类似迭代对象的惰性计算特点，尽量避免将其转换为列表、元组等类型；
j.减少内循环中的无关计算，尽量往外层提取;

1.6 安装扩展库的方法

pip download SomePackage[==version] 下载不安装
pip install SomePackage[==version] 下载并安装
pip install SomePackage.whl 通过 whl 文件安装
pip install -r requirements.txt 安装 requirements.txt 中指定的扩展库
pip install --upgrade SomePackage 升级
pip uninstall SomePackage[==version] 卸载
pip list 列出所有已安装模块
pip freeze 以 requirements 的格式列出所有已安装模块
requirements.txt 格式：numpy==1.15.0

1.7 标准库与扩展库中对象的导入与使用

import module_name [as other_name] 
from module_name import object_name [as other_name]
from module_name import *
模块导入搜索路径：首先是当前目录，而后是 sys 模块中 path 变量指定的目录，如果都没有抛出异常
从 zip 文件中导入模块：
import sys
sys.path.append('testZip.zip')
import Vector3
'''

'''
2.1 Python 常用内置对象

① python定义变量名需要注意：以字母或下划线开头，不允许空格和标点符号，不得使用关键字，不建议使用系统内置模块名、类型名或函数名以及已导入的模块名及其成员名，对大小写敏感

② 字典、集合的查找速度快于列表，列表快于元组；字典、集合增删元素速度也快于列表。

③ range、map、zip、filter、enumerate、reversed 等可迭代对象具有与 python 序列相似的操作方法，但其还有惰性求值的特点，这种惰性求值减少了对内存的占用。

2.2 Python 运算符与表达式

④ python 有逻辑运算符（and or not）、位运算符（仅用于整数 << >> & | 位异或^ 位求反~）、算术运算符（+ - * / // % ** @）、关系运算符（< > = <= >= == !=）以及 python 特有的成员测试运算符（in）、同一性测试运算符（is,同一性指二者具有相同的内存地址）以及集合运算符（交集& 并集| 对称差集^ 差集-）。

2.3 Python 关键字（通过 print(keyword.kwlist) 查看所有关键字）：

模块导入：import from as 
变量与运算符：global nonlocal None True False and or not in is 
生成器：yield
选择与循环：if elif else pass break continue for while 
函数：lambda def return 
类：class del 
异常处理：try except raise finally assert 
上下文管理：with

2.4 Python 内置函数（使用 dir(__builtins__) 查看所有内置对象）

① 类型转换
bin(integer),oct(integer),hex(integer)
int(num/str=0),int(str,base=10),float(num/str),complex(real,imag)
ord(str),chr(num),ascii(obj)
bytes(int),bytes(str,encoding),str(obj),str(bytes,encoding)
list(),tuple(),dict(),set(),frozenset()

② 类型判断
type(obj),isinstance(obj,class_or_tuple)

③ 最值与求和
max/min(iterable, *[, default=obj, key=func])
max/min(arg1, arg2, *args, *[, key=func])
sum(iterable, start=0, /)
/ 代表不允许以关键参数的形式进行传值，例如 sum([1,2,3],start=4) 会报错

④ 基本输入输出
input(prompt=None, /)
print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
sys.stdin.read(size=-1, /)
sys.stdin.readline(size=-1, /)
pprint.pprint()

⑤ 排序与逆序
sorted(iterable, key=None, reverse=False)
reversed(sequence)

⑥ 枚举与迭代
enumerate(iterable[, start]) -> iterator for index, value of iterable
iter(iterable) -> iterator
iter(callable, sentinel) -> iterator
next(iterator[, default])

⑦ map(),reduce(),filter()
map(func, *iterables) --> map object
reduce(function, sequence[, initial])
filter(function or None, iterable)

⑧ range(),zip(),eval(),exec()
range(stop)
range(start, stop[, step])
zip(iter1 [,iter2 [...]])
eval(source, globals=None, locals=None, /)
exec(source, globals=None, locals=None, /)



'''