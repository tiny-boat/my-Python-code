#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from urllib.request import ProxyHandler, build_opener
from urllib.request import install_opener, Request, urlopen
from lxml import etree
import re
import numpy as np
import pandas as pd

# 设置代理
proxy_handler = ProxyHandler({'http':'http://127.0.0.1:57858',\
                              'https':'https://127.0.0.1:57858'})
opener = build_opener(proxy_handler)
install_opener(opener)

# 获奖国家页面非公共地址列表
url = "https://en.wikipedia.org/wiki/2008_Summer_Olympics_medal_table"
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36"}
request = Request(url, headers=header)
response = urlopen(request)
html = etree.HTML(response.read())
privateURL_list = html.xpath('//table[@class="wikitable sortable plainrowheaders jquery-tablesorter"]/tbody/tr//a/@href')
# list(map(lambda x: re.sub('/wiki/|_at.*', '', x), privateURL_list))

# 获奖国家页面公共地址
publicURL = 'https://en.wikipedia.org'

privateURL = privateURL_list[17]
athURL = publicURL + privateURL
country = re.sub('/wiki/|_at.*', '', privateURL)
#print('正在爬取第 %d 个页面，剩余 %d 个页面：\n%s' % (i+1, len(privateURL_list)-i-1, athURL))
response = urlopen(Request(athURL, headers=header))
html = etree.HTML(response.read())
html.xpath('(//table[contains(@class, "wikitable")])[1]/tbody/tr[4]/td//text()')

# 初始化存储运动员数据的数组
colNum = 5
athData = np.zeros([1, colNum])

# 运动员数据爬取与存储
for i, privateURL in enumerate(privateURL_list):
#    if i > 63:
        athURL = publicURL + privateURL
        country = re.sub('/wiki/|_at.*', '', privateURL)
        print('正在爬取第 %d 个页面，剩余 %d 个页面：\n%s' % (i+1, len(privateURL_list)-i-1, athURL))
        response = urlopen(Request(athURL, headers=header))
        html = etree.HTML(response.read())
        athDataCol = html.xpath('(//table[contains(@class, "wikitable")])[1]/tbody/tr/th/text()')
        athDataCol = list(map(lambda x: re.sub('\n', '', x), athDataCol))
        j = 2
        while True:
        	str_j = str(j)
        	has_s = html.xpath('(//table[contains(@class, "wikitable")])[1]/tbody/tr[' + str_j + ']/td/s')
            data = html.xpath('(//table[contains(@class, "wikitable")])[1]/tbody/tr[' + str_j + ']/td//text()')
            if len(has_s) > 0:
            	continue
            elif len(data) == 0:
                break
            else:
            	# 处理 Romania 特殊的 Date: August 20/18 Nov 2016, 其被拆分为
            	# ‘August 20’,'/','18 Nov 2016\n' 三个部分，多出一个 ‘/’，导致
            	# 通用处理失效
            	data = list(map(lambda x: re.sub('\[\d\]|/', '\n', x), data))
            	# 过滤掉 '\n'，‘*’，‘*’ 源于某些运动员名字上多加了一个上标*
                data = list(filter(lambda x: x != '\n' and x != '*', data))
                # 合并团体项目中的多名运动员
                end = 2 - len(athDataCol)
                temp = data[1:end]
                if len(temp) > 1:
                    athNameSet = ', '.join(temp)
                    del data[2:end]
                    data[1] = athNameSet
                # 去掉字符中多余的 ‘\xa0’ 和 ‘\n’
                data = list(map(lambda x: re.sub('\xa0|\n', '', x), data))
                if len(data) > 4:
                    data = data[0:4]
                data.append(country)
                data = np.array([data])
                athData = np.concatenate((athData, data), axis=0)
                j += 1

# 删除数组中为使 np.concatenate 有效而多余的一行
athData = athData[1:]

# 将 numpy 数组转换为 pandas 数据框
def convert_df(df: pd.DataFrame, deep_copy: bool = True) -> pd.DataFrame:
    """Automatically converts columns that are worth stored as
    ``categorical`` dtype.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to convert.
    deep_copy: bool
        Whether or not to perform a deep copy of the original data frame.
    Returns
    -------
    pd.DataFrame
        Optimized copy of the input data frame.
    """
    return df.copy(deep=deep_copy).astype({
        col: 'category' for col in df.columns
        if df[col].nunique() / df[col].shape[0] < 0.5})

athDF2008 = pd.DataFrame(athData, columns=['mdal_type','name','sport','discipline','country'])
athDF2008 = convert_df(athDF2008)

athDF2008.to_excel('F:/web_crawler_result/Olympic/medalInfo2.xlsx', sheet_name='2008')

df_medal = pd.read_excel('F:/web_crawler_result/Olympic/medalInfo2.xlsx')


df3 = df2.groupby('country').sum(axis=0).sort_values(by=('Gold','Silver','Bronze'),ascending=False)

'''
privateURL = privateURL_list[3]
athURL = publicURL + privateURL
country = re.sub('/wiki/|_at.*', '', privateURL)
#print('正在爬取第 %d 个页面，剩余 %d 个页面：\n%s' % (i+1, len(privateURL_list)-i-1, athURL))
response = urlopen(Request(athURL, headers=header))
html = etree.HTML(response.read())
data = html.xpath('(//table[contains(@class, "wikitable")])[1]/tbody/tr/th/text()')
'''

'''
# 运动员获奖页面非公共地址列表
response1 = urllib.request.urlopen('http://www.theolympicdatabase.nl/winners.php?country_id=NULL&event_id=49&sport_id=&discipline_id=&is_team=0&sporter_name=&record_count=5000')
html1 = etree.HTML(response1.read())
privateURL_list = html1.xpath('//td[contains(@class, "datagrid_td")][2]/a/@href')

# 运动员获奖页面公共地址
publicURL = 'http://www.theolympicdatabase.nl/'

# 初始化存储运动员数据的数组
athData = np.zeros([1, 9])

# 运动员数据爬取与存储
colNum = 7
for i, privateURL in enumerate(privateURL_list):
    if i > 96:
        # 发送请求
        athURL = publicURL + privateURL
        print('正在爬取第 %d 个页面，剩余 %d 个页面：\n%s' % (i+1, len(privateURL_list)-i-1, athURL))
        response = urllib.request.urlopen(athURL)
        html = etree.HTML(response.read())
        # 获取运动员基本信息
        athBasicInfo = html.xpath('//div[@class="data_title"]//b//text()') + html.xpath('//table[@style="width:100%;height;25px;margin-top:4px;"][position()<3]//td[3]//text()')
        athBasicInfo = list(map(lambda x: x.replace('\xa0', ''), athBasicInfo))
        # 获取运动员获奖信息
        athMedalInfo = html.xpath('//table[@class="datagrid_header_table"]//tr[position()>1]/td//text() | //table[@class="datagrid_header_table"]//tr[position()>1]/td//@src')
        athMedalInfo = list(map(lambda x: re.sub('^images.*gif$', '', x), athMedalInfo))
        athMedalInfo = list(map(lambda x: re.sub('^images.*png$', '1', x), athMedalInfo))
        athMedalInfo = list(map(lambda x: re.sub('\xa0', '0', x), athMedalInfo))
        rowNum = len(athMedalInfo) // colNum
        athMedalInfo = np.array(athMedalInfo).reshape(rowNum, colNum)
        athMedalInfo = np.delete(athMedalInfo, 1, axis=1)
        # 整合运动员基本信息与获奖信息
        athData_sect = np.concatenate((np.array([athBasicInfo]*rowNum), athMedalInfo), axis=1)
        # 添加整合信息到存储运动员数据的数组
        athData = np.concatenate((athData, athData_sect), axis=0)

# 删除数组中为使 np.concatenate 有效而多余的一行
athData = athData[1:]

# 将 numpy 数组转换为 pandas 数据框
def convert_df(df: pd.DataFrame, deep_copy: bool = True) -> pd.DataFrame:
    """Automatically converts columns that are worth stored as
    ``categorical`` dtype.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to convert.
    deep_copy: bool
        Whether or not to perform a deep copy of the original data frame.
    Returns
    -------
    pd.DataFrame
        Optimized copy of the input data frame.
    """
    return df.copy(deep=deep_copy).astype({
        col: 'category' for col in df.columns
        if df[col].nunique() / df[col].shape[0] < 0.5})

athDF = pd.DataFrame(athData, columns=['name','sex','country','year','sport','discipline','gold_medal','sliver_medal','brozen_medal'])
athDF = convert_df(athDF)
athDF2008 = athDF[athDF.year == '2008']

athDF2008.to_excel('F:/web_crawler_result/Olympic/medalInfo.xlsx', sheet_name='2008')

df_medal = pd.read_excel('F:/web_crawler_result/Olympic/medalInfo.xlsx')
'''