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

# 获奖国家页面公共地址
publicURL = 'https://en.wikipedia.org'

# 初始化存储运动员数据的数组
colNum = 4
athData = np.zeros([1, colNum])

# 运动员数据爬取与存储
for i, privateURL in enumerate(privateURL_list):
i = 2
privateURL = privateURL_list[i]
athURL = publicURL + privateURL
response = urlopen(Request(athURL, headers=header))
html = etree.HTML(response.read())
j = 2
while True:
    data = html.xpath('//table[@class="wikitable sortable" and @style="font-size:95%"][1]/tbody/tr['+str(j)+']/td//text()')
    print('---第{0}个运动员---')
    print("1.{0}".format(data))
    if len(data) == 0:
        break
    else:
        data = list(filter(lambda x: x != '\n' and x != '*', data))
        print("2.{0}".format(data))
        temp = data[1:-3]
        if len(temp) > 1:
            athNameSet = ', '.join(temp)
            del data[2:-3]
            data[1] = athNameSet
        print("3.{0}".format(data))
        data = list(map(lambda x: re.sub('\xa0 | \n', '', x), data))
        if len(data) > 4:
        	data = data[:4]
        print("4.{0}".format(data))
        data = np.array([data])
        print("5.{0}\n".format(data))
        athData = np.concatenate((athData, data), axis=0)
        j += 1




# 运动员数据爬取与存储
for i, privateURL in enumerate(privateURL_list):
    # 发送请求
    athURL = publicURL + privateURL
    print('正在爬取第 %d 个页面，剩余 %d 个页面：\n%s' % (i+1, len(privateURL_list)-i-1, athURL))
    response = urlopen(Request(athURL, headers=header))
    html = etree.HTML(response.read())
    athData_sec = html.xpath('//table[@class="wikitable sortable jquery-tablesorter"]//tbody/tr//text()')
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