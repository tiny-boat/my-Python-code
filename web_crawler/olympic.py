#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from urllib.request import ProxyHandler, build_opener
from urllib.request import install_opener, Request, urlopen
from urllib.error import URLError
from lxml import etree
import re
import numpy as np
import pandas as pd


# ----------------
# 1 data crawling
# ----------------


def proc_data(data, data_header, country):
    data = list(map(lambda x: re.sub(r'\[\d\]|/|.*\n', '\n', x), data))
    data = list(filter(lambda x: x != '\n' and x != '*' and
                       x != '\xa0' and x != ' ', data))

    # merge names of athletes for team events
    end = 2 - len(data_header)
    temp = data[1:end]
    if len(temp) > 1:
        athNameSet = ', '.join(temp)
        del data[2:end]
        data[1] = athNameSet

    data = list(map(lambda x: re.sub('\xa0|\n', '', x), data))
    if len(data) > 4:
        data = data[0:4]
    data.append(country)
    return np.array([data])


def select_property_name(html, country):
    data = html.xpath('(//table[contains(@class, "wikitable sortable"'
                      + ')])[1]/tbody/tr[2]/td//text()')
    data_header = html.xpath('(//table[contains(@class, "wikitable sortable"'
                             + ')])[1]/tbody/tr/th/text()')
    data = proc_data(data, data_header, country)
    if data.size == 5:
        return '"wikitable sortable"'
    else:
        return '"wikitable"'


# ---1.1 preparation---

# set proxy
proxy_handler = ProxyHandler({'http': 'http://127.0.0.1:57858',
                              'https': 'https://127.0.0.1:57858'})
opener = build_opener(proxy_handler)
install_opener(opener)

# get and set page url
publicURL = 'https://en.wikipedia.org'
year = '2008'
url = 'https://en.wikipedia.org/wiki/' + year + '_Summer_Olympics_medal_table'
# set UA，disguise crawler as browser
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          + "AppleWebKit/537.36 (KHTML, like Gecko) "
          + "Chrome/77.0.3865.90 Safari/537.36"}
response = urlopen(Request(url, headers=header))
html = etree.HTML(response.read())
privateURL_list = html.xpath('//table[@class="wikitable sortable '
                             + 'plainrowheaders jquery-tablesorter"]'
                             + '/tbody/tr//a/@href')
countryList = list(map(lambda x: re.sub('/wiki/|_at.*', '', x),
                       privateURL_list))

# initialize a ndarray athData for athletes' data storage
# initialize a list for crawler failure page
# initialize a list for crawler problem page
athData, errorList, problemList = np.zeros([1, 5]), [], []

# ---1.2 crawling---

print('\nCrawling start...\n')

# request, extract and store pages
for i, privateURL in enumerate(privateURL_list):
    has_problem = False
    athURL = publicURL + privateURL
    print('------ requesting {0}th page，{1} pages are remaining ------\n'
          .format(i + 1, len(privateURL_list) - i - 1))
    print('[URL]:     {0}'.format(athURL))

    for request_num in range(1, 6):
        try:
            response = urlopen(Request(athURL, headers=header))
            html = etree.HTML(response.read())
        except Exception as e:
            if request_num < 5:
                print(('[Warning]:   {0}th request was failed，'
                      + 'trying {1}th request...')
                      .format(request_num, request_num + 1))
            else:
                errorList.append(athURL)
                print('[Error]:   two much failure，requesting '
                      + 'next page soon...\n')
        else:
            print('[Success]: extracting data from page...')
            country = countryList[i]
            propertyName = select_property_name(html, country)
            athDataHeader = html.xpath('(//table[contains(@class, '
                                       + propertyName + ')])[1]'
                                       + '/tbody/tr/th/text()')
            # extract data of page row by row
            j = 2
            while True:
                str_j = str(j)
                has_s = html.xpath('(//table[contains(@class, ' + propertyName
                                   + ')])[1]/tbody/tr[' + str_j + ']/td/s')
                rowData = html.xpath('(//table[contains(@class, '
                                     + propertyName + ')])[1]/tbody/tr['
                                     + str_j + ']/td//text()')
                # if <s></s> exits，ignore current rowData
                # <s> represents strickout
                if len(has_s) > 0:
                    j += 1
                    continue
                # current page's data was extracted when rowData is null
                # so stop while iteration
                elif len(rowData) == 0:
                    break
                else:
                    rowData = proc_data(rowData, athDataHeader, country)
                    # add current page's data to athData
                    try:
                        athData = np.concatenate((athData, rowData), axis=0)
                    except ValueError as e:
                        problemList.append(athURL)
                        has_problem = True
                    else:
                        j += 1

            if has_problem is True:
                print('[Warning]: current page may be incompatible with rule'
                      + ' of crawler, check manually')
                print('[Error]:   incomplete data extraction，'
                      + 'requesting next page soon...\n')
            elif i < len(privateURL_list) - 1:
                print('[Success]: data extraction was done，'
                      + 'requesting next page soon...\n')
            else:
                print('[Success]: data extraction was done\n')

            break

if len(errorList) > 0:
    print('[Warning]: following pages\' crawling are failed！\n')
    for item in errorList:
        print(item)
    print('\nCheck network, program and web pages carefully, '
          + 'failure is the mother of success （づ￣3￣）づ╭❤～\n')
elif len(problemList) > 0:
    print('[Warning]: following pages\' crawling are incomplete！\n')
    for item in problemList:
        print(item)
    print('\nCheck network, program and web pages carefully, '
          + 'failure is the mother of success （づ￣3￣）づ╭❤～\n')
else:
    print('Congratulations to you, all are successful（づ￣3￣）づ╭❤～\n')

# ---1.3 output data---

athData = athData[1:]
athDF = pd.DataFrame(athData, columns=['mdal_type', 'name', 'sport',
                                       'discipline', 'country'])
athDF.to_excel('F:/web_crawler_result/Olympic/medalInfo' + year + '.xlsx',
               sheet_name=year)


# ----------------
# 2 data analysis
# ----------------

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


df_medal = pd.read_excel('F:/web_crawler_result/Olympic/medalInfo'
                         + year + '.xlsx')
df_medal = convert_df(df_medal)


# df3 = df2.groupby('country').sum(axis=0).sort_values(by=('Gold','Silver',
#                                                      'Bronze'),ascending=False)

'''
privateURL = privateURL_list[5]
athURL = publicURL + privateURL
country = re.sub('/wiki/|_at.*', '', privateURL)
response = urlopen(Request(athURL, headers=header))
html = etree.HTML(response.read())
html.xpath('(//table[contains(@class, "wikitable sortable")])[1]/tbody/tr[7]
/td//text()')
'''
