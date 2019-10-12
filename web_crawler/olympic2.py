#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import requests
from requests.exceptions import RequestException
from lxml import etree
import re
import time


def get_one_page(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   + 'AppleWebKit/537.36 (KHTML, like Gecko) '
                   + 'Chrome/77.0.3865.90 Safari/537.36'}
        proxies = {'http': 'http://127.0.0.1:57858',
                   'https': 'https://127.0.0.1:57858'}
        response = requests.get(url, headers=headers, proxies=proxies)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None


def parse_one_page(html, country):
    pattern = re.compile('<tr>\n*<td><span data-sort-value=\"0[123]&#160;!\"><img.*?/>&#160;(.*?)</.*?<td>(.*?)</td>.*?<td>.*?<a.*?>(.*?)</a>.*?<td>.*?<a.*?>(.*?)</a>', re.S)
    items = re.findall(pattern, html)
    num = 0
    for item in items:
        item0 = item[0]
        if item0 in ('Gold', 'Silver', 'Bronze'):
            item1 = re.sub(r'<.*?>|\*|\n|\[\d\]', '  ', item[1]).strip()
            item1 = re.sub(r'\s{2,}', ', ', item1)
            yield {
                    'medal_type': item0,
                    'name': item1,
                    'sport': item[2],
                    'discipline': item[3],
                    'country': country
            }


def write_to_file(content):
    with open('f:/web_crawler_result/Olympic/result2000.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\n')


def main(url, country):
    html = get_one_page(url)
    for item in parse_one_page(html, country):
        print(item)
        write_to_file(item)


if __name__ == '__main__':
    publicURL = 'https://en.wikipedia.org'
    year = '2000'
    url = 'https://en.wikipedia.org/wiki/' + year + '_Summer_Olympics_medal_table'
    html = etree.HTML(get_one_page(url))
    privateURL_list = html.xpath('//table[@class="wikitable sortable '
                                     + 'plainrowheaders jquery-tablesorter"]'
                                     + '/tbody/tr//a/@href')
    countryList = list(map(lambda x: re.sub('/wiki/|_at.*', '', x),
                           privateURL_list))
    for i, privateURL in enumerate(privateURL_list):
        main(publicURL + privateURL, countryList[i])
        #time.sleep(1)
