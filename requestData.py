#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def request_data(playerID, playerIDList, errorList, emptyList):
    import requests
    url = 'https://stats.nba.com/stats/shotchartdetail?'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
               + 'AppleWebKit/537.36 (KHTML, like Gecko) '
               + 'Chrome/77.0.3865.90 Safari/537.36'}
    # proxies = {'http': 'http://127.0.0.1:50926',
    #            'https': 'https://127.0.0.1:50926'}
    params = {
        "SeasonType": "Regular Season",
        "TeamID": '0',
        "PlayerID": playerID,
        "PlayerPosition": '',
        "GameID": '',
        "Outcome": '',
        "Location": '',
        "Month": '0',
        "SeasonSegment": '',
        "DateFrom": '',
        "DateTo": '',
        "OpponentTeamID": '0',
        "VsConference": '',
        "VsDivision": '',
        "RookieYear": '',
        "GameSegment": '',
        "Period": '0',
        "LastNGames": '0',
        "ContextMeasure": "FGA",
    }
    # resp = requests.get(url, params=params, proxies=proxies, headers=headers)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20).json()["resultSets"][0]["rowSet"]
    except Exception as e:
        errorList.append(playerID)
        print('错误：第{0}个球员（ID:{1}）数据获取失败'.format(playerIDList.index(playerID) + 1, playerID))
        resp = None
    else:
        print('成功：第{0}个球员（ID:{1}）数据获取成功'.format(playerIDList.index(playerID) + 1, playerID))
        if resp == []:
            emptyList.append(playerID)
            print('警告：该球员数据为空！')
    print('\n')
    return resp
