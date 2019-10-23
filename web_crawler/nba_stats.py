
#from matplotlib import pyplot as plt
#from matplotlib.patches import Arc, Circle, Rectangle, Polygon
import requests
import json
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from urllib.parse import urlencode
#import seaborn as sns
#import matplotlib as mpl


'''
def Arc_fill(center, radius, theta1, theta2, resolution=50, **kwargs):
    # generate the points
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    points = np.vstack((radius*np.cos(theta) + center[0], 
                        radius*np.sin(theta) + center[1]))
    # build the polygon and add it to the axes
    poly = Polygon(points.T, closed=True, **kwargs)
    return poly

def get_page(url, params=None, isProx=False):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   + 'AppleWebKit/537.36 (KHTML, like Gecko) '
                   + 'Chrome/77.0.3865.90 Safari/537.36'}
        proxies = {'http': 'http://127.0.0.1:57858',
                   'https': 'https://127.0.0.1:57858'}
        if isProx:
            response = requests.get(url, headers=headers, proxies=proxies,
                                    params=params)
        else:
            response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response
        return None
    except RequestException:
        return None
'''

# -------
# 获取球员 ID
# -------

# url = "https://stats.nba.com/stats/commonallplayers?"
# params = {
#     "LeagueID": "00",
#     "Season": "2019",
#     "IsOnlyCurrentSeason": 0
# }
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
#            + 'AppleWebKit/537.36 (KHTML, like Gecko) '
#            + 'Chrome/77.0.3865.90 Safari/537.36'}
# try:
#     idInfo = requests.get(url, params=params, headers=headers).json()["resultSets"][0]
# except Exception as e:
#     print("\n错误：球员 ID 信息获取失败")
#     exit()
# else:
#     print("\n成功：球员 ID 信息获取成功\n")
#     idInfo = pd.DataFrame(idInfo["rowSet"], columns=idInfo["headers"])
#     #idInfo.to_csv('F:/web_crawler_results/NBA/idInfo.csv')

idInfo = pd.read_csv('f:/web_crawler_results/NBA/idInfo.csv', index_col=0)
playerIDList = idInfo["PERSON_ID"].tolist()

# --------
# 获取球员投篮数据
# --------

async def get_page(url, headers):
    session = aiohttp.ClientSession()
    response = await session.get(url, headers=headers)
    result = await response.json()
    session.close()
    return response

async def data_request(playerID):
    url = 'https://stats.nba.com/stats/shotchartdetail?'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
               + 'AppleWebKit/537.36 (KHTML, like Gecko) '
               + 'Chrome/77.0.3865.90 Safari/537.36'}
    params = {
        "SeasonType": "Regular Season",
        "TeamID": 0,
        "PlayerID": playerID,
        "PlayerPosition": '',
        "GameID": '',
        "Outcome": '',
        "Location": '',
        "Month": 0,
        "SeasonSegment": '',
        "DateFrom": '',
        "DateTo": '',
        "OpponentTeamID": 0,
        "VsConference": '',
        "VsDivision": '',
        "RookieYear": '',
        "GameSegment": '',
        "Period": 0,
        "LastNGames": 0,
        "ContextMeasure": "FGA",
    }
    url = url + urlencode(params)
    # try:
    shotInfoSec = await get_page(url, headers)
    print(shotInfoSec)
    # shotInfoSec = shotInfoSec.json()["resultSets"][0]
    # except Exception as e:
        # errorList.append(playerID)
        # print('错误：第{0}个球员（ID:{1}）数据获取失败'
              # .format(playerIDList.index(playerID) + 1, playerID))
    # else:
    # playerIndex = playerIDList.index(playerID) + 1
    # print('成功：第{0}个球员（ID:{1}）数据获取成功'
    #       .format(playerIndex, playerID))
    # if shotInfoSec["rowSet"] != []:
    #     shotInfoSec = pd.DataFrame(shotInfoSec["rowSet"], 
    #                                columns=shotInfoSec["headers"])
    #     shotInfo = shotInfo.append(shotInfoSec)
    # else:
    #     print('警告：第{0}个球员（ID:{1}）数据为空'.format(playerIndex, playerID))
    #     emptyList.append(playerID)
    print('\n')

shotInfo, errorList, emptyList = pd.DataFrame(), [], []

tasks = [asyncio.ensure_future(data_request(playerID)) for playerID in playerIDList[3:5]]
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))

if emptyList != []:
    print('警告：以下球员 ID 数据为空\n{0}\n'.format(emptyList))
    
if errorList != []:
    print('错误：以下球员 ID 数据获取失败\n{0}\n'.format(errorList))

shotInfo.to_csv('F:/web_crawler_results/NBA/shotInfo3.csv')


import requests
import time
from multiprocessing import Pool
import pandas as pd
from requestData import request_data

shotInfo, erl, eml = pd.DataFrame(), [], []
start = time.time()

idInfo = pd.read_csv('f:/web_crawler_results/NBA/idInfo.csv', index_col=0)
idl = idInfo["PERSON_ID"].tolist() 

resultL = []
p = Pool(10)
for playerID in idl:
    result = p.apply_async(request_data, args=(playerID, ))
    resultL.append(result)

# print('Waiting for all subprocesses done...')
p.close()
p.join()

for result in resultL:
    shotInfo = shotInfo.append(pd.DataFrame(result.get()))

print(time.time() - start)


import requests
import time
import pandas as pd

idInfo = pd.read_csv('f:/web_crawler_results/NBA/idInfo.csv', index_col=0)
playerIDList = idInfo["PERSON_ID"].tolist()

start = time.time()
# for year in range(1896, 2017, 4):
for playerID in playerIDList[0:5]:
    # url = 'https://en.wikipedia.org/wiki/' + str(year) + '_Summer_Olympics_medal_table'
    url = 'https://stats.nba.com/stats/shotchartdetail?'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
               + 'AppleWebKit/537.36 (KHTML, like Gecko) '
               + 'Chrome/77.0.3865.90 Safari/537.36'}
    proxies = {'http': 'http://127.0.0.1:50926',
               'https': 'https://127.0.0.1:50926'}
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
    resp = requests.get(url, params=params, proxies=proxies, headers=headers)
    print('^^^^^')
    print(resp.url)
    print('&&&&&')

print(time.time() - start)



loop.run_until_complete(asyncio.sleep(0.250))
loop.close()
# shotInfo, errorList, emptyList = pd.DataFrame(), [], []
# for i, playerID in enumerate(playerIDList):
#     url = 'https://stats.nba.com/stats/shotchartdetail?'
#     params = {
#         "SeasonType": "Regular Season",
#         "TeamID": 0,
#         "PlayerID": playerID,
#         "PlayerPosition": '',
#         "GameID": '',
#         "Outcome": '',
#         "Location": '',
#         "Month": 0,
#         "SeasonSegment": '',
#         "DateFrom": '',
#         "DateTo": '',
#         "OpponentTeamID": 0,
#         "VsConference": '',
#         "VsDivision": '',
#         "RookieYear": '',
#         "GameSegment": '',
#         "Period": 0,
#         "LastNGames": 0,
#         "ContextMeasure": "FGA",
#     }
#     try:
#         shotInfoSec = get_page(url, params).json()["resultSets"][0]
#     except Exception as e:
#         errorList.append(playerID)
#         print('错误：第{0}个球员（ID:{1}）数据获取失败'.format(i+1, playerID))
#     else:
#         print('成功：第{0}个球员（ID:{1}）数据获取成功'.format(i+1, playerID))
#         if shotInfoSec["rowSet"] != []:
#             shotInfoSec = pd.DataFrame(shotInfoSec["rowSet"], 
#                                        columns=shotInfoSec["headers"])
#             shotInfo = shotInfo.append(shotInfoSec)
#         else:
#             print('警告：第{0}个球员（ID:{1}）数据为空'.format(i+1, playerID))
#             emptyList.append(playerID)
#     print('\n')

# shotInfo.to_csv('F:/web_crawler_results/NBA/shotInfo.csv')

# if errorList != []:
#     print('以下球员 ID 数据获取失败')
#     for value in errorList:
#         print(value)
#     print('\n')

# if emptyList != []:
#     print('以下球员 ID 数据为空')
#     for value in emptyList:
#         print(value)
#     print('\n')


# '''
#     --------
#     读取数据
#     --------
# '''

# shotDF = pd.read_csv('f:/web_crawler_results/NBA/shotInfo2.csv', index_col=0,
#                      low_memory=False)


# '''
#     ---------
#     绘制篮球场
#     ---------
# '''

# def draw_ball_field(color='#003370', lw=2):
#     # 新建一个大小为(6,6)的绘图窗口
#     plt.figure(figsize=(5.36, 5.06), frameon=False)
#     # 获得当前的Axes对象ax,进行绘图
#     ax = plt.gca(frame_on=False)
#     # 设置坐标轴范围
#     ax.set_xlim(-268, 268)
#     ax.set_ylim(440.5, -65.5)
#     # 消除坐标轴刻度
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # 添加备注信息
#     # plt.annotate('By xiao F', xy=(100, 160), xytext=(178, 418))
#     # 对篮球场进行底色填充
#     lines_outer_rec = Rectangle(xy=(-268, -65.5), width=536, height=506,
#                                 color='#f1f1f1', fill=True, zorder=0)
#     # 设置篮球场填充图层为最底层
#     # lines_outer_rec.set_zorder(0)
#     # 将rec添加进ax
#     ax.add_patch(lines_outer_rec)
#     # 绘制篮筐,半径为7.5
#     circle_ball = Circle(xy=(0, 0), radius=7.5, linewidth=lw, color=color,
#                          fill=False, zorder=4)
#     # 将circle添加进ax
#     ax.add_patch(circle_ball)
#     # 绘制限制区
#     restricted_arc = Arc(xy=(0, 0), width=80, height=80, theta1=0,
#                          theta2=180, linewidth=lw, color=color, 
#                          fill=False, zorder=4)
#     ax.add_patch(restricted_arc)
#     # 绘制篮板,尺寸为(60,1)
#     plate = Rectangle(xy=(-30, -7.5), width=60, height=-1, linewidth=lw,
#                       color=color, fill=False, zorder=4)
#     # 将rec添加进ax
#     ax.add_patch(plate)
#     # 绘制2分区的外框线,尺寸为(160,190)
#     outer_rec_fill = Rectangle(xy=(-80, -47.5), width=160, height=190,
#                                linewidth=lw, color="#fefefe", fill=True, zorder=2)
#     outer_rec = Rectangle(xy=(-80, -47.5), width=160, height=190,
#                           linewidth=lw, color=color, fill=False, zorder=4)
#     # 将rec添加进ax
#     ax.add_patch(outer_rec_fill)
#     ax.add_patch(outer_rec)
#     # 绘制罚球站位点
#     lane_space_left1 = Rectangle(xy=(-90, 20.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_left2 = Rectangle(xy=(-90, 30.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_left3 = Rectangle(xy=(-90, 60.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_left4 = Rectangle(xy=(-90, 90.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_right1 = Rectangle(xy=(80, 20.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_right2 = Rectangle(xy=(80, 30.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_right3 = Rectangle(xy=(80, 60.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     lane_space_right4 = Rectangle(xy=(80, 90.5), width=10, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     ax.add_patch(lane_space_left1)
#     ax.add_patch(lane_space_left2)
#     ax.add_patch(lane_space_left3)
#     ax.add_patch(lane_space_left4)
#     ax.add_patch(lane_space_right1)
#     ax.add_patch(lane_space_right2)
#     ax.add_patch(lane_space_right3)
#     ax.add_patch(lane_space_right4)
#     # 绘制罚球区域圆圈,半径为60
#     circle_punish1 = Arc(xy=(0, 142.5), width=120, height=120, theta1=0,
#                          theta2=180, linewidth=lw, color=color, 
#                          fill=False, zorder=4)
#     circle_punish2 = Arc(xy=(0, 142.5), width=120, height=120, theta1=180,
#                          theta2=360, linewidth=lw, linestyle='--', 
#                          color=color, fill=False, zorder=4)
#     # circle_punish = Circle(xy=(0, 142.5), radius=60, linewidth=lw,
#     #                       color=color, fill=False)
#     # 将circle添加进ax
#     ax.add_patch(circle_punish1)
#     ax.add_patch(circle_punish2)
#     # 绘制低位防守区域标志线
#     hash_marks_left1 = Rectangle(xy=(-110, -47.5), width=0, height=5,
#                                 linewidth=lw, color=color,
#                                 fill=False, zorder=4)
#     hash_marks_right1 = Rectangle(xy=(110, -47.5), width=0, height=5,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     hash_marks_left2 = Rectangle(xy=(-50, 82.5), width=5, height=0,
#                                 linewidth=lw, color=color,
#                                 fill=False, zorder=4)
#     hash_marks_right2 = Rectangle(xy=(45, 82.5), width=5, height=0,
#                                  linewidth=lw, color=color,
#                                  fill=False, zorder=4)
#     ax.add_patch(hash_marks_left1)
#     ax.add_patch(hash_marks_right1)
#     ax.add_patch(hash_marks_left2)
#     ax.add_patch(hash_marks_right2)
#     # 绘制三分线的左边线
#     three_left_rec_fill = Rectangle(xy=(-220, -47.5), width=440, height=140,
#                                     ec="#dfdfdf", fc="#dfdfdf", 
#                                     fill=True, zorder=1)
#     three_left_rec = Rectangle(xy=(-220, -47.5), width=0, height=140,
#                                linewidth=lw, color=color, fill=False, zorder=4)
#     # 将rec添加进ax
#     ax.add_patch(three_left_rec_fill)
#     ax.add_patch(three_left_rec)
#     # 绘制三分线的右边线
#     three_right_rec = Rectangle(xy=(220, -47.5), width=0, height=140,
#                                 linewidth=lw, color=color, 
#                                 fill=False, zorder=4)
#     # 将rec添加进ax
#     ax.add_patch(three_right_rec)
#     # 绘制三分线的圆弧,圆心为(0,0),半径为238.66,起始角度为22.8,结束角度为157.2
#     three_arc_fill = Arc_fill(center=(0, 0), radius=238.66, theta1=22.8, 
#                               theta2=157.2, resolution=50, linewidth=0,
#                               ec="#dfdfdf", fc="#dfdfdf", fill=True, zorder=1)
#     three_arc = Arc(xy=(0, 0), width=477.32, height=477.32, theta1=22.8,
#                     theta2=157.2, linewidth=lw, color=color,
#                     fill=False, zorder=4)
#     # 将arc添加进ax
#     ax.add_patch(three_arc_fill)
#     ax.add_patch(three_arc)
#     # 绘制中场标记线
#     midcourt_area_marker_left = Rectangle(xy=(-250, 232.5), width=30, height=0,
#                                           color=color, linewidth=lw, 
#                                           fill=False, zorder=4)
#     midcourt_area_marker_right = Rectangle(xy=(220, 232.5), width=30, height=0,
#                                            color=color, linewidth=lw,
#                                            fill=False, zorder=4)
#     ax.add_patch(midcourt_area_marker_left)
#     ax.add_patch(midcourt_area_marker_right)
#     # 绘制中场处的外半圆,半径为60
#     center_outer_arc = Arc(xy=(0, 422.5), width=120, height=120, theta1=180,
#                            theta2=0, linewidth=lw, color=color,
#                            fill=False, zorder=4)
#     # 将arc添加进ax
#     ax.add_patch(center_outer_arc)
#     # 绘制中场处的内半圆,半径为20
#     center_inner_arc = Arc(xy=(0, 422.5), width=40, height=40, theta1=180,
#                            theta2=0, linewidth=lw, color=color,
#                            fill=False, zorder=4)
#     # 将arc添加进ax
#     ax.add_patch(center_inner_arc)
#     # 绘制篮球场外框线,尺寸为(500,470)
#     lines_outer_rec = Rectangle(xy=(-250, -47.5), width=500, height=470,
#                                 linewidth=lw, color=color,
#                                 fill=False, zorder=4)
#     # 将rec添加进ax
#     ax.add_patch(lines_outer_rec)
#     return ax

# axs = draw_ball_field()
# #plt.show()
# '''
#     -------------
#     绘制投篮热点图
#     -------------
# '''

# # 分类数据
# shotDF_curry = shotDF[shotDF['PLAYER_NAME'] == 'James Harden']
# shotDF_curry = shotDF_curry[shotDF_curry['GAME_DATE']>20120901]
# shotDF_curry = shotDF_curry[shotDF_curry['GAME_DATE']<20130901]
# shotDF_curry_made = shotDF_curry[shotDF_curry['EVENT_TYPE'] == 'Made Shot']
# shotDF_curry_miss = shotDF_curry[shotDF_curry['EVENT_TYPE'] == 'Missed Shot']

# # 绘制散点图
# axs.scatter(x=shotDF_curry_miss['LOC_X'], y=shotDF_curry_miss['LOC_Y'], s=30,
#             marker='x', c='#ad0a0f', zorder=3)
# axs.scatter(x=shotDF_curry_made['LOC_X'], y=shotDF_curry_made['LOC_Y'], s=30, 
#             marker='o', linewidth=1.5, ec='green', fc='none', zorder=3)

# plt.show()

# # '''
# #     -------------
# #     绘制投篮热力图
# #     -------------
# # '''

# # def colormap():
# #     """
# #     颜色转换
# #     """
# #     return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#C5C5C5',
# #                                                         '#9F9F9F', '#706A7C',
# #                                                         '#675678', '#713A71',
# #                                                         '#9D3E5E', '#BC5245',
# #                                                         '#C86138', '#C96239',
# #                                                         '#D37636', '#D67F39',
# #                                                         '#DA8C3E', '#E1A352'],
# #                                                         256)


# # # 绘制球员投篮热力图
# # shot_heatmap = sns.jointplot(df['width'], df['height'], stat_func=None,
# #                              kind='kde', space=0, color='w', cmap=colormap())
# # # 设置图像大小
# # shot_heatmap.fig.set_size_inches(6, 6)
# # # 图像反向
# # ax = shot_heatmap.ax_joint
# # # 绘制投篮散点图
# # ax.scatter(x=df['width'], y=df['height'], s=0.1, marker='o', color="w",
# #            alpha=1)
# # # 添加篮球场
# # draw_ball_field(color='w', lw=2)
# # # 将坐标轴颜色更改为白色
# # lines = plt.gca()
# # lines.spines['top'].set_color('none')
# # lines.spines['left'].set_color('none')
# # # 去除坐标轴标签
# # ax.axis('off')


# # from selenium import webdriver
# # from selenium.webdriver.common.by import By
# # from selenium.webdriver.common.keys import keys
# # from selenium.webdriver.common.keys import Keys
# # from selenium.webdriver.support import expected_conditions as EC
# # from selenium.webdriver.support.wait import WebDriverWait
# # browser = webdriver.Chrome()
# # #try:
# # browser.get('https://www.baidu.com')
# # input = browser.find_element_by_id('kw')
# # input.send_keys('灰汤')
# # input.send_keys(Keys.ENTER)
# # wait = WebDriverWait(browser, 10)
# # wait.until(EC.presence_of_element_located((By.ID, 'content_left')))
# # print(browser.current_url)
# #     #print(browser.get_cookies())
# #     #print(browser.page_source)
# # #finally:
# # #    browser.close()


# import asyncio
# import aiohttp
# import time
# import pandas as pd

# idInfo = pd.read_csv('f:/web_crawler_results/NBA/idInfo.csv', index_col=0)
# playerIDList = idInfo["PERSON_ID"].tolist()

# start = time.time()
# async def get(url, params, headers, proxy):
#     async with aiohttp.ClientSession() as session:
#         # print('********')
#         # time.sleep(2)
#         # print('--------')
#         async with session.get(url, params=params, headers=headers, proxy=proxy) as resp:
#             print('^^^^^')
#             print(resp.url)
#             # assert resp.url == 'https://stats.nba.com/teams/boxscores-traditional/?Season=2014-15&SeasonType=Regular%20Season'
#             print('&&&&&')
#             # print(await resp.text())
#             # print(await resp.text())
#     # session = aiohttp.ClientSession()
#     # response = await session.get(url, headers=headers)
#     # result = await response.text()
#     # session.close()
#     return resp

# async def request(playerID):
#     # url = 'https://en.wikipedia.org/wiki/' + str(year) + '_Summer_Olympics_medal_table'
#     proxy = 'http://127.0.0.1:50926'
#     # params = None
#     # url = 'https://stats.nba.com/'
#     # headers, params = None, None
#     # url = 'https://stats.nba.com/teams/boxscores-traditional/'
#     # params = {'Season':'2014-15', 'SeasonType':'Regular%20Season'}
#     # url = 'https://stats.nba.com/stats/shotchartdetail?SeasonType=Regular%20Season&TeamID=0&PlayerID=201939&PlayerPosition=&GameID=&Outcome=&Location=&Month=0&SeasonSegment=&DateFrom=&DateTo=&OpponentTeamID=0&VsConference=&VsDivision=&RookieYear=&GameSegment=&Period=0&LastNGames=0&ContextMeasure=FGA'
#     url = 'https://stats.nba.com/stats/shotchartdetail'
#     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
#                + 'AppleWebKit/537.36 (KHTML, like Gecko) '
#                + 'Chrome/77.0.3865.90 Safari/537.36',
#                'Connection': 'close'}
#     params = {
#         "SeasonType": "Regular Season",
#         "TeamID": '0',
#         "PlayerID": playerID,
#         "PlayerPosition": '',
#         "GameID": '',
#         "Outcome": '',
#         "Location": '',
#         "Month": '0',
#         "SeasonSegment": '',
#         "DateFrom": '',
#         "DateTo": '',
#         "OpponentTeamID": '0',
#         "VsConference": '',
#         "VsDivision": '',
#         "RookieYear": '',
#         "GameSegment": '',
#         "Period": '0',
#         "LastNGames": '0',
#         "ContextMeasure": "FGA",
#     }
#     # print('waiting for', url)
#     result = await get(url, params, headers, proxy)
#     #print(result)
#     # print('Get response from', url, 'Result:', result)

# tasks = [asyncio.ensure_future(request(playerID)) for playerID in playerIDList[0:5]]
# loop = asyncio.get_event_loop()
# loop.run_until_complete(asyncio.wait(tasks))
# print(time.time() - start)

