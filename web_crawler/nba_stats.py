
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib as mpl


# --------
# 数据获取
# --------

header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          + "AppleWebKit/537.36 (KHTML, like Gecko) "
          + "Chrome/77.0.3865.90 Safari/537.36"}
# 球员职业生涯时间
start_year, end_year = 2018, 2019
# 球员ID
player_id = ('201939', )
for player in player_id:
    for year in range(start_year, end_year):
        season = str(year) + '-' + str(year + 1)[-2:]
params = {
    # "LeagueID": "00",
    # "Season": season,
    "SeasonType": "Regular Season",
    "TeamID": 0,
    "PlayerID": player,
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
    # "Position": '',
    "RookieYear": '',
    "GameSegment": '',
    "Period": 0,
    "LastNGames": 0,
    # "ClutchTime": '',
    # "AheadBehind": '',
    # "PointDiff": '',
    # "RangeType": 0,
    # "StartPeriod": 1,
    # "EndPeriod": 10,
    # "StartRange": 0,
    # "EndRange": 28800,
    # "ContextFilter": '',
    "ContextMeasure": "FGA",
}

        # 请求网址
        url = 'https://stats.nba.com/stats/shotchartdetail?'
        response = requests.get(url=url, headers=headers, params=params)
        result = json.loads(response.text)
        # 获取数据
        for item in result['resultSets'][0]['rowSet']:
            # 是否进球得分
            flag = item[10]
            # 横坐标
            loc_x = str(item[17])
            # 纵坐标
            loc_y = str(item[18])
            with open('f:/web_crawler_result/NBA/curry.csv', 'a+') as f:
                f.write(loc_x + ',' + loc_y + ',' + flag + '\n')

'''
    --------
    读取数据
    --------
'''

df = pd.read_csv('f:/web_crawler_result/NBA/curry.csv', header=None,
                 names=['width', 'height', 'type'])

'''
    ---------
    绘制篮球场
    ---------
'''

def draw_ball_field(color='#20458C', lw=2):

    # 新建一个大小为(6,6)的绘图窗口
    plt.figure(figsize=(6, 6))
    # 获得当前的Axes对象ax,进行绘图
    ax = plt.gca()
    # 设置坐标轴范围
    ax.set_xlim(-250, 250)
    ax.set_ylim(422.5, -47.5)
    # 消除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 添加备注信息
    plt.annotate('By xiao F', xy=(100, 160), xytext=(178, 418))

    # 对篮球场进行底色填充
    lines_outer_rec = Rectangle(xy=(-250, -47.5), width=500, height=470,
                                linewidth=lw, color='#F0F0F0', fill=True)
    # 设置篮球场填充图层为最底层
    lines_outer_rec.set_zorder(0)
    # 将rec添加进ax
    ax.add_patch(lines_outer_rec)

    # 绘制篮筐,半径为7.5
    circle_ball = Circle(xy=(0, 0), radius=7.5, linewidth=lw, color=color,
                         fill=False)
    # 将circle添加进ax
    ax.add_patch(circle_ball)

    # 绘制篮板,尺寸为(60,1)
    plate = Rectangle(xy=(-30, -7.5), width=60, height=-1, linewidth=lw,
                      color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(plate)

    # 绘制2分区的外框线,尺寸为(160,190)
    outer_rec = Rectangle(xy=(-80, -47.5), width=160, height=190,
                          linewidth=lw, color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(outer_rec)

    # 绘制2分区的内框线,尺寸为(120,190)
    inner_rec = Rectangle(xy=(-60, -47.5), width=120, height=190,
                          linewidth=lw, color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(inner_rec)

    # 绘制罚球区域圆圈,半径为60
    circle_punish = Circle(xy=(0, 142.5), radius=60, linewidth=lw,
                           color=color, fill=False)
    # 将circle添加进ax
    ax.add_patch(circle_punish)

    # 绘制三分线的左边线
    three_left_rec = Rectangle(xy=(-220, -47.5), width=0, height=140,
                               linewidth=lw, color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(three_left_rec)

    # 绘制三分线的右边线
    three_right_rec = Rectangle(xy=(220, -47.5), width=0, height=140,
                                linewidth=lw, color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(three_right_rec)

    # 绘制三分线的圆弧,圆心为(0,0),半径为238.66,起始角度为22.8,结束角度为157.2
    three_arc = Arc(xy=(0, 0), width=477.32, height=477.32, theta1=22.8,
                    theta2=157.2, linewidth=lw, color=color, fill=False)
    # 将arc添加进ax
    ax.add_patch(three_arc)

    # 绘制中场处的外半圆,半径为60
    center_outer_arc = Arc(xy=(0, 422.5), width=120, height=120, theta1=180,
                           theta2=0, linewidth=lw, color=color, fill=False)
    # 将arc添加进ax
    ax.add_patch(center_outer_arc)

    # 绘制中场处的内半圆,半径为20
    center_inner_arc = Arc(xy=(0, 422.5), width=40, height=40, theta1=180,
                           theta2=0, linewidth=lw, color=color, fill=False)
    # 将arc添加进ax
    ax.add_patch(center_inner_arc)

    # 绘制篮球场外框线,尺寸为(500,470)
    lines_outer_rec = Rectangle(xy=(-250, -47.5), width=500, height=470,
                                linewidth=lw, color=color, fill=False)
    # 将rec添加进ax
    ax.add_patch(lines_outer_rec)

    return ax


axs = draw_ball_field(color='#20458C', lw=2)

'''
    -------------
    绘制投篮热点图
    -------------
'''

# 分类数据
df1 = df[df['type'] == 'Made Shot']
df2 = df[df['type'] == 'Missed Shot']
# 绘制散点图
axs.scatter(x=df2['width'], y=df2['height'], s=30,
            marker='x', color='#A82B2B')
axs.scatter(x=df1['width'], y=df1['height'], s=30, marker='o',
            edgecolors='#3A7711', color="#F0F0F0", linewidths=2)

plt.show()

'''
    -------------
    绘制投篮热力图
    -------------
'''

def colormap():
    """
    颜色转换
    """
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#C5C5C5',
                                                        '#9F9F9F', '#706A7C',
                                                        '#675678', '#713A71',
                                                        '#9D3E5E', '#BC5245',
                                                        '#C86138', '#C96239',
                                                        '#D37636', '#D67F39',
                                                        '#DA8C3E', '#E1A352'],
                                                        256)


# 绘制球员投篮热力图
shot_heatmap = sns.jointplot(df['width'], df['height'], stat_func=None,
                             kind='kde', space=0, color='w', cmap=colormap())
# 设置图像大小
shot_heatmap.fig.set_size_inches(6, 6)
# 图像反向
ax = shot_heatmap.ax_joint
# 绘制投篮散点图
ax.scatter(x=df['width'], y=df['height'], s=0.1, marker='o', color="w",
           alpha=1)
# 添加篮球场
draw_ball_field(color='w', lw=2)
# 将坐标轴颜色更改为白色
lines = plt.gca()
lines.spines['top'].set_color('none')
lines.spines['left'].set_color('none')
# 去除坐标轴标签
ax.axis('off')


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import keys
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
browser = webdriver.Chrome()
#try:
browser.get('https://www.baidu.com')
input = browser.find_element_by_id('kw')
input.send_keys('灰汤')
input.send_keys(Keys.ENTER)
wait = WebDriverWait(browser, 10)
wait.until(EC.presence_of_element_located((By.ID, 'content_left')))
print(browser.current_url)
    #print(browser.get_cookies())
    #print(browser.page_source)
#finally:
#    browser.close()