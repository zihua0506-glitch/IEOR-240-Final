import pandas as pd
from io import StringIO
import itertools
import random
import csv

# --- 1. 数据准备 (使用提供的距离矩阵) ---

TEAMS_W = ['LAL', 'GSW', 'SAS', 'HOU', 'OKC']
TEAMS_E = ['MIL', 'MIA', 'BOS', 'NYK', 'CLE']
ALL_TEAMS = TEAMS_W + TEAMS_E
N_DAYS = 40
N_TEAMS = len(ALL_TEAMS)

# 提供的距离矩阵 (km)
csv_distance_content = """Team_ID,LAL,GSW,SAS,HOU,OKC,MIL,MIA,BOS,NYK,CLE
LAL,0,560,2020,2500,1890,3400,4300,4200,3900,3200
GSW,560,0,2120,2800,2100,3600,4500,4700,4400,3800
SAS,2020,2120,0,320,810,2400,2100,3000,2700,2100
HOU,2500,2800,320,0,700,2000,1600,2600,2300,1700
OKC,1890,2100,810,700,0,1700,2300,2800,2500,1400
MIL,3400,3600,2400,2000,1700,0,1900,1500,1300,500
MIA,4300,4500,2100,1600,2300,1900,0,2100,1800,1800
BOS,4200,4700,3000,2600,2800,1500,2100,0,300,900
NYK,3900,4400,2700,2300,2500,1300,1800,300,0,700
CLE,3200,3800,2100,1700,1400,500,1800,900,700,0"""

df_dist = pd.read_csv(StringIO(csv_distance_content)).set_index('Team_ID')
DISTANCES = df_dist.to_dict(orient='index')


def get_dist(u, v):
    """从字典中获取城市 u 到 v 的距离"""
    return DISTANCES.get(u, {}).get(v, 0.0)


# 比赛场次需求 (22 场)
GAMES_CONF = 3
GAMES_CROSS = 2
# ... (GAMES_TOTAL calculation is correct) ...

# --- 2. 初始化状态 ---

schedule = {team: ['---'] * N_DAYS for team in ALL_TEAMS}
games_needed = {}
for i in ALL_TEAMS:
    games_needed[i] = {j: 0 for j in ALL_TEAMS if i != j}

# 计算初始所需的比赛场次 (Home vs Away)
for i in ALL_TEAMS:
    for j in ALL_TEAMS:
        if i == j: continue

        if (i in TEAMS_W and j in TEAMS_E) or (i in TEAMS_E and j in TEAMS_W):
            games_needed[j][i] = 1
            games_needed[i][j] = 1

        elif (i in TEAMS_W and j in TEAMS_W) or (i in TEAMS_E and j in TEAMS_E):
            if i < j:
                games_needed[j][i] = 2
                games_needed[i][j] = 1

            # 球队位置追踪 (第 0 天在主场)
location = {team: team for team in ALL_TEAMS}
total_distance = 0.0

# --- 3. 贪婪启发式核心 ---

total_games_required = sum(games_needed[i][j] for i in ALL_TEAMS for j in ALL_TEAMS if i != j) // 2
print(f"总共需要安排 {total_games_required} 场比赛 (即 220 场).")

for t in range(N_DAYS):
    current_day = t + 1

    all_game_pairs = []
    for i in ALL_TEAMS:  # 主队
        for j in ALL_TEAMS:  # 客队
            if i == j or games_needed[i][j] == 0:
                continue
            all_game_pairs.append((i, j))

    # 2. 评估每场比赛的旅行成本 (包含赛后返回主场的距离)
    game_costs = []
    for home_team, away_team in all_game_pairs:
        game_city = home_team  # 比赛城市
        away_home_city = away_team  # 客队自己的主场

        # 1. 客队旅行成本:
        # L_{t-1} -> game_city (去程)
        travel_to_game_away = get_dist(location[away_team], game_city)
        # game_city -> away_home_city (回程)
        travel_return_away = get_dist(game_city, away_home_city)
        away_travel = travel_to_game_away + travel_return_away  # 包含回程成本

        # 2. 主队旅行成本:
        # L_{t-1} -> game_city (去程). 赛后已在主场，无回程成本
        home_travel = get_dist(location[home_team], game_city)

        total_cost = away_travel + home_travel

        # 优先级：鼓励安排跨部比赛 (长途旅行)
        priority_score = 1
        if (home_team in TEAMS_W and away_team in TEAMS_E) or (home_team in TEAMS_E and away_team in TEAMS_W):
            priority_score = 0.5

        score = total_cost / priority_score

        game_costs.append({'cost': total_cost, 'score': score, 'home': home_team, 'away': away_team})

    # 3. 贪婪选择: 按分数排序
    game_costs.sort(key=lambda x: x['score'])

    # 4. 安排当日比赛
    games_today = []
    teams_playing_today = set()

    for game in game_costs:
        home, away = game['home'], game['away']

        if home not in teams_playing_today and away not in teams_playing_today:
            if games_needed[home][away] > 0:
                games_today.append(game)
                teams_playing_today.add(home)
                teams_playing_today.add(away)

            if len(games_today) >= N_TEAMS // 2:
                break

    # 5. 更新状态 (赛程、比赛计数、位置跟踪)

    next_day_location = location.copy()

    for game in games_today:
        home, away = game['home'], game['away']

        games_needed[home][away] -= 1

        schedule[home][t] = f'vs {away}'
        schedule[away][t] = f'@ {home}'

        total_distance += game['cost']

        # *** 新规则：所有比赛队伍赛后回到自己的主场 ***
        next_day_location[away] = away  # 客队回到自己的主场 (Team ID)
        next_day_location[home] = home  # 主队回到自己的主场 (即比赛地点)

    # 6. 处理休息日球队的位置 (Stay-Put 约束)
    for team in ALL_TEAMS:
        if team not in teams_playing_today:
            # 休息的球队，位置保持不变 (L_{t} = L_{t-1})
            next_day_location[team] = location[team]

    location = next_day_location

# --- 7. 最终结果和检查 ---

games_left = sum(games_needed[i][j] for i in ALL_TEAMS for j in ALL_TEAMS if i != j)

if games_left == 0:
    feasibility = "可行解 (所有 220 场比赛已安排并满足所有 H/A 约束)"
else:
    feasibility = f"次优解 (由于贪婪选择限制，仍有 {games_left // 2} 场比赛未安排。贪婪启发式未能找到完全满足约束的解)"

print("\n--- 贪婪启发式求解结果 (包含赛后返回主场距离) ---")
print(f"求解方法: 纯 Python 贪婪启发式 (Greedy Heuristic)")
print(f"结果: {feasibility}")
print(f"总奔波距离 (最小化目标): {total_distance:.2f} km")
print(f"平均每日移动成本: {total_distance / N_DAYS:.2f} km/天")

# --- 8. 赛程表展示 ---

header = ["Team"] + [f"D{t + 1}" for t in range(40)]
print("\n--- 赛程表 (Scheduling Table - 前 10 天) ---")
print("| " + " | ".join(header) + " |")
print("|" + "---|" * (len(header)))

for team in ALL_TEAMS:
    row = [team.ljust(6)] + schedule[team][:40]
    print("| " + " | ".join(row) + " |")