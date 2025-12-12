import pandas as pd
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局参数设置
# ==========================================
NUM_TEAMS = 10
DAYS = 40  # 比赛日总数
TEAMS_INDICES = list(range(NUM_TEAMS)) # [0, 1, ..., 9]

# ==========================================
# 1. 定义选定的球队 & 加载数据
# ==========================================
# 西部 (索引 0-4)
west_teams = ['LAL', 'GSW', 'HOU', 'PHX', 'SAS']
# 东部 (索引 5-9)
east_teams = ['MIL', 'NYK', 'MIA', 'CLE', 'BOS']

# 合并列表 (索引顺序很重要: 0-4为西部, 5-9为东部)
selected_teams_ids = west_teams + east_teams
print(f"选定的 10 支球队: {selected_teams_ids}")

# 创建索引映射 (方便打印结果时看是谁)
# id_map: {0: 'LAL', 1: 'GSW', ...}
id_map = {i: name for i, name in enumerate(selected_teams_ids)}

# --- 读取并筛选距离矩阵 ---
dist_matrix = np.zeros((NUM_TEAMS, NUM_TEAMS)) # 初始化

try:
    full_df = pd.read_csv('nba_distance_matrix.csv', index_col=0)
    
    # 核心步骤：使用 .loc 同时筛选行和列，并按 selected_teams_ids 排序
    sub_df = full_df.loc[selected_teams_ids, selected_teams_ids]
    
    # 转换为 NumPy 数组 (供算法使用)
    dist_matrix = sub_df.to_numpy()
    
    print(f"成功加载距离矩阵，形状: {dist_matrix.shape}")
    print("\n距离矩阵预览 (前 2x2):")
    print(sub_df.iloc[:2, :2])

except FileNotFoundError:
    print("错误：找不到 'nba_distance_matrix.csv'。将使用随机矩阵进行测试。")
    dist_matrix = np.random.randint(100, 3000, size=(10, 10))
    np.fill_diagonal(dist_matrix, 0)
except KeyError as e:
    print(f"错误：CSV 中找不到某些球队 ID。请检查 CSV 表头。缺失的 ID: {e}")
    exit()

# ==========================================
# 2. 生成比赛列表 (Match Generation)
# ==========================================
# 规则：同部打 3 场，异部打 2 场
all_required_matches = []

print("\n正在生成赛季比赛安排...")
for i in TEAMS_INDICES:
    for j in range(i + 1, NUM_TEAMS): # 只遍历 j > i，避免重复
        # 判断是否同区
        # 索引 < 5 为西部，>= 5 为东部
        i_is_west = (i < 5)
        j_is_west = (j < 5)
        
        is_same_conf = (i_is_west == j_is_west)
        num_games = 3 if is_same_conf else 2
        
        # 简单分配主客场：交替做主场
        # k=0: i主场, k=1: j主场, k=2: i主场...
        for k in range(num_games):
            if k % 2 == 0:
                all_required_matches.append((i, j)) # i 主场 vs j
            else:
                all_required_matches.append((j, i)) # j 主场 vs i

print(f"总共需要安排的比赛场次: {len(all_required_matches)}")

# ==========================================
# 3. 核心成本计算函数
# ==========================================
def calculate_schedule_cost(schedule_matrix):
    """
    计算给定赛程表的总旅行距离。
    schedule_matrix[t] 是一个列表，包含当天所有的比赛元组 [(home, away), ...]
    """
    total_dist = 0
    
    # 记录每支球队当前位置，初始都在自己的主场 (索引即城市ID)
    current_locations = {i: i for i in TEAMS_INDICES}
    
    num_days = len(schedule_matrix)
    
    for t in range(num_days):
        # 1. 确定当天每支球队的目标位置
        day_targets = {} 
        
        matches_today = schedule_matrix[t]
        playing_teams = set()
        
        for home, away in matches_today:
            # 主队在自家
            day_targets[home] = home
            playing_teams.add(home)
            # 客队去主队家
            day_targets[away] = home
            playing_teams.add(away)
            
        # 对于没比赛的球队，位置保持不变 (Target = Current)
        # 这是题目要求的关键逻辑：无比赛日不移动
        for team in TEAMS_INDICES:
            if team not in playing_teams:
                day_targets[team] = current_locations[team]
        
        # 2. 计算移动距离并更新位置
        for team in TEAMS_INDICES:
            start = current_locations[team]
            end = day_targets[team]
            
            if start != end:
                # 只有位置改变才产生距离
                total_dist += dist_matrix[start][end]
            
            # 更新位置状态
            current_locations[team] = end
            
        for team in TEAMS_INDICES:
            final_loc = current_locations[team]
            home_base = team  # 在我们的设定中，球队 i 的主场就是城市 i
            
            if final_loc != home_base:
                total_dist += dist_matrix[final_loc][home_base]
                # print(f"Season End: Team {team} flies back home ({final_loc} -> {home_base})")

    return total_dist

# ==========================================
# 4. 模拟退火算法类 (Simulated Annealing)
# ==========================================
class SimulatedAnnealingSolver:
    def __init__(self, teams_list, matches, days, dist_mat):
        self.teams = teams_list
        self.matches = matches  # 所有需要打的比赛 [(h, a), ...]
        self.days = days
        self.dist_matrix = dist_mat
        
    def generate_initial_solution(self):
        """
        生成初始解：随机将比赛填入 40 天，尽量保证不冲突。
        """
        schedule = [[] for _ in range(self.days)]
        # 复制一份比赛列表并打乱
        matches_pool = self.matches.copy()
        random.shuffle(matches_pool)
        
        # 贪心填充
        for match in matches_pool:
            h, a = match
            assigned = False
            
            # 尝试找到能塞进去的一天
            for day in range(self.days):
                # 检查当天这两队是否有空
                busy_teams = set()
                for existing_match in schedule[day]:
                    busy_teams.add(existing_match[0])
                    busy_teams.add(existing_match[1])
                
                if h not in busy_teams and a not in busy_teams:
                    schedule[day].append(match)
                    assigned = True
                    break
            
            if not assigned:
                # 如果40天都满了填不进去（极其罕见情况），强行塞入最后一天
                # 实际应用中应该有惩罚机制，这里简化处理
                schedule[self.days-1].append(match)
                
        return schedule

    def get_neighbor(self, schedule):
        """
        产生邻域解：
        1. 交换两天 (Swap Days)
        2. 移动一场比赛 (Move Game)
        """
        new_schedule = copy.deepcopy(schedule)
        move_type = random.random()
        
        if move_type < 0.5:
            # --- 动作 A: 交换两天 ---
            d1, d2 = random.sample(range(self.days), 2)
            new_schedule[d1], new_schedule[d2] = new_schedule[d2], new_schedule[d1]
            
        else:
            # --- 动作 B: 移动一场比赛 ---
            # 1. 找一个有比赛的天 (Source)
            days_with_games = [d for d in range(self.days) if len(new_schedule[d]) > 0]
            if not days_with_games: return new_schedule
            
            src_day = random.choice(days_with_games)
            # 取出一场比赛
            match_idx = random.randint(0, len(new_schedule[src_day])-1)
            match = new_schedule[src_day].pop(match_idx)
            
            # 2. 找一个目标天 (Dest)
            dest_day = random.randint(0, self.days - 1)
            
            # 3. 检查目标天是否有冲突
            h, a = match
            busy = False
            for m in new_schedule[dest_day]:
                if h in m or a in m:
                    busy = True
                    break
            
            if not busy:
                new_schedule[dest_day].append(match)
            else:
                # 如果冲突，放回原处（取消这次移动）
                new_schedule[src_day].append(match)
                
        return new_schedule

    def solve(self, initial_temp=5000, cooling_rate=0.995, max_iter=10000):
        # 1. 初始化
        current_schedule = self.generate_initial_solution()
        current_cost = calculate_schedule_cost(current_schedule)
        
        best_schedule = copy.deepcopy(current_schedule)
        best_cost = current_cost
        
        temp = initial_temp
        costs_history = []
        
        print(f"初始随机解的总距离: {int(current_cost)}")
        
        # 2. 退火循环
        for i in range(max_iter):
            # 生成新解
            neighbor_schedule = self.get_neighbor(current_schedule)
            neighbor_cost = calculate_schedule_cost(neighbor_schedule)
            
            # 计算能量差
            delta = neighbor_cost - current_cost
            
            # 接受准则
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_schedule = neighbor_schedule
                current_cost = neighbor_cost
                
                # 更新全局最优
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_schedule = copy.deepcopy(current_schedule)
                    if i % 100 == 0: # 每100次打印一次，避免刷屏
                        print(f"Iter {i}: Found new best cost = {int(best_cost)}")
            
            # 降温
            temp *= cooling_rate
            costs_history.append(current_cost)
            
            if temp < 1e-6: break
            
        return best_schedule, best_cost, costs_history

# ==========================================
# 5. 主程序运行
# ==========================================
if __name__ == "__main__":
    print("\n------ 开始运行模拟退火 (SA) ------")
    
    # 实例化求解器
    sa_solver = SimulatedAnnealingSolver(TEAMS_INDICES, all_required_matches, DAYS, dist_matrix)
    
    # 运行求解
    final_schedule, final_cost, history = sa_solver.solve(max_iter=50000)

    print(f"\n最终优化结果 - 总距离: {int(final_cost)}")

    # --- 结果可视化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('Simulated Annealing Convergence (Total Travel Distance)')
    plt.xlabel('Iteration')
    plt.ylabel('Miles')
    plt.grid(True)
    plt.show()

    # --- 打印前5天的赛程示例 ---
    print("\n赛程示例 (前 5 天):")
    for t in range(5):
        day_str = f"Day {t+1}: "
        matches = final_schedule[t]
        if not matches:
            day_str += "No Games"
        else:
            # 将 ID 转换为 球队名 打印
            match_strs = [f"{id_map[h]} vs {id_map[a]}" for h, a in matches]
            day_str += ", ".join(match_strs)
        print(day_str)

    # ==========================================
    # 6. 保存赛程到 CSV (新增部分)
    # ==========================================
    print("\n正在保存赛程到 nba_schedule_sa.csv ...")
    
    schedule_data = []

    # 遍历每一天的赛程
    for t, matches in enumerate(final_schedule):
        day_num = t + 1 # 天数从1开始
        
        # 遍历当天的每一场比赛
        for home_id, away_id in matches:
            schedule_data.append({
                'Day': day_num,
                'Home_Team': id_map[home_id], # 将ID转为名字
                'Away_Team': id_map[away_id]
            })

    # 创建 DataFrame
    df_schedule = pd.DataFrame(schedule_data)

    # 保存为 CSV (不包含索引列)
    df_schedule.to_csv('nba_schedule_sa.csv', index=False)

    print("保存成功！预览前 5 行数据：")
    print(df_schedule.head())