import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# 1. 读取数据
try:
    df = pd.read_csv('nba_locations.csv')
    print("成功读取 nba_locations.csv")
except FileNotFoundError:
    print("错误：未找到文件。请确保 nba_locations.csv 在当前目录下。")
    exit()

# 2. 定义 Haversine 距离公式 (计算地球表面两点间距离)
def haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine 公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # 地球半径 (英里)
    return round(c * r, 2) # 保留两位小数

# 3. 初始化矩阵
teams = df['Team_ID'].tolist()
n = len(teams)
dist_matrix = np.zeros((n, n))

# 4. 计算距离
print("正在计算距离矩阵...")
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i][j] = haversine(
                df.iloc[i]['Longitude'], df.iloc[i]['Latitude'],
                df.iloc[j]['Longitude'], df.iloc[j]['Latitude']
            )

# 5. 保存为 CSV (带表头和索引)
dist_df = pd.DataFrame(dist_matrix, index=teams, columns=teams)
dist_df.to_csv('nba_distance_matrix.csv')

print("\n计算完成！")
print("文件已保存为: nba_distance_matrix.csv")
print("\n预览 (前 5x5):")
print(dist_df.iloc[:5, :5])