import numpy as np

# 

# 요일 사이클릭 피처 (1=일요일, 7=토요일)
def create_dow_cyclic(day_of_week):
    # 일요일을 시작점(0)으로 하는 사이클
    angle = 2 * np.pi * (day_of_week - 1) / 7
    return np.sin(angle), np.cos(angle)

# 시간 사이클릭 피처
def create_hour_cyclic(hour):
    angle = 2 * np.pi * hour / 24
    return np.sin(angle), np.cos(angle)

# 복합 사이클릭 피처 (주간 패턴 고려)
def create_week_hour_cyclic(day_of_week, hour):
    # 168시간 주기 (7일 * 24시간)
    total_hours = (day_of_week - 1) * 24 + hour
    angle = 2 * np.pi * total_hours / 168
    return np.sin(angle), np.cos(angle)