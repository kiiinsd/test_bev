from enum import Enum
from typing import Tuple, List
import numpy as np
from torch import Tensor

class Risk(Enum):
    NO_RISK = 0
    LOW_RISK = 1
    HIGH_RISK = 2

def calculate_circle_radius(x1, y1, x2, y2, x3, y3):
    """
    给定三个点的坐标，计算这三个点确定的圆的半径
    
    参数:
    x1, y1: 第一个点的坐标
    x2, y2: 第二个点的坐标
    x3, y3: 第三个点的坐标
    
    返回:
    圆的半径，如果三点共线则返回None
    """
    
    # 计算三边长度
    a = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)  # 边23的长度
    b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)  # 边13的长度
    c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)  # 边12的长度
    
    # 检查三点是否共线
    # 使用向量叉积判断：如果三点共线，则叉积为0
    cross_product = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(cross_product) < 1e-10:  # 考虑浮点数精度问题
        # print("错误：三点共线，无法确定圆")
        return None
    
    # 使用海伦公式计算三角形面积
    s = (a + b + c) / 2  # 半周长
    try:
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    except ValueError:
        # print("错误：无法构成三角形")
        return None
    
    # 使用公式 R = (a * b * c) / (4 * S) 计算外接圆半径
    radius = (a * b * c) / (4 * area)
    
    return radius

def warning(
    pos: Tensor, #目标车与自车的相对位置
    vel: Tensor, #目标车与自车的相对速度
    ego_vel: float,
    ego_idx: int,
    lines: np.ndarray, #车道中心线
    tp: float, #预留给驾驶员的反应时间
    width: float, #车道宽度
    warn_dist : Tuple[float, float] #碰撞预警距离，根据自车与目标车尺寸计算得到
):
    x, y = pos.numpy()
    vx, vy = vel.numpy()
    warning_x, warning_y = warn_dist
    point_dist = np.sqrt(np.sum((lines[0][0]-lines[0][1])**2))
    line_id = -1
    for i, line in enumerate(lines):
        line = line[:,:2]
        dist = np.sum((line-pos.numpy())**2, axis=1)
        point_idx = np.argmin(dist)
        nearest_dist = np.sqrt(dist[point_idx])
        if nearest_dist <= width / 2:
            line_id = i
            break
    x1, y1, _, _ = lines[0][ego_idx]
    x2, y2, _, _ = lines[0][ego_idx+1]
    x3, y3, _, _ = lines[0][ego_idx+2]
    radius = calculate_circle_radius(x1, y1, x2, y2, x3, y3)

    if radius == None:
        vy = vy - ego_vel
        if line_id == -1:
            if x * vx < 0:
                if y * vy >= 0:
                    return Risk.NO_RISK
                else:
                    tx = abs((x-warning_x)/vx)
                    ty = abs((y-warning_y)/vy)
                    if tx <= tp/2 and ty <= tp/2:
                        return Risk.HIGH_RISK
                    elif tx > tp and ty > tp:
                        return Risk.NO_RISK
                    else:
                        return Risk.LOW_RISK
            else:
                return Risk.NO_RISK
        else:
            if y * vy < 0:
                ty = abs((y-warning_y)/vy) #预计发生碰撞的时间
                if ty <= tp/2:
                    return Risk.HIGH_RISK
                elif ty > tp/2 and ty <= tp:
                    return Risk.LOW_RISK
                else:
                    return Risk.NO_RISK
            else:
                return Risk.NO_RISK

    else:
        if line_id == -1: #在当前车道之外
            return Risk.NO_RISK

        else: #在当前车道内
            y = (ego_idx - point_idx) * point_dist
            v = np.sqrt(np.sum(vel**2))
            if y * v < 0: #目标在y方向接近自车
                ty = abs((y-warning_y)/v) #预计发生碰撞的时间
                if ty <= tp/2:
                    return Risk.HIGH_RISK
                elif ty > tp/2 and ty <= tp:
                    return Risk.LOW_RISK
                else:
                    return Risk.NO_RISK
            else: #目标在y方向远离自车
                return Risk.NO_RISK



        

