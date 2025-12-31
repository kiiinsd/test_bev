from enum import Enum
from typing import Tuple, List
import numpy as np

class Risk(Enum):
    NO_RISK = 0
    LOW_RISK = 1
    HIGH_RISK = 2

def warning(
    pos: Tuple[float, float], #目标车与自车的相对位置
    vel: Tuple[float, float], #目标车与自车的相对速度
    lines: np.ndarray, #车道中心线
    tp: float, #预留给驾驶员的反应时间
    width: float, #车道宽度
    warn_dist : Tuple[float, float] #碰撞预警距离，根据自车与目标车尺寸计算得到
):
    point_dist = np.sqrt(np.sum(lines[0][0]-lines[0][1])**2)
    line_id = -1
    for i, line in enumerate(lines):
        line = line[:,:2]
        dist = np.sum((line-pos)**2, axis=1)
        point_idx = np.argmin(dist)
        nearest_dist = np.sqrt(dist[point_idx])
        if nearest_dist <= width / 2:
            line_id = i
            break
    
    

    if line_id == -1:
        return Risk.NO_RISK
    
    if 

    if abs(x) < warning_x: #目标在x方向预警范围内
        if y * vy < 0: #目标在y方向接近自车
            ty = abs((y-warning_y)/vy) #预计发生碰撞的时间
            if ty <= tp/2:
                return Risk.HIGH_RISK
            elif ty > tp/2 and ty <= tp:
                return Risk.LOW_RISK
            else:
                return Risk.NO_RISK
        else: #目标在y方向远离自车
            return Risk.NO_RISK
    
    else: #目标在x方向预警范围外
        if direction == 0:  #目标与自车同向行驶
            if x * vx < 0 and y * vy < 0: #目标在x,y方向均接近自车
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
        else: #目标与自车相向行驶
            if x * vx < 0: #目标在x方向接近自车
                tx = tx = abs((x-warning_x)/vx)
                if tx <= tp/2:
                    return Risk.HIGH_RISK
                elif tx > tp/2 and tx <= tp:
                    return Risk.LOW_RISK
                else:
                    return Risk.NO_RISK
            else:
                return Risk.NO_RISK



        

