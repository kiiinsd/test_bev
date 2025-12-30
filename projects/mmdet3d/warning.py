from enum import Enum

class Risk(Enum):
    NO_RISK = 0
    LOW_RISK = 1
    HIGH_RISK = 2

def warning(
    x, y, #目标车与自车的相对位置
    vx, vy, #目标车与自车的相对速度
    direction, #目标车行驶方向，0-同向，1-相向
    tp, #预留给驾驶员的反应时间
    warning_x, warning_y #碰撞预警距离，根据自车与目标车尺寸计算得到
):
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



        

