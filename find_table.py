import math
import numpy as np


'''
获取距离
'''
def cal_distance(staging_arg, robot_arg):
    x_d = staging_arg[1] - robot_arg[-2]
    y_d = staging_arg[2] - robot_arg[-1]
    distance = math.sqrt(x_d**2 + y_d**2)
    return distance


def cal_best_table(staging_args, robot_arg):
    dic = {}
    for i in range(len(staging_args)):
        staging = staging_args[i]
        distance = cal_distance(staging, robot_arg)
        if robot_arg[1] != 0:
            dic[i] = value[int(robot_arg[1])] / distance
        else:
            dic[i] = value[int(staging[0])] / distance
    if not dic:
        return -1, -1
    # 排序[(序号，距离),..]
    dic = sorted(dic.items(), key=lambda x: x[1])
    best_table_idx = dic[-1][0]
    dis = dic[-1][1]
    return best_table_idx, dis


'''
计算距离各robot最近的收货点，去掉还在恢复的收获点
并返回最近收获点的序号
'''
def cal_robot_to_jinhuo_most_neighbour(staging_args, robot_arg):
    # [序号,x,y,生产剩余时间，原料格状态，产品格状态]
    # [所处工作台,携带物品,时间价值系数,碰撞价值系数, w, vx, vy, yaw, x, y]
    dic = {}
    # 判断机器人是否带有货物
    #assert robot_arg[1] == 0, "机器人带有货物，不能购买货物"
    # 计算各工作台路径(1-7均可以购买货物)
    for i in range(len(staging_args)):
        staging = staging_args[i]
        # 判断是否工作台是否有货
        if staging[-1] == 0:
            continue
        distance = cal_distance(staging, robot_arg)
        dic[i] = value[int(staging[0])]/distance
    if not dic:
        return -1, -1
    # 排序[(序号，距离),..]
    dic = sorted(dic.items(), key=lambda x: x[1],)
    most_neighbour_jinhuo = dic[-1][0]
    return most_neighbour_jinhuo, dic[-1][1]


goods_table = {1: [4, 5, 9], 2: [4, 6, 9], 3: [5, 6, 9], 4: [7, 9], 5: [7, 9], 6: [7, 9], 7: [8, 9]}
table_good = {4: [1, 2], 5: [1, 3], 6: [2, 3], 7: [4, 5, 6], 8: [7], 9: [1,2,3,4,5,6,7]}
value = {1: 3000, 2: 3200, 3: 3400, 4: 7100, 5: 7800, 6: 8300, 7: 29000}


def cal_jinhuo_to_chuhuo_most_neighbour(staging_args, robot_arg):
    dic = {}
    for i in range(len(staging_args)):
        staging = staging_args[i]
        # 判断是否可以出售
        if staging[0] in [1,2,3]:  # 123不收购
            continue
        if staging[0] in goods_table[robot_arg[1]]:  # 判断当前工作台是否是可以出售的
            # 获取可出售工作台的二进制状态
            if staging[-2] == 0:  # 原料格为0 直接卖
                distance = cal_distance(staging, robot_arg)
                dic[i] = value[int(robot_arg[1])]/distance
                continue
            zt_s_bin = bin(int(staging[-2]))[2:]  # 换成2进制 1000
            zt_s = zt_s_bin[::-1]  # 倒序
            if robot_arg[1]+1>len(zt_s):
                distance = cal_distance(staging, robot_arg)
                dic[i] = value[int(robot_arg[1])]/distance
                continue
            if robot_arg[1]+1<=len(zt_s) and zt_s[int(robot_arg[1])] == '0':
                distance = cal_distance(staging, robot_arg)
                dic[i] = value[int(robot_arg[1])]/distance

    if not dic:
        return -1, -1
    # 排序[(序号，距离),..]
    dic = sorted(dic.items(), key=lambda x: x[1])
    most_neighbour_chuhuo = dic[-1][0]

    return most_neighbour_chuhuo, dic[-1][1]


def needs_goods(staging_args, robot_args):
    '''

    :param staging_args:[序号,x,y,生产剩余时间，原料格状态，产品格状态]
    :param robot_args: [所处工作台索引,携带物品,时间价值系数,碰撞价值系数, w, vx, vy, yaw, x, y]
    :return: need 工作台需求量  good 机器人携带的量
    '''

    a = np.zeros((7,))
    b = np.arange(1,8)  # 7,6,5,4,3,2,1 可供应的货物品种
    c = np.arange(4,10)  # 4，5，6，7，8，9 可以出售工作台号
    d = np.zeros((6,))
    need = dict(zip(b, a))  # 需求量
    good = dict(zip(b, a))  # 机器人当下拥有量
    table = dict(zip(c,d))# 可售货台
    for staging in staging_args:
        id = int(staging[0])
        # if staging[-1]==1:
        #     good[id] += 1 #拥有量+1
        if id in [4,5,6,7,8,9]:
            std = table_good[id] #标准状态
            now_bin = bin(int(staging[-2]))[2:]
            now_bin = now_bin[::-1]
            now = [i for i, j in enumerate(now_bin) if j == '1']
            if len(now) ==0:
                for i in std:
                    need[i] += 1
                    table[id] += 1
            elif std == now:
                continue
            else:
                std_now = [i for i in std if i not in now]  # 补集
                for i in std_now:
                    need[i] += 1
                    table[id] += 1
    table[8] = 1000
    table[9] = 1000
    for robot_arg in robot_args:
        i = robot_arg[1]
        if i:
            good[i] += 1
            need[i] -= 1
    return need, good, table


def find5(staging_args, robot_args):
    '''

    :param staging_args: [序号,x,y,生产剩余时间，原料格状态，产品格状态]
    :param robot_args: [所处工作台索引,携带物品,时间价值系数,碰撞价值系数, w, vx, vy, yaw, x, y]
    :return:
    '''
    staging_args1 = np.copy(staging_args)
    s = np.zeros((4, 6))  # 返回工作台信息
    r = np.array([False, False, False, False])
    needs, goods, tables = needs_goods(staging_args1,robot_args)

    while not np.all(r):
        sell_xuqiu = [i for i in range(4,10) if tables[i] > 0]  # 卖的时候，只看需求
        buy_xuqiu = [i for i in range(1,8) if needs[i] > 0]  # 买的时候，看供需，需求大于供给
        buy_di = {}
        sell_di = {}
        for i, j in enumerate(r):
            if j:
                continue
            if robot_args[i][1] != 0:  # 判断是否携带货物,先判断卖
                staging_args_sell = [si for si in staging_args1 if si[0] in sell_xuqiu]  # 获取可卖工作台
                # staging_args_sell = [si for si in staging_args1 if si[0] in [4,5,6,7,8,9]]
                idx0, dis0 = cal_jinhuo_to_chuhuo_most_neighbour(staging_args_sell, robot_args[i])
                sell_di[i] = [idx0, dis0]
            else:
                staging_args_buy = [si for si in staging_args1 if si[0] in buy_xuqiu]
                idx0, dis0 = cal_robot_to_jinhuo_most_neighbour(staging_args_buy, robot_args[i])
                buy_di[i] = [idx0, dis0]
        # 买
        if len(buy_di) != 0:
            buy_di_ = sorted(buy_di.items(), key=lambda x: x[1][1], reverse=True)  # 排序[id:[序号，距离],..]
            id = int(buy_di_[0][0])
            idx = int(buy_di_[0][1][0])
            r[id] = True

            if idx == -1:
                s[id] = -1
            else:
                s[id] = staging_args_buy[idx]
                id1 = int(staging_args_buy[idx][0]) #目标类型供、需-1
                needs[id1] -= 1
                goods[id1] -= 1
            for i in range(len(staging_args1)):
                if np.all(s[id] == staging_args1[i]):
                    staging_args1 = np.delete(staging_args1, i, axis=0)
                    break

        # 卖
        if len(sell_di) != 0:
            sell_di_ = sorted(sell_di.items(), key=lambda x: x[1][1], reverse=True)
            id = int(sell_di_[0][0])
            idx = int(sell_di_[0][1][0])
            r[id] = True
            if idx == -1:
                s[id] = -1
            else:
                s[id] = staging_args_sell[idx]
                id1 = int(robot_args[id][1])  # 目标类型需-1
                id2 = int(staging_args_sell[idx][0])  # 工作台需 -1
                needs[id1] -= 1
                tables[id2] -= 1
                if tables[id2] == 0:
                    staging_args1 = np.delete(staging_args1, idx, axis=0)
    return s


def buy_sell(staging_arg, robot_arg):
    # robot_arg [所处工作台,携带物品,时间价值系数,碰撞价值系数, w, vx, vy, yaw, x, y]
    # staging_arg [序号,x,y,生产剩余时间，原料格状态，产品格状态]
    robot_buy = 0
    robot_sell = 0
    if robot_arg[0] != -1 and cal_distance(staging_arg, robot_arg)<0.4:  # 到达工作台
        if robot_arg[1] != 0:  # 携带物品
            robot_sell += 1
        else:
            robot_buy += 1
    return robot_buy, robot_sell
