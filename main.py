#!/bin/bash
import math
import sys
from time import sleep
import numpy as np
# from run_model import *
from dwa1 import *
from find_table import *
import sys

sys.setrecursionlimit(100000000)
min_sell_distance = 0.4
u0, pred0 = [0,0], []
u1, pred1 = [0,0], []
u2, pred2 = [0,0], []
u3, pred3 = [0,0], []
target_state = np.array([])
buy_flag = False
sell_flag = False
def read_util_ok():
    '''
    获取判题器的状态输入
    :return:
    '''
    aa = []
    while True:
        a = input()
        if a == 'OK':
            break
        aa.append(a)
    return aa


def finish():
    '''
    完成操作回复 OK
    :return:
    '''
    sys.stdout.write('OK\n')
    sys.stdout.flush()



if __name__ == '__main__':
    # 接入进程
    # sleep(10)
    # 初始化 地图数据
    map = read_util_ok()
    finish()

    # 逐帧机器人调控
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.split(' ')
        frame_id = int(parts[0])  # 获取当前帧数
        context = read_util_ok()  # 获取当前状态
        sys.stdout.write('%d\n' % frame_id)

        # 数据转化
        # [序号,x,y,生产剩余时间，原料格状态，产品格状态]
        tables_state = np.array([[float(i) for i in j.split()] for j in context[1:-4]])
        # [所处工作台,携带物品,时间价值系数,碰撞价值系数, w, vx, vy, yaw, x, y]
        robots_state = np.array([[float(i) for i in j.split()] for j in context[-4:]])
        if frame_id >= 230:
            x=2
        if frame_id >= 150:
            x = 2
        if frame_id >= 50 or buy_flag or sell_flag:
            aa, bb, cc = needs_goods(tables_state, robots_state)
            target_state = find5(tables_state, robots_state)


        if len(target_state) == 0 or np.all(target_state[:][0] == -1):
            finish()
            continue
        if target_state[0][0] != -1:
            u0, pred0 = daw_predict(target_state[0], robots_state, 0)
        if target_state[1][0] != -1:
            u1, pred1 = daw_predict(target_state[1], robots_state, 1)
        if target_state[2][0] != -1:
            u2, pred2 = daw_predict(target_state[2], robots_state, 2)
        if target_state[3][0] != -1:
            u3, pred3 = daw_predict(target_state[3], robots_state, 3)

        for i in range(4):
            buy, sell = buy_sell(target_state[i], robots_state[i])
            if buy:  # buy 返回 0 or 1
                sys.stdout.write('buy %d\n' % i)
                buy_flag = True
            if sell:
                sys.stdout.write('sell %d\n' % i)
                sell_flag = True

            # g = robots_state[i][1]
            # if g!=0 and aa[int(g)] == 0:
            #     sys.stdout.write('destroy %d\n' % i)

        for i, u in enumerate([u0, u1, u2, u3]):
            sys.stdout.write('forward %d %f\n' % (i, u[0]))
            sys.stdout.write('rotate %d %f\n' % (i, u[1]))
        finish()