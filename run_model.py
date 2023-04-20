import numpy as np
import math


class Config:
    def __init__(self):
        # define robot move speed ,accelerate,radius ...and so on
        # 定义机器人移动极限速度、加速度等信息
        self.m1 = 20 * math.pi * (0.45 ** 2)
        self.m2 = 20 * math.pi * (0.53 ** 2)

        self.v_min = -2.0  # 最小速度
        self.v_max = 6.0  # 最大速度

        self.w_max = math.pi  # 最大角速度
        self.w_min = -math.pi  # 最小角速度

        self.a_vmax = 250 / self.m1  # 空载加速度
        self.a_vmax2 = 250 / self.m2  # 负载加速度

        self.a_wmax = 50 / (self.m1 * 0.45 ** 2)  # 空载角加速度
        self.a_wmax2 = 50 / (self.m1 * 0.53 ** 2)  # 负载角加速度

        self.robot_radius = 0.45  # 机器人模型半径
        self.robot_radius2 = 0.53  # 机器人满载半径

        self.dt = 1 / 50  # 离散时间间隔
        self.predict_time = 1 / 10  # 模拟轨迹的持续时间

        # 轨迹评价函数系数
        self.alpha = 1.0  # 距离目标点的评价函数的权重系数
        self.beta = 0.5  # 速度评价函数的权重系数
        self.gamma = 0.5  # 距离障碍物距离的评价函数的权重系数

        self.judge_distance = 0.5  # 若与障碍物的最小距离大于阈值（例如这里设置的阈值为robot_radius+0.2）,则设为一个较大的常值
        self.target = np.array([0, 0])
        self.ob = np.array([0, 0])

        self.theta = [-0.05, 0.05]  # 最小角度偏差 弧度


def KinematicModel(state, control, dt):
    """机器人运动学模型
    Args:
        state (_type_): 状态量---x,y,yaw,v,w
        control (_type_): 控制量---v,w,线速度和角速度
        dt (_type_): 离散时间
    Returns:
        _type_: 下一步的状态
    """
    state[0] += control[0] * math.cos(state[2]) * dt
    state[1] += control[0] * math.sin(state[2]) * dt
    state[2] += control[1] * dt
    state[3] = control[0]
    state[4] = control[1]

    return state


class DWA:
    def __init__(self, config, op) -> None:
        """初始化
        Args:
            config (_type_): 参数类
        """
        if op == 0:
            self.radius = config.robot_radius
            self.a_vmax = config.a_vmax
            self.a_wmax = config.a_wmax
        else:
            self.radius = config.robot_radius2
            self.a_vmax = config.a_vmax2
            self.a_wmax = config.a_wmax2

        self.dt = config.dt
        self.v_min = config.v_min
        self.w_min = config.w_min
        self.v_max = config.v_max
        self.w_max = config.w_max
        self.predict_time = config.predict_time

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma

        self.judge_distance = config.judge_distance
        self.theta = config.theta

    def dwa_control(self, state, goal, obstacle):
        """算法入口
        Args:
            state (_type_): 机器人当前状态--[x,y,yaw,v,w]
            goal (_type_): 目标点位置，[x,y]
            obstacle (_type_): 障碍物位置，dim:[num_ob,2]
        Returns:
            _type_: 控制量、轨迹（便于绘画）
        """
        control, trajectory = self.trajectory_evaluation(state, goal, obstacle)
        return control, trajectory

    def __cal_vel_limit(self):
        """计算速度边界限制Vm
        Returns:
            _type_: 速度边界限制后的速度空间Vm
        """
        return [self.v_min, self.v_max, self.w_min, self.w_max]

    def __cal_accel_limit(self, v, w):
        """计算加速度限制Vd
        Args:
            v (_type_): 当前时刻线速度
            w (_type_): 当前时刻角速度
        Returns:
            _type_:考虑加速度时的速度空间Vd
        """
        v_low = v - self.a_vmax * self.dt
        v_high = v + self.a_vmax * self.dt
        w_low = w - self.a_wmax * self.dt
        w_high = w + self.a_wmax * self.dt
        return [v_low, v_high, w_low, w_high]

    def cal_dynamic_window_vel(self, v, w, state, obstacle):
        """速度采样,得到速度空间窗口
        Args:
            v (_type_): 当前时刻线速度
            w (_type_): 当前时刻角速度
            state (_type_): 当前机器人状态
            obstacle (_type_): 障碍物位置
        Returns:
            [v_low,v_high,w_low,w_high]: 最终采样后的速度空间
        """
        Vm = self.__cal_vel_limit()
        Vd = self.__cal_accel_limit(v, w)
        a = max([Vm[0], Vd[0]])
        b = min([Vm[1], Vd[1]])
        c = max([Vm[2], Vd[2]])
        d = min([Vm[3], Vd[3]])
        return [a, b, c, d]

    def trajectory_predict(self, state_init, v, w):
        """轨迹推算
        Args:
            state_init (_type_): 当前状态---x,y,yaw,v,w
            v (_type_): 当前时刻线速度
            w (_type_): 当前时刻线速度
        Returns:
            _type_: _description_
        """
        state = np.array(state_init)
        trajectory = state
        time = 0
        # 在预测时间段内
        while time <= self.predict_time:
            x = KinematicModel(state, [v, w], self.dt)  # 运动学模型
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory

    def cal_turn_yow(self, state_init, goal):
        """转角度计算
        Args:
            state_init (_type_): 当前状态---x,y,yaw,v,w
            goal (_type_): 目标点位置[x,y]
        Returns:
            _type_: _yow_
        """
        dx = goal[0] - state_init[0]
        dy = goal[1] - state_init[1]
        thet1 = np.arctan(dy/dx)
        if dx < 0 and dy > 0:  # 第四象限
            thet1 = math.pi + thet1
        if dx < 0 and dy < 0:  # 第三象限
            thet1 = thet1 - math.pi

        dthet = thet1 - state_init[2]

        if dthet < -math.pi:
            dthet += 2*math.pi

        if dthet > math.pi:
            dthet = dthet - 2*math.pi

        return dthet

    def cal_turn_yow_v2(self, state_init, goal):
        dx = goal[0] - state_init[0]
        dy = goal[1] - state_init[1]
        thet1 = math.atan2(dy, dx)
        dthet = thet1 - state_init[2]
        if dthet < -math.pi:
            dthet += 2*math.pi

        if dthet > math.pi:
            dthet = dthet - 2*math.pi
        return dthet



    def trajectory_evaluation(self, state, goal, obstacle):
        """轨迹评价函数,评价越高，轨迹越优
        Args:
            state (_type_): 当前状态---x,y,yaw,v,w
            goal (_type_): 目标点位置，[x,y]
            obstacle (_type_): 障碍物位置，dim:[num_ob,2]
            dynamic_window_vel (_type_): 可操控速度空间窗口---[v_low,v_high,w_low,w_high]
        Returns:
            _type_: 最优控制量、最优轨迹
        """
        G_max = -float('inf')  # 最优评价
        trajectory_opt = state  # 最优轨迹
        control_opt = [0., 0.]  # 最优控制
        goal_theta = self.cal_turn_yow_v2(state, goal)  # 航向角偏差
        dynamic_window_vel = self.cal_dynamic_window_vel(state[3], state[4], state, obstacle)  # 第1步--计算速度空间
        if goal_theta < self.theta[0] or goal_theta > self.theta[1]:  # 转向时
            v = dynamic_window_vel[0]
            if v < 0.8:
                v = 0.8
            if goal_theta > 0:  # 航向角大于0顺时针转，取最大角速度
                w = dynamic_window_vel[-1]
            else:
                w = dynamic_window_vel[-2]

        else:  # 方向校准之后可以不用角速度调整,直线加速
            w = 0
            v = dynamic_window_vel[1]

        # trajectory_opt = self.trajectory_predict(state, v, w)  # 第2步--轨迹推算
        control_opt = [v, w]

        return control_opt, trajectory_opt


def daw_predict(staging_arg, robot_args, ID):
    """
    模拟DWA过程
    :param staging_arg: 目标工作台状态参数
    :param robot_args: 所有机器人状态参数
    :return: 控制指令
    """
    line_speed = np.hypot(robot_args[ID][-4], robot_args[ID][-5])

    x = np.array([robot_args[ID][-2],robot_args[ID][-1],robot_args[ID][-3],line_speed,robot_args[ID][-6]])
    # u = np.array([np.hypot(x[-2], x[-1]), x[4]])
    config = Config()
    config.target = np.array([staging_arg[1],staging_arg[2]])
    ob = [[-100,-100]]
    # for i in range(1, robot_args.shape[0]):
    #      ob.append([robot_args[i][-2], robot_args[i][-1]])
    config.ob = np.array(ob)
    dwa = DWA(config,robot_args[ID][1])
    u, predicted_trajectory = dwa.dwa_control(x, config.target, config.ob)

    return u, predicted_trajectory

