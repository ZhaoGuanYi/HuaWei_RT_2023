import numpy as np
import math


class Config:
    def __init__(self):
        # define robot move speed ,accelerate,radius ...and so on
        # 定义机器人移动极限速度、加速度等信息
        self.m1 = 20*math.pi*(0.45**2)
        self.m2 = 20*math.pi*(0.53**2)

        self.v_min = -2.0   # 最小速度
        self.v_max = 6.0    # 最大速度

        self.w_max = math.pi   # 最大角速度
        self.w_min = -math.pi   # 最小角速度

        self.a_vmax = 250/self.m1  # 空载加速度
        self.a_vmax2 = 250/self.m2   # 负载加速度

        self.a_wmax = 50/(self.m1*0.45**2)   # 空载角加速度
        self.a_wmax2 = 50/(self.m1*0.53**2)  # 负载角加速度

        self.v_sample = 0.1  # 速度分辨率
        self.w_sample = 5 * math.pi / 180.0  # 角速度分辨率

        self.robot_radius = 0.45  # 机器人模型半径
        self.robot_radius2 = 0.53  # 机器人满载半径

        self.dt = 1/50  # 离散时间间隔
        self.predict_time = 1/10  # 模拟轨迹的持续时间(一帧)

        # 轨迹评价函数系数
        self.alpha = 1.0  # 距离目标点的评价函数的权重系数
        self.beta = 1.0  # 速度评价函数的权重系数
        self.gamma = 3.0  # 距离障碍物距离的评价函数的权重系数

        self.judge_distance = 0.5  # 若与障碍物的最小距离大于阈值（例如这里设置的阈值为robot_radius+0.2）,则设为一个较大的常值
        self.target = np.array([0,0])
        self.ob = np.array([0,0])


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
        self.v_sample = config.v_sample  # 线速度采样分辨率
        self.w_sample = config.w_sample  # 角速度采样分辨率
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma

        self.judge_distance = config.judge_distance

    def dwa_control(self, state, goal, obstacle):
        """滚动窗口算法入口
        Args:
            state (_type_): 机器人当前状态--[x,y,yaw,v,w]
            goal (_type_): 目标点位置，[x,y]
            obstacle (_type_): 障碍物位置，dim:[num_ob,2]
        Returns:
            _type_: 控制量、轨迹（便于绘画）
        """
        control, trajectory = self.trajectory_evaluation(state, goal, obstacle)
        return control, trajectory

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
        Va = self.__cal_obstacle_limit(state, obstacle)
        a = max([Vm[0], Vd[0], Va[0]])
        b = min([Vm[1], Vd[1], Va[1]])
        c = max([Vm[2], Vd[2], Va[2]])
        d = min([Vm[3], Vd[3], Va[3]])
        return [a, b, c, d]

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

    def __cal_obstacle_limit(self, state, obstacle):
        """环境障碍物限制Va
        Args:
            state (_type_): 当前机器人状态
            obstacle (_type_): 障碍物位置
        Returns:
            _type_: 某一时刻移动机器人不与周围障碍物发生碰撞的速度空间Va
        """
        v_low = self.v_min
        v_high = np.sqrt(2 * self._dist(state, obstacle) * self.a_vmax)
        w_low = self.w_min
        w_high = np.sqrt(2 * self._dist(state, obstacle) * self.a_wmax)
        return [v_low, v_high, w_low, w_high]

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

    def trajectory_evaluation(self, state, goal, obstacle):
        """轨迹评价函数,评价越高，轨迹越优
        Args:
            state (_type_): 当前状态---x,y,yaw,v,w
            dynamic_window_vel (_type_): 采样的速度空间窗口---[v_low,v_high,w_low,w_high]
            goal (_type_): 目标点位置，[x,y]
            obstacle (_type_): 障碍物位置，dim:[num_ob,2]
        Returns:
            _type_: 最优控制量、最优轨迹
        """
        G_max = -float('inf')  # 最优评价
        trajectory_opt = state  # 最优轨迹
        control_opt = [0., 0.]  # 最优控制
        dynamic_window_vel = self.cal_dynamic_window_vel(state[3], state[4], state, obstacle)  # 第1步--计算速度空间

        # 在速度空间中按照预先设定的分辨率采样
        sum_heading, sum_dist, sum_vel = 1, 1, 1  # 不进行归一化
        for v in np.arange(dynamic_window_vel[0], dynamic_window_vel[1], self.v_sample):
            for w in np.arange(dynamic_window_vel[2], dynamic_window_vel[3], self.w_sample):

                trajectory = self.trajectory_predict(state, v, w)  # 第2步--轨迹推算

                heading_eval = self.alpha * self.__heading(trajectory, goal) / sum_heading
                dist_eval = self.beta * self.__dist(trajectory, obstacle) / sum_dist
                vel_eval = self.gamma * self.__velocity(trajectory) / sum_vel
                G = heading_eval + dist_eval + vel_eval  # 第3步--轨迹评价

                if G_max <= G:
                    G_max = G
                    trajectory_opt = trajectory
                    control_opt = [v, w]

        return control_opt, trajectory_opt

    def _dist(self, state, obstacle):
        """计算当前移动机器人距离障碍物最近的几何距离
        Args:
            state (_type_): 当前机器人状态
            obstacle (_type_): 障碍物位置
        Returns:
            _type_: 移动机器人距离障碍物最近的几何距离
        """
        ox = obstacle[:, 0]
        oy = obstacle[:, 1]
        dx = state[0, None] - ox[:, None]
        dy = state[1, None] - oy[:, None]
        r = np.hypot(dx, dy)
        return np.min(r)

    def __dist(self, trajectory, obstacle):
        """距离评价函数
        表示当前速度下对应模拟轨迹与障碍物之间的最近距离；
        如果没有障碍物或者最近距离大于设定的阈值，那么就将其值设为一个较大的常数值。
        Args:
            trajectory (_type_): 轨迹，dim:[n,5]

            obstacle (_type_): 障碍物位置，dim:[num_ob,2]
        Returns:
            _type_: _description_
        """
        ox = obstacle[:, 0]
        oy = obstacle[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)
        return np.min(r) if np.array(r < self.radius + 0.2).any() else self.judge_distance

    def __heading(self, trajectory, goal):
        """方位角评价函数
        评估在当前采样速度下产生的轨迹终点位置方向与目标点连线的夹角的误差
        Args:
            trajectory (_type_): 轨迹，dim:[n,5]
            goal (_type_): 目标点位置[x,y]
        Returns:
            _type_: 方位角评价数值
        """
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = math.pi - abs(cost_angle)

        return cost

    def __velocity(self, trajectory):
        """速度评价函数， 表示当前的速度大小，可以用模拟轨迹末端位置的线速度的大小来表示
        Args:
            trajectory (_type_): 轨迹，dim:[n,5]
        Returns:
            _type_: 速度评价
        """
        return trajectory[-1, 3]


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

