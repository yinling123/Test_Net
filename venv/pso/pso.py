# 本问题的固定函数是样本变化前后的置信度变化和与原图片的像素矩阵差别总值
import random
import image_process as Ip
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 构建粒子群算法类
class Pso:
    """
    粒子群算法实现类
    """
    def __init__(self, w, c1, c2, size, A, y, s, c, k, box_kind, local_net, box_net, file_name, b=0):
        """
        进行粒子群的初始化
        :param w: 惯性因素，后期会进行动态变化
        :param c1: 自身的学习因子，前期应尽量大，后期慢慢变小
        :param c2: 朝着种群最优解的地方的学习因子，前期应尽量小，后期慢慢变大
        :param size:初始化的粒子数目
        :param X:存储各个粒子的坐标
        :param V:存储各个粒子的速度
        :param A:初始图片像素矩阵
        :param y:缩减因子（应该在0-1之间）
        :param a:惩罚项系数求解的常系数
        :param s:检验完返回的字符串
        :param a:适应度函数的惩罚项系数
        :param c:计算惩罚项系数的常系数
        :param prob:记录置信度变化值
        :param file_name:保存图片路径
        最大最小速度限制在-3——3
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.size = size
        self.X = []
        self.V = []
        self.A = A
        self.y = y
        self.prob_init = s[1]
        self.c = c
        self.prob = None
        self.a = 1
        self.l2 = None
        self.g_position = None  # 记录最佳的矩阵位置
        self.g_fitness = 0  # 记录最佳的函数值
        self.p_best = []  # 记录每个个体最优位置
        self.g_best = []  # 记录每个个体最优适应度
        self.k = k  # 记录每次的循环次数
        self.success = []  # 存储攻击成功的数目
        self.kind = None  # 记录当前的返回值类型
        self.box_kind = box_kind  # 记录黑盒攻击的判别结果
        self.local_net = local_net  # 记录本地网络
        self.box_net = box_net  # 记录黑盒网络模型
        self.identity = False # 判断是否攻击成功
        self.file_name = file_name
        self.X_copy = [] # copy每个粒子的坐标，用于规范大小
        # self.X_MIN = None # 记录位置的最小值
        # self.X_MAX = None # 记录位置的最大值
        self.b = b

    def noise_gaussian(self):
        """
        进行初始化粒子群的扰动生成
        :return:
        """
        for i in range(self.size):
            self.X.append(Ip.interrupt_position(self.A))
        # 将位置进行copy
        self.X_copy = self.X.copy()
        # self.X_MIN = self.X_copy[self.X_copy - self.b < 0] = 0
        # self.X_MAX = self.X_copy[self.X_copy - self.b > 255] = 255

    def v_init(self):
        """
        进行速度初始化
        :return:
        """
        for i in range(self.size):
            self.V.append(Ip.interrupt_velocity(self.y))

    def v_update(self, k):
        """
        更新粒子的速度
        :param k代表为第k个粒子进行更新
        :return:
        """
        # 计算矩阵的形状及其维度
        # shape = self.V.shape
        shape = (400, 400, 3)
        # 生成对应维度的随机数
        r1 = np.random.random(shape)
        r2 = np.random.random(shape)
        # 进行速度更新
        # print(self.p_best)
        self.V[k] = self.w * self.V[k] + self.c1 * r1 * (self.p_best[k] - self.X[k]) + self.c2 * r2 * (
                    self.g_position[k] - self.X[k])
        # 修改不符合条件的值
        self.V[k][self.V[k] < -3] = -3
        self.V[k][self.V[k] > 3] = 3

    def position_update(self, k):
        """
        进行更新
        :param k表示当前的对抗样本个数
        :return:
        """
        # 限制变化的方位
        temp = self.X[k] + self.V[k]
        temp[temp - self.b < 0] = 0
        temp[temp + self.b > 255] = 255

        self.X[k] = temp

        # self.X[k] = np.clip(self.X[k] + self.V[k], 0, 255)

    def similarity(self, next_generation):
        """
        根据l2公式求解出模型相关性
        :param next_generation: 下一代的像素矩阵
        :return:返回l2范数值
        """
        # 调用numpy库函数求解l2范数
        self.l2 = np.linalg.norm(self.A - next_generation)
        # print(self.l2)

    def prob_abs(self, s):
        """
        进行置信度绝对值计算
        :param s: 扰动后的下一代对抗样本的字符串
        :return: 返回绝对值
        """
        if self.kind == s[0]:
            # print(s[1])
            return abs(s[1] - self.prob)
        else:
            # print(s[1])
            return abs(s[1] + self.prob)

    def penalty_coefficient(self):
        """
        计算惩罚因子
        :return:返回惩罚因子的值
        """
        # 进行循环迭代处理求解符合条件的值
        self.a = self.prob / self.l2
        for i in range(100):
            # 当满足条件时退出循环
            if self.a < self.prob / self.l2 < self.a / self.y:
                break
            else:
                self.a = self.c * (self.y ** i)

    def fitness_func(self, s, k):
        """
        进行适应度函数求解
        :param s:表示传输的字符串
        :param k:表示当前的样本次序
        :return:
        """
        # 进行求解之前先将前面的函数进行调用
        self.similarity(self.X[k])
        prob_now = self.prob_abs(s) - self.a * self.l2
        # print(s)
        # 进行比较判断
        # print(self.g_fitness)
        if self.g_fitness > prob_now:
            self.g_position = self.X[k]
            self.g_fitness = prob_now
        # print('种群最优位置为:',self.g_position)

        # 当满足条件时更新当前最优位置
        if len(self.p_best) > k and self.p_best[k] < s[1]:
            self.p_best[k] = s[1]
            self.g_best[k] = self.X[k]

    def pos_init(self):
        """
        进行pso迭代的初始化操作
        :return:
        """
        # 选取初始对抗样本的前k个
        temp = [self.X[i] for i in range(self.k)]
        # 进行依次的比较更新最优位 置
        for i in range(len(temp)):
            # s为监测后回传的字符串
            s = self.local_net.predict(self.X[i])
            self.fitness_func(s, i)
            self.p_best.append(s[1])
            self.g_best.append(temp[i])
        # print(self.p_best)

    def distinguish(self, s):
        """
        判断是否攻击成功
        :return:
        """
        # 当黑盒网络判别结果和初次判别结果不同时，视为成功
        if self.kind in s[0] or s[0] == self.box_kind:
            return False
        return True

    def pso(self):
        """
        进行pso算法迭代生成对抗样本
        :return:
        """
        # 进行粒子和速度的初始化
        self.noise_gaussian()
        self.v_init()
        temp = self.local_net.predict(self.A, file_name=self.file_name)
        self.kind = temp[0]
        self.prob = temp[1]

        # 进行循环迭代求最优
        # 直到当前存储的值数目大于0
        times = 0
        # 限制最多攻击200次
        while len(self.success) == 0 and times < 100:
            # 先进行初始化操作
            self.pos_init()
            # 进行惯性权重的线性降低
            self.w = (100 - times) * 0.6 / 100 + 0.4
            # 然后迭代求最优解
            for i in range(self.k):
                # 先更新速度，再更新位置
                self.v_update(i)
                self.position_update(i)
                # 更新最优解
                self.fitness_func(self.local_net.predict(self.X[i], file_name=self.file_name), i)

            # print(66666)
            # 使用适应度函数最好的进行黑盒攻击
            s = self.box_net.predict(self.g_position, self.file_name)
            print(s)
            if self.distinguish(s):
                # 如果攻击成功，则将数目加1
                self.success.append(s)
                self.success.append(times)
                self.identity = True
                # 进行攻击成功的图片显示
                self.box_net.predict(self.g_position, file_name=self.file_name, flag=True)
                print(times + 1)
            else:
                # 清空初始最优位置
                self.p_best.clear()
                # 进行随机排序
                random.shuffle(self.X)

            times += 1
        # 返回攻击情况
        return times, self.identity
