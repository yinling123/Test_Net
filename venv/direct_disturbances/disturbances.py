import warnings
import numpy as np
import random

warnings.filterwarnings("ignore")


class disturbance:
    """
    进行子空间采样攻击类
    """
    def __init__(self, w, c1, c2, k, size, times, kind, prob, local, model, image, weight, flag):
        """
        进行类初始化
        :param model: 指向攻击模型
        :param times: 设置最大攻击次数
        :param kind: 记录模型检测类型
        :param prob: 记录模型检测置信度
        :param image: 记录模型图片矩阵(numpy)
        :param weight: 标记可以修改权重的重点关注区域
        :param size: 进行选取的数目
        :param flag: 如果为true则为无目标攻击，否则为目标攻击
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.model = model
        self.times = times
        self.kind = kind
        self.prob = prob
        self.image = image
        self.weight = weight
        self.size = size
        self.res = None
        self.identity = False
        self.sample = [] # 记录粒子位置
        self.V = []
        self.flag = flag
        self.p_best_position = None # 种群最佳位置
        self.p_best_fitness = 0 # 种群最佳适应度
        self.l_best_position = [] # 个体最佳位置
        self.l_best_fitness = [] # 个体最佳适应度
        self.final_best_position = None
        self.final_best_fitness = 0
        self.success = []
        self.local = local


    def l2(self, x):
        """
        计算l2距离
        :return:
        """
        distance = np.linalg.norm(x - self.image)
        return distance

    def fitness(self, prob, k):
        """
        计算适应度函数
        :return:
        """
        # 计算系数
        l2 = self.l2(self.sample[k])
        c = l2 * 1000
        fit = l2 + c * self.prob_abs(prob)
        return fit

    def prob_abs(self, prob):
        """
        计算置信度变化量
        :param prob:
        :return:
        """
        # 如果当前是无目标攻击
        if self.flag:
            if self.kind == prob[0]:
                return abs(prob[1] - self.prob)
            else:
                return abs(prob[1] + self.prob)

    def interrupt_position(self):
        """
        进行粒子的扰动生成
        :return:
        """
        shape = (400, 400, 3)
        # 产生随机扰动数
        noise = np.random.normal(0, np.random.rand() * 10, shape)
        noise[self.weight == 0] = 0
        # 改变对应维度上的值
        temp = np.round(noise + self.image)
        # 进行取整操作
        noise_matrix = temp.astype(np.uint8)
        # 进行数组范围限制
        np.clip(noise_matrix, 0, 255)

        return noise_matrix

    def interrupt_velocity(self):
        """
        随机生成粒子各个维度分量的速度
        :return:
        """
        # 产生随机速度，在-3-3之间
        v = np.random.randint(-3, 4, size=(400, 400, 3))
        v[self.weight == 0] = 0
        return v

    def noise_gaussian(self):
        """
        进行粒子的初始化
        :return:
        """
        for i in range(self.size):
            self.sample.append(self.interrupt_position())

    def velocity_init(self):
        """
        进行速度初始化
        :return:
        """
        for i in range(self.size):
            self.V.append(self.interrupt_velocity())

    def update_velocity(self, k):
        """
        更新粒子速度
        :return:
        """
        # 计算矩阵的形状及其维度
        shape = (400, 400, 3)
        # 生成对应维度的随机数
        r1 = np.random.random(shape)
        r2 = np.random.random(shape)
        r1[self.weight == 0] = 0
        r2[self.weight == 0] = 0

        # print(self.w)
        # print(self.V[k])
        # print(self.c1)
        # print(self.l_best_position[k])
        # print(self.sample[k])
        # print(self.c2)
        # print(self.p_best_position)
        # 更新速度
        self.V[k] = self.w * self.V[k] + self.c1 * r1 * (self.l_best_position[k] - self.sample[k]) + self.c2 * r2 * (
                self.p_best_position - self.sample[k])
        # 修改不符合条件的值
        self.V[k][self.V[k] < -3] = -3
        self.V[k][self.V[k] > 3] = 3

    def position_update(self, k):
        """
        进行更新
        :param k表示当前的对抗样本索引
        :return:
        """
        # 限制变化的方位
        temp = self.sample[k] + self.V[k]
        temp[temp < 0] = 0
        temp[temp > 255] = 255
        self.sample[k] = temp

    def update_best(self, prob, k):
        """
        更新个人最优位置及其种群最优情况
        :return:
        """
        # 进行求解之前先将前面的函数进行调用
        prob_now = self.fitness(prob, k)
        # 进行比较判断
        if self.p_best_fitness < prob_now:
            self.p_best_position = self.sample[k]
            self.p_best_fitness = prob_now

        # 当满足条件时更新当前最优位置
        if len(self.l_best_position) > k and self.l_best_fitness[k] < prob[1]:
            self.l_best_fitness[k] = prob[1]
            self.l_best_position[k] = self.sample[k]

    def pos_init(self):
        """
        进行pso迭代的初始化操作
        :return:
        """
        # 选取初始对抗样本的前k个
        temp = [self.sample[i] for i in range(self.k)]
        # 进行依次的比较更新最优位 置
        for i in range(len(temp)):
            # s为监测后回传的字符串
            prob = self.model.predict(self.sample[i])
            self.update_best(prob, i)
            # 更新个人最好位置和适应度函数
            self.l_best_fitness.append(prob[1])
            self.l_best_position.append(temp[i])


    def distinguish(self, s):
        """
        判断是否攻击成功
        :return:
        """
        # 当攻击网络判别结果和初次判别结果不同时，视为成功
        if s[0] == self.kind:
            return False
        return True

    def pso(self):
        """
        进行pso算法迭代生成对抗样本
        :return:
        """
        # 进行粒子和速度的初始化
        self.noise_gaussian()
        self.velocity_init()

        # 进行循环迭代求最优
        # 直到当前存储的值数目大于0
        times = 1
        # 限制最多迭代规定次数
        while times < self.times:
            # 先进行初始化操作
            self.pos_init()
            # 进行惯性权重的线性降低
            self.w = (self.times - times) * 0.6 / self.times + 0.4
            # 然后迭代求最优解
            for i in range(self.k):
                # 先更新速度，再更新位置
                self.update_velocity(i)
                self.position_update(i)
                # 更新最优解
                self.update_best(self.model.predict(self.sample[i]), i)
                times += 1

            # 使用适应度函数最好的进行迭代攻击
            s = self.local.predict(self.p_best_position)
            times += 1
            # 进行攻击判断
            if self.distinguish(s):
                # 如果攻击成功，则将对应参数转化
                self.identity = True
                if self.p_best_fitness > self.final_best_fitness:
                    self.final_best_position = self.p_best_position.copy()
                    self.res = s
            else:
                # 清空初始最优位置
                self.l_best_position.clear()
                self.l_best_fitness.clear()
                self.p_best_fitness = 0

                # 将速度和位置进行绑定排序
                t = list(zip(self.sample, self.V))
                random.shuffle(t)
                self.sample[:], self.V[:] = zip(*t)

        # 绘制最完美图像，如果攻击成功的话
        if self.identity:
            self.model.predict(self.final_best_position, flag=True, prob=self.res)
        # 返回攻击情况
        return times, self.identity

