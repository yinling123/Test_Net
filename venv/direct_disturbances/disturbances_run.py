from disturbances import disturbance
from image_process import convert_jpg
from net_1.net1_predict import resnet50
from net_2.net2_predict import alexnet
from image_process import image_to_matrix
from pylab import *
import os
import warnings

warnings.filterwarnings("ignore")
class attack:
    """
    进行攻击的类
    """
    def __init__(self):
        pass

    def network_attack(self, box, pth):
        """
        传入攻击模型进行攻击的方法
        :param box: 黑盒模型
        :param pth: 图片路径
        :return: 攻击结果和最后的攻击次数
        """
        # 读取图片进行转化
        img = convert_jpg(pth)
        data = image_to_matrix(img)
        # 转换类型
        # data = data.astype(np.float32)
        # 生成模型对象，并且进行初始化图像判别
        prob = box.predict(data)
        weight = box.get_importance(pth)
        p = disturbance(2, 2, 2, 10, 20, 200, prob[0], prob[1], box, data, weight, flag=True)
        return p.pso()

    def circular_attack(self, name, url, box):
        """
        进行文件夹下循环遍历攻击(直接找数据集)
        :return:
        """
        pth = url  # 记录路径
        img_num = 0  # 记录图片数目
        visit_times = 0  # 记录平均访问次数
        success_time = 0  # 记录成功次数
        # 遍历过程出现异常记录异常的图片信息
        try:
            for file_name in os.listdir(pth):
                # 遍历文件夹获取文件路径
                img_pth = ('/'.join([url, file_name]))
                # 获取结果
                try:
                    res = self.network_attack(box, img_pth)
                    img_num += 1
                    visit_times += res[0]
                    if res[1]:
                        success_time += 1
                        # 记录总的访问次数
                except Exception as e:
                    print(e)
                    with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
                        f.write(name + '出错的图片名称' + file_name + '\n')
        except Exception as e:
            print(e)
        finally:
            # 统计总的成功率和平均访问次数
            print("攻击成功次数", success_time)
            print("检测次数", visit_times)
            print("检测的图片总数", img_num)
            return round(success_time / img_num, 2), round(visit_times / img_num, 2)


    def run(self, url):
        """
        进行代码运行
        :param 传入文件夹对应路径
        :return: 生成对应的图像
        """
        # 生成对应的网络
        res_net = resnet50()
        alex_net = alexnet()

        # 生成相应的计数
        res_visit_times = 0
        res_success_times = 0
        alex_visit_times = 0
        alex_success_times = 0

        # 进行网络攻击
        name = "res"
        res_success_times, res_visit_times = self.circular_attack(name, url, res_net)

        with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均res_success' + str(res_success_times) + '\n')
            f.write('平均res_visit' + str(res_visit_times) + '\n')

        name = "alex"
        alex_success_times, alex_visit_times = self.circular_attack(name, url, alex_net)

        with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均alex_success' + str(alex_success_times) + '\n')
            f.write('平均alex_visit' + str(alex_visit_times) + '\n')


if __name__ == '__main__':
    # img目录存放数据集
    # img_out存储攻击成功的数据
    attack = attack()
    attack.run(r'dog_img')
