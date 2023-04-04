from pso import Pso
from image_process import convert_jpg
from net_1.net1_predict import resnet50
from net_2.net2_predict import alexnet
from image_process import image_to_matrix
from pylab import *
import os
import warnings

matplotlib.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']

warnings.filterwarnings("ignore")
class attack:
    """
    进行攻击的类
    """
    def __init__(self):
        pass

    def network_attack(self, local, box, pth, file_name=None):
        """
        传入攻击模型进行攻击的方法
        :param file_name: 图片名称
        :param local: 本地模型
        :param box: 黑盒模型
        :param pth: 图片路径
        :return:
        """
        # 读取图片进行转化
        img = convert_jpg(pth)
        data = image_to_matrix(img)
        # 转换类型
        # data = data.astype(np.float32)
        # 生成模型对象，并且进行初始化图像判别
        box_kind = box.predict(data, file_name=file_name)
        s = local.predict(data, file_name)
        p = Pso(2, 2, 2, 20, img, 0.2, s, 2, 10, box_kind=box_kind[0], local_net=local, box_net=box, file_name=file_name, b=30)
        return p.pso()

    def circular_attack(self, name, url, local, box):
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
                    res = self.network_attack(local, box, img_pth, file_name=file_name)
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
        进行网络之间的互相攻击
        :param 传入文件夹对应路径
        :return: 生成对应的图像
        """
        # 生成对应的网络
        res_net = resnet50()
        alex_net = alexnet()

        # 生成相应的计数
        res_net_to_alex_net_success = 0 # resnet50去攻击alex成功次数
        res_net_to_alex_net_visit = 0  # resnet50去攻击alex访问次数
        res_net_to_res_net_success = 0 # resnet50去攻击res成功次数
        res_net_to_res_net_visit = 0  # resnet50去攻击res访问次数
        alex_net_to_alex_net_success = 0 # alex去攻击alex成功次数
        alex_net_to_alex_net_visit = 0 # alex攻击alex的访问次数

        # 目前已经测试完成
        # name = "res_to_alex"
        # res_net_to_alex_net_success, res_net_to_alex_net_visit = self.circular_attack(name, url, res_net, alex_net)
        #
        # with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
        #     f.write('平均res_net_to_alex_net_success' + str(res_net_to_alex_net_success) + '\n')
        #     f.write('平均res_net_to_alex_net_visit' + str(res_net_to_alex_net_visit) + '\n')

        # 等待测试
        name = "res_res"
        res_net_to_res_net_success, res_net_to_res_net_visit = self.circular_attack(name, url, res_net, res_net)

        with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均res_net_to_res_net_success' + str(res_net_to_res_net_success) + '\n')
            f.write('平均res_net_to_res_net_visit' + str(res_net_to_res_net_visit) + '\n')

        name = "alex_alex"
        alex_net_to_alex_net_success, alex_net_to_alex_net_visit = self.circular_attack(name, url, alex_net, alex_net)

        with open('record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均alex_net_to_alex_net_success' + str(alex_net_to_alex_net_success) + '\n')
            f.write('平均alex_net_to_alex_net_visit' + str(alex_net_to_alex_net_visit) + '\n')


        # 进行图像绘画
        scale_ls = range(3)
        index_ls = [' res_net_to_alex_net', 'res_net_to_res, alex_net_to_alex_net']
        val_ls = [0.89, res_net_to_res_net_success, alex_net_to_alex_net_success]
        plt.bar(scale_ls, val_ls) # 进行绘图
        plt.xticks(scale_ls, index_ls) # 设置坐标字
        plt.xlabel('网络关系')
        plt.ylabel('成功率')
        plt.title('网络攻击成功率对比')
        plt.savefig('img_data/success1.jpg')

        val_ls = [17.41, res_net_to_res_net_visit, alex_net_to_alex_net_visit]
        plt.bar(scale_ls, val_ls)  # 进行绘图
        plt.xticks(scale_ls, index_ls)  # 设置坐标字
        plt.xlabel('网络关系')
        plt.ylabel('平均访问次数')
        plt.title('平均访问次数对比')
        plt.savefig('img_data/visit1.jpg')




if __name__ == '__main__':
    # img目录存放数据集
    # img_out存储攻击成功的数据
    attack = attack()
    attack.run(r'dog_img')
