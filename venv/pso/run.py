from pso import Pso
from image_process import convert_jpg
from net_1.net1_predict import resnet50
from net_2.net2_predict import alexnet
from net_4.net4_predict import faster_cnn
from image_process import image_to_matrix
from pylab import *

matplotlib.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']

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

    def circular_attack(self, url, local, box):
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
                print(img_pth)
                # 获取结果
                res = self.network_attack(local, box, img_pth, file_name=file_name)
                # 如果攻击成功,成功次数+1
                if res[1]:
                    success_time += 1
                # 记录总的访问次数
                img_num += 1
                visit_times += visit_times
        except Exception:
            print()
            with open('../record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
                f.write('出错的图片名称' + file_name + '\n')
        finally:
            # 统计总的成功率和平均访问次数
            return round(success_time / img_num, 2), round(visit_times / img_num)

    def run(self, url):
        """
        进行网络之间的互相攻击
        :param 传入图片路径
        :return: 生成对应的图像
        """
        # 生成对应的网络
        res_net = resnet50()
        alex_net = alexnet()
        faster_net = faster_cnn()

        # 生成相应的计数
        res_net_to_alex_net_success = 0 # resnet50去攻击alex
        res_net_to_alex_net_visit = 0  # resnet50去攻击alex
        res_net_to_faster_cnn_success = 0 # resnet50去攻击faster
        res_net_to_faster_cnn_visit = 0  # resnet50去攻击faster

        res_net_to_alex_net_success, res_net_to_alex_net_visit = self.circular_attack(url, res_net, alex_net)

        with open('../record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均res_net_to_alex_net_success' + str(res_net_to_alex_net_success) + '\n')
            f.write('平均res_net_to_alex_net_visit' + str(res_net_to_alex_net_visit) + '\n')

        res_net_to_faster_cnn_success, res_net_to_faster_cnn_visit = self.circular_attack(url, res_net, faster_net)

        with open('../record.txt', 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            f.write('平均res_net_to_faster_cnn_success' + str( res_net_to_faster_cnn_success) + '\n')
            f.write('平均res_net_to_faster_cnn_visit' + str(res_net_to_faster_cnn_visit) + '\n')

        # 进行图像绘画
        scale_ls = range(2)
        index_ls = [' res_net_to_alex_net', 'res_net_to_faster_cnn']
        val_ls = [res_net_to_alex_net_success, res_net_to_faster_cnn_success]
        plt.bar(scale_ls, val_ls) # 进行绘图
        plt.xticks(scale_ls, index_ls) # 设置坐标字
        plt.xlabel('网络关系')
        plt.ylabel('成功率')
        plt.title('网络攻击成功率对比')
        plt.savefig('img_out/success1.jpg')

        val_ls = [res_net_to_alex_net_visit, res_net_to_faster_cnn_visit]
        plt.bar(scale_ls, val_ls)  # 进行绘图
        plt.xticks(scale_ls, index_ls)  # 设置坐标字
        plt.xlabel('网络关系')
        plt.ylabel('平均访问次数')
        plt.title('平均访问次数对比')
        plt.savefig('img_out/visit1.jpg')




if __name__ == '__main__':
    # img目录存放数据集
    # img_out存储攻击成功的数据
    attack = attack()
    attack.run(r'img')
