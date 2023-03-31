import matplotlib.pyplot as plt
import numpy
import pylab as pl
from torchvision import models
import torch
import warnings
from torchvision import transforms
from PIL import Image
import torchvision.transforms as T
import os


warnings.filterwarnings('ignore')

# 下载预训练的网络模型
# alexnet = models.alexnet(pretrained=True)

# 进行图形转变
transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
 )])

# 开启评估模式
# alexnet.eval()

# 读取类的标签
with open(r'D:\Python\Test_Net\venv\pso\net_2\imagenet_classes.txt') as f:
  classes = [line.strip()  for line in f.readlines()]

classes_final = []

for i in classes:
    if ',' in i:
        temp = i.split(',')
        for j in temp:
            classes_final.append(j)
    else:
        classes_final.append(i)

# print(111)

def convertjpg(jpgfile, width=32, height=32):
    """
    将图片格式进行转化
    :param jpgfile:
    :param outdir:
    :param width:
    :param height:
    :return:
    """
    # 将图片进行读取
    img = Image.open(jpgfile)
    # 重新转化尺寸
    try:
        new_img = img.resize((width, height), Image.LANCZOS)
    except Exception as e:
        print(e)
    # 返回结果
    return new_img

class alexnet:
    """
    被攻击的模型
    """
    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.eval()


    def predict(self, img, file_name, flag = False):
        # 进行图像转换和预处理
        print(type(img))
        img = Image.fromarray(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t,0)

        # 进行模型推理
        out = self.model(batch_t)

        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0]

        if flag:
            plt.imshow(img)
            plt.title(classes[index[0]] + ':' + str(round(percentage[index[0]].item(), 2)))
            plt.savefig(r'D:/Note/Test_Net/venv/pso/img_out/' + 'successAlex' + file_name);
        return classes[index[0]], percentage[index[0]].item()


if __name__ == '__main__':
    box_net = alexnet()
    pth = r'D:\Note\Test_Net\venv\pso\img'
    img_pth = r'dog.jpg'
    img = convertjpg(img_pth)
    img = numpy.asarray(img).copy()
    box_net.predict(img, file_name=img_pth, flag=True)
    # for file_name in os.listdir(pth):
    #     # print(file_name)
    #     # 遍历文件夹获取文件路径
    #     img_pth = ('/'.join([pth, file_name]))
    #     img = convertjpg(img_pth)
    #     # img = Image.open(r'mm.jpeg')
    #     img = numpy.asarray(img).copy()
    #     box_net.predict(img,file_name=file_name,flag=True)

