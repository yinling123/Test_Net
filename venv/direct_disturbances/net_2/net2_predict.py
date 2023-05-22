import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torchvision
import torch
import warnings
from torchvision import transforms
from PIL import Image
import cv2
import time
import torch.nn.functional as F
from torch.autograd import Variable


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
with open(r'/mnt/test/venv/direct_disturbances/net_2/imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]

classes_final = []

for i in classes:
    if ',' in i:
        temp = i.split(',')
        for j in temp:
            classes_final.append(j)
    else:
        classes_final.append(i)

# print(111)

def convertjpg(jpgfile, width=400, height=400):
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


    def predict(self, img, file_name=" ", flag=False, prob=None, cam=False):
        """
        进行图片检测
        :param img:
        :param file_name:
        :param flag:
        :param prob:
        :return:
        """
        # 进行图像转换和预处理
        img = Image.fromarray(np.uint8(img))
        # 进行判断
        if flag:
            plt.imshow(img)
            plt.title(prob[0] + ':' + str(round(prob[1], 2)))
            plt.savefig(r'/mnt/test/venv/direct_disturbances/img_out/alex/' + time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time())) + '.jpg')
            return
        # print(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t, 0)

        # 进行模型推理
        out = self.model(batch_t)

        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0]
        if cam == False:
            return classes[index[0]], percentage[index[0]].item()
        else:
            return classes[index[0]], percentage[index[0]].item(), index[0]

    def generate_cam(self, img, target_class):
        # 获取目标类别的得分
        output = self.model(img)

        cam = output[0].detach().cpu().numpy()

        return cam

    def draw_cam(self, img_pth, target_class):
        """
        绘制类间激活热图
        :param img_pth:
        :return:
        """
        img = cv2.imread(img_pth)
        if img.shape[-1] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))

        # Preprocess the image
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.cuda()
        batch_t = torch.unsqueeze(img_tensor, 0)
        # Run the image through the model

        cam = self.generate_cam(batch_t, target_class)

        # Resize the CAM to the original image size
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Normalize the CAM values between 0 and 1
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        # Convert the CAM to RGB
        # cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2RGB)
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        # Overlay the CAM on the original image
        cam = cv2.addWeighted(img, 0.5, cam, 0.5, 0)

        # Plot the result
        self.img_path = r'/mnt/test/venv/direct_disturbances/img_heat/alex/' + time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time())) + '区间热图.jpg'
        plt.imshow(cam)
        plt.title("Interval heat map")
        plt.savefig(self.img_path)
        self.cam = cam

    def get_importance(self, img_pth, target_class):
        """
        读取类间激活热图，并且获取
        :param img_pth:
        :return:
        """
        self.draw_cam(img_pth, target_class)
        # img = cv2.imread(self.img_path)
        img = cv2.resize(self.cam, (400, 400))

        # 将RGB图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 将灰度图像转换为二进制图像
        # 像素点低于127的全部转化为0，高于127的全部转化为255
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary

if __name__ == '__main__':
    box_net = alexnet()
    pth = r'D:\Note\Test_Net\venv\pso\img'
    img_pth = r'dog.jpg'
    img = convertjpg(img_pth)
    img = np.asarray(img).copy()
    prob = box_net.predict(img, file_name=img_pth, flag=False,cam=True)
    # box_net.draw_cam(img_pth)
    # for file_name in os.listdir(pth):
    #     # print(file_name)
    #     # 遍历文件夹获取文件路径
    #     img_pth = ('/'.join([pth, file_name]))
    #     img = convertjpg(img_pth)
    #     # img = Image.open(r'mm.jpeg')
    #     img = numpy.asarray(img).copy()
    #     box_net.predict(img,file_name=file_name,flag=True)
    print(box_net.get_importance('dog.jpg', target_class=prob[2]))

