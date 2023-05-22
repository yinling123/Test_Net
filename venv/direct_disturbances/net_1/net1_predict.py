from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import warnings
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.transforms import transforms

# from net_2.net2_predict import transform

warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
 mean = [0.485, 0.456, 0.406],
 std = [0.229, 0.224, 0.225]
 )])

# pip install opencv-python



# 下载已经训练好的模型
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class resnet50:
    """
    创建本地模型类
    """
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        if torch.cuda.is_available():
            self.model.cuda().eval()
        else:
            self.model.eval()

    def predict(self, img, file_name=None, threshold=0.2, flag=False, prob=None, cam=False, rect_th=3, text_size=1, text_th=3):
        """
        进行网络识别预测
        :param img:
        :param file_name:
        :param threshold:
        :param flag:
        :param rect_th:
        :param text_size:
        :param text_th:
        :return:
        """
        # 转换一个PIL库的图片或者numpy的数组为tensor张量类型；转换从[0,255]->[0,1]
        if flag:
            return self.object_detection_api(img, prob)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        img = img.cuda()
        img = torch.tensor(img, dtype=torch.float32)

        pred = self.model([img])
        # print(pred)
        # print(pred[0]['labels'].numpy())

        # 类别提取
        pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        # 坐标提取
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]

        # 找出符合相似度要求的
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        # print(pred_score[0])

        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]

        pred_boxes = pred_boxes[:pred_t + 1]
        pred_class = pred_class[:pred_t + 1]
        # print("pred_class:", pred_class)
        # print("pred_boxes:", pred_boxes)
        if cam:
            return str(pred_class[0]), max(pred_score), COCO_INSTANCE_CATEGORY_NAMES.index(pred_class[0])
        if flag == False:
            return str(pred_class[0]), max(pred_score)

    def object_detection_api(self, img, prob, threshold=0.5, rect_th=3, text_size=1, text_th=3):
        """
        进行照片的绘制
        :param img:
        :param pred_class:
        :param pred_boxes:
        :param pred_score:
        :param threshold:
        :param rect_th:
        :param text_size:
        :param text_th:
        :return:
        """
        # image = img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        img = Image.fromarray(np.uint8(img))
        plt.imshow(img)
        plt.title(prob[0] + ':' + str(round(prob[1], 2)))
        plt.savefig(r'/mnt/test/venv/direct_disturbances/img_out/res/' + time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time())) + '.jpg')
        # plt.show()
        return prob

    def get_cam(self, img_tensor, target_class):
        output = self.model([img_tensor])
        cam = output[0]['boxes'].detach().cpu().numpy()
        return cam

    def draw_cam(self, img_path, target_class, transform=None, visual_heatmap=False, out_layer=None):
        """
        绘制热力图
        :param model:
        :param img_path:
        :param save_path:
        :param transform:
        :param visual_heatmap:
        :param out_layer:
        :return:
        """
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))

        # Preprocess the image
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.cuda()

        # Get the class activation map
        cam = self.get_cam(img_tensor, target_class)

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
        self.cam = cam

        # Plot the result
        self.img_path = r'/mnt/test/venv/direct_disturbances/img_heat/res/' + time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time())) + '区间热图.jpg'
        plt.imshow(cam)
        plt.title("Interval heat map")
        plt.savefig(self.img_path)


    def get_importance(self, img_path, target_class):
        """
        读取类间激活热图，并且获取
        :param img_path:
        :return:
        """
        self.draw_cam(img_path, target_class)
        img = self.cam
        # img = cv2.resize(img, (400, 400))
        # 将RGB图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 将灰度图像转换为二进制图像
        # 像素点低于127的全部转化为0，高于127的全部转化为255
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary


if __name__ == '__main__':
    img = Image.open(r"mm.jpeg")
    local_net = resnet50()
    prob = local_net.predict(img, cam=True)
    # matrix = np.asarray(img).copy()
    # img = img + np.zeros(matrix.shape)
    # print(img == matrix)

    # print(local_net.predict(img, flag=False))
    # print(local_net.e)
    print(local_net.get_importance(r'dog.jpg', target_class=prob[2]))
