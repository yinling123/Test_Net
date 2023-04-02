from PIL import Image
import matplotlib.pyplot as plt
# pip install -U matplotlib
import torch
# pip install pytorch
import torchvision.transforms as T
import torchvision
# pip install torchvision
import numpy as np
import cv2
import time

import os
import warnings

from torchvision.transforms import transforms

from net_2.net2_predict import transform

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

    def predict(self, img, file_name = None, threshold=0.2, flag=False, rect_th=3, text_size=1, text_th=3):
        # img = Image.open(img_path)
        # 转换一个PIL库的图片或者numpy的数组为tensor张量类型；转换从[0,255]->[0,1]

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
        if flag == False:
            return str(pred_class[0]), max(pred_score)
        else:
           return self.object_detection_api(img, pred_class, pred_boxes, pred_score)

    def object_detection_api(self, img, pred_class, pred_boxes, pred_score, threshold=0.5, rect_th=3, text_size=1, text_th=3):
        boxes, pred_cls, pred_score = pred_boxes, pred_class, pred_score
        image = img.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
        plt.imshow(image)
        plt.title(pred_cls[0] + ':' + str(round(max(pred_score), 2)))
        plt.savefig(r'/mnt/test/venv/pso/img_out/' + 'successRes' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + '.jpg')
        # plt.show()
        return str(pred_cls[0]), max(pred_score)

if __name__ == '__main__':
    img = Image.open(r"10.jpg")
    local_net = resnet50()
    matrix = np.asarray(img).copy()
    print(local_net.predict(matrix, flag=True))
    # print(local_net.e)

