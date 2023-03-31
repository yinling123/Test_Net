import numpy as np
import pickle
import imageio
import os


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

if __name__ == '__main__':

    list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for z in range(10):
        if os.path.exists('D:/数据集/cifar-10/train/' + list[z]) == False:
            os.makedirs('D:/数据集/cifar-10/train/' + list[z])
        if os.path.exists('D:/数据集/cifar-10/val/' + list[z]) == False:
            os.makedirs('D:/数据集/cifar-10/val/' + list[z])

    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    for j in range(1, 6):
        dataName = r"D:/数据集/cifar-10-batches-py/data_batch_" + str(j)
        # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。也就是这里的dataName的绝对路径
        Xtr = unpickle(dataName)
        print(dataName + " is loading...")

        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            picName = r'D:/数据集/cifar-10/train/' + list[Xtr['labels'][i]] + '/' + str(i + (j - 1) * 10000) + '.jpg'  # png
            # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
            imageio.imwrite(picName, img)
        print(dataName + " loaded.")
    print("test_batch is loading...")

    # 生成测试集图片
    testXtr = unpickle(r"D:/数据集/cifar-10-batches-py/test_batch")  # 同dataName一样也是自己的文件所在路径

    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = 'D:/数据集/cifar-10/val/' + list[testXtr['labels'][i]] + '/' + str(i) + '.jpg'
        imageio.imwrite(picName, img)
    print("test_batch loaded.")