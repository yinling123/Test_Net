import matplotlib.pyplot as plt


if __name__ == '__main__':
    scale_ls = range(2)
    index_ls = ['resnet', 'alexnet']
    color = [None, 'orange']
    val_ls = [100, 52]
    for i in scale_ls:
        plt.bar(index_ls[i], val_ls[i], label=index_ls[i])
    plt.text(index_ls[0], val_ls[0], val_ls[0], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    plt.text(index_ls[1], val_ls[1], val_ls[1], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    plt.title('Network attack success rate')
    plt.xlabel("Networks", fontproperties='Times New Roman', fontsize=15)
    plt.ylabel("Accuracy (%)", fontproperties='Times New Roman', fontsize=15)
    plt.legend()
    plt.savefig('img_data/success2.jpg')

    # val_ls = [17.41, res_net_to_res_net_visit, alex_net_to_alex_net_visit]
    # plt.bar(scale_ls, val_ls)  # 进行绘图
    # plt.xticks(scale_ls, index_ls)  # 设置坐标字
    # plt.xlabel('网络关系')
    # plt.ylabel('平均访问次数')
    # plt.title('平均访问次数对比')
    # plt.savefig('img_data/visit1.jpg')