import matplotlib.pyplot as plt

# ncol = 2 表示图例要放两列
# plt.legend(prop={'family' : 'Times New Roman', 'size': 35}, ncol = 2)

if __name__ == '__main__':

    scale_ls = range(3)
    index_ls = ['res_net_to_alex_net', 'res_net_to_res', 'alex_net_to_alex_net']
    # val_ls = [89, 100, 96]
    # for i in range(3):
    #     plt.bar(index_ls[i], val_ls[i])
    # plt.xlabel("Type", fontproperties='Times New Roman', fontsize=20)
    # plt.ylabel("Accuracy (%)", fontproperties='Times New Roman', fontsize=20)
    # plt.xticks(scale_ls, index_ls)  # 设置坐标字
    # plt.text(index_ls[0], val_ls[0], val_ls[0], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    # plt.text(index_ls[1], val_ls[1], val_ls[1], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    # plt.text(index_ls[2], val_ls[2], val_ls[2], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    # # plt.xlabel('网络关系')
    # # plt.ylabel('成功率')
    # plt.title('Comparison of attack success rates')
    # plt.savefig('img_data/success1.jpg')

    val_ls = [17.41, 4.84, 9.19]
    for i in range(3):
        plt.bar(index_ls[i], val_ls[i])
    plt.xticks(scale_ls, index_ls)  # 设置坐标字
    plt.xlabel("Type", fontproperties='Times New Roman', fontsize=20)
    plt.ylabel("Times", fontproperties='Times New Roman', fontsize=20)
    plt.text(index_ls[0], val_ls[0], val_ls[0], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    plt.text(index_ls[1], val_ls[1], val_ls[1], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    plt.text(index_ls[2], val_ls[2], val_ls[2], ha='center', fontproperties='Times New Roman', fontsize=15, zorder=10)
    # plt.xlabel('网络关系')
    # plt.ylabel('平均访问次数')
    plt.title('Comparison of average visits')
    plt.savefig('img_data/visit1.jpg')