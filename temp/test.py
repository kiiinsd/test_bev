import torch
import torchvision

def getMeanAndBias(dataset):
    '''
        输入：
            iter_dataloader 要求：元素为tensor类型、转换为torch.utils.data.Dataset类
    '''

    # 1. dataloader加载数据
    # 注意batch_size的用法。这种条件下，数据集会被逐样本分割，而不是成批在一起
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. 初始化通道平均值和标准差
    means = torch.zeros(3)
    bias = torch.zeros(3)

    # 3. 计算每张图片的每个通道的平均值和以及方差和
    for img, _ in data_iter:
        for d in range(3):
            # 注意针对单个通道计算平均值和标准差的方法
            means[d] += img[:, d, :, :].mean()
            bias[d] += img[:, d, :, :].std()
    
    means = means / len(data_iter)
    bias = bias / len(data_iter)

    return [means, bias]

# 注意要设计转换器，将元素转换为tensor类型
transfroms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# 填写自己的图片文件夹路径
train_data = torchvision.datasets.ImageFolder('./data/panosim/samples', transform=transfroms)

stat1 = getMeanAndBias(train_data)
print(stat1)
# 输出
# [tensor([0.5734, 0.4586, 0.2882]), tensor([0.2108, 0.2169, 0.1982])]