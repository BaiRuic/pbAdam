import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import model.cnn as cnn
import optim.pbAdam as pbAdam
import optim.pbSGD as pbSGD 



def dataload(batch_size):
    
    # 数据预处理。transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行标准化（数据在0~1之间）
    # transforms.Normalize()做归一化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
    # transforms.Compose()函数则是将各种预处理的操作组合到了一起
    
    data_tf = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    # 数据集的下载器 如果下载过了， 参数download就是False
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=data_tf, download=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(epoch):
    """模型训练部分

    Args:
        epoch ([int]): [当前的epoch]
    """
    model.train()
    train_ls = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_ls += loss.item()
        _, predicted = outputs.max(1)
        # 该批次 样本总数
        total += targets.size(0)
        # 计算正确率
        correct += predicted.eq(targets).sum().item()
    train_loss.append(train_ls/(batch_idx+1))
    print('Train Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_ls/(batch_idx+1), 
            100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    model.eval()
    test_ls = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_ls += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    test_loss.append(test_ls/(batch_idx+1))
    print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_ls/(batch_idx+1), 100.*correct/total, correct, total))

# 定义参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 3
batch_size=64
learning_rate = 0.02

# 定义模型，损失函数，优化器
model = cnn.CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = pbSGD.pbSGD(model.parameters(), lr=learning_rate)
optimizer = pbAdam.pbAdam(model.parameters(), lr=learning_rate)

# 数据生成器
trainloader,testloader = dataload(batch_size)

# 训练模型
train_loss = []
test_loss = []
for epoch in range(epochs):
    print(f'\nEpoch: {epoch + 1} / {epochs}')
    train(epoch)
    test(epoch)

# 绘图
plt.figure(figsize=(12,6))
plt.plot(train_loss,label='train_loss')
plt.plot(test_loss, label='valid_loss')
plt.legend()
plt.show()
    



