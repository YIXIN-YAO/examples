import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)  # x = torch.flatten(x, 1) print(x.shape())
        x = self.dense(x)
        return x


model = Net()
model.load_state_dict(torch.load("mnist_model.pt"))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),  # 0-255转为0-1
                                transforms.Normalize(mean=[0.5], std=[0.5])])  # 转为-1，1之间
data_test = datasets.MNIST(root='../data',
                           transform=transform,
                           train=False)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)
data, label = next(iter(data_loader_test))
# 预测单张图片
img = data[1]
img = torch.unsqueeze(img, dim=1).type(torch.FloatTensor)[:2000]  # 增加一个维度变为[1, 1, 28, 28]才能作为网络输入
print(img.size())
output = model(img)
pred = output.argmax(dim=1, keepdim=True)
print(pred.item())
# 将单张图片显示出来
img = torchvision.utils.make_grid(img)  # 将图片变为[3, 28, 28]
print(img.size())
img = img.numpy().transpose(1, 2, 0)  # （channels,imagesize,imagesize）>>>（imagesize,imagesize,channels）
std = [0.5]
mean = [0.5]
img = img * std + mean  # 此时像素值变为0-1之间
plt.imshow(img)
plt.show()
