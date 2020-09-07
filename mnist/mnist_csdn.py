import torchvision
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


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


transform = transforms.Compose([transforms.ToTensor(), # 0-255转为0-1
                                transforms.Normalize(mean=[0.5], std=[0.5])]) # 转为-1，1之间

data_train = datasets.MNIST(root='../data',
                            transform=transform,
                            train=True,
                            download=True)
data_test = datasets.MNIST(root='../data',
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)
# print(len(data_loader_train))  # 这个的长度居然是九百多
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda")
net = Net().to(device)
optimizer = optim.Adadelta(net.parameters(), lr=1)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # 固定步长学习率衰减

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(3):

    for i,data in enumerate(data_loader_train):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, i * len(inputs), len(data_loader_train.dataset),
                       100. * i / len(data_loader_train), loss.item()))


    test(net, device, data_loader_test)
    scheduler.step()

torch.save(net.state_dict(), "mnist_model.pt")



# img = torchvision.utils.make_grid(data)  # 这时候img变成了一张三通道的大图
# img = img.numpy().transpose(1,2,0)  # （channels,imagesize,imagesize）>>>（imagesize,imagesize,channels）
# std = [0.5]
# mean = [0.5]
# img = img*std+mean  # 此时像素值变为0-1之间
# plt.imshow(img)
# plt.show()
