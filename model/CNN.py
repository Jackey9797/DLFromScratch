from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, input_size, input_channel, activation='S', alpha=4) -> None:
        super().__init__() 
        if activation == 'S': self.activation = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(*[
            nn.Linear(input_channel, input_channel // alpha), 
            self.activation, 
            nn.Linear(input_channel // alpha, input_channel),
            self.activation, 
        ])

    def forward(self, x): 
        # print(x.shape, self.net(x).shape)
        y = self.pool(x).view(x.shape[0],-1) 
        # print(y.shape)
        y = self.fc(y).view(y.shape[0],x.shape[1], 1, 1) 
        return x * y.expand_as(x)   

class CNN(nn.Module):                    #继承来着nn.Module的父类
    def __init__(self, C_att=False):                    # 初始化网络
        super(CNN, self).__init__()      #super()继承父类的构造函数，多继承需用到super函数
        self.C_att = C_att

        self.conv1 = nn.Conv2d(1, 16, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        if self.C_att: self.C_att1 = ChannelAttention(12, 16)

        self.conv2 = nn.Conv2d(16, 32, 3, bias=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x):            # input(3, 32, 32)        
        x = F.relu(self.conv1(x))    # output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        if self.C_att: x = self.C_att1(x) # = ChannelAttention(12, 16)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x