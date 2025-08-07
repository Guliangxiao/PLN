import torch
import torch.nn as nn
import torchvision

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

def ConvBNReLUFactorization(in_channels,out_channels,kernel_sizes,paddings):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes, stride=1,padding=paddings),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)#1*1

        self.branch2 = nn.Sequential(#1*1后3*3
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),#1*1卷积减少输入层数
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(#1*1后5*5
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=out_channels3, out_channels=out_channels3, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(#池化后1*1
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV2ModuleB(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2reduce, kernel_sizes=(1,3),paddings=(0,1)),
            ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(3,1),paddings=(1, 0)),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=(1,3), paddings=(0,1)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce,kernel_sizes=(3,1), paddings=(1, 0)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3reduce, kernel_sizes=(1,3), paddings=(0,1)),
            ConvBNReLUFactorization(in_channels=out_channels3reduce, out_channels=out_channels3,kernel_sizes=(3,1), paddings=(1, 0)),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV2ModuleC(nn.Module):#c的输出频道为out1+2*out2+2*out3+out4
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(1,3),paddings=(0,1))
        self.branch2_conv2b = ConvBNReLUFactorization(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_sizes=(3,1),paddings=(1,0))

        self.branch3_conv1 = ConvBNReLU(in_channels=in_channels,out_channels=out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(in_channels=out_channels3reduce, out_channels=out_channels3, kernel_size=3,stride=1,padding=1)
        self.branch3_conv3a = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=(3, 1),paddings=(1,0))
        self.branch3_conv3b = ConvBNReLUFactorization(in_channels=out_channels3, out_channels=out_channels3, kernel_sizes=(1,3),paddings=(0,1))

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels4, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV2ModuleD(nn.Module):#缩减一半
    def __init__(self, in_channels,out_channels1reduce,out_channels1,out_channels2reduce, out_channels2):
        super(InceptionV2ModuleD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels1reduce, out_channels=out_channels1, kernel_size=3,stride=2,padding=1)
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels2reduce, kernel_size=1),
            ConvBNReLU(in_channels=out_channels2reduce, out_channels=out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=out_channels2, out_channels=out_channels2, kernel_size=3, stride=2,padding=1),
        )

        self.branch3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out




class InceptionV2(nn.Module):
    def __init__(self):
        super(InceptionV2, self).__init__()

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.block3 = nn.Sequential(
            InceptionV2ModuleA(in_channels=192, out_channels1=64, out_channels2reduce=64, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV2ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=64, out_channels2=96, out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV2ModuleD(in_channels=320, out_channels1reduce=128, out_channels1=160, out_channels2reduce=64, out_channels2=96),
        )

        self.block4 = nn.Sequential(
            InceptionV2ModuleB(in_channels=576, out_channels1=224, out_channels2reduce=64, out_channels2=96, out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=192, out_channels2reduce=96, out_channels2=128, out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=160, out_channels2reduce=128, out_channels2=160, out_channels3reduce=128, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=96, out_channels2reduce=128, out_channels2=192, out_channels3reduce=160, out_channels3=160, out_channels4=128),
            InceptionV2ModuleD(in_channels=576, out_channels1reduce=128, out_channels1=192, out_channels2reduce=192, out_channels2=256),
        )

        self.block5 = nn.Sequential(
            InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128),
            #InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160, out_channels3reduce=160, out_channels3=112, out_channels4=128),

            InceptionV2ModuleC(in_channels=1024, out_channels1=816, out_channels2reduce=800, out_channels2=720, out_channels3reduce=300, out_channels3=240, out_channels4=592),
        )#输出为816+2*720+2*240+592=3328
        self.block6 = nn.Sequential(
            nn.Conv2d(3328,1536,1,1),
            nn.Conv2d(1536,1536,3,1,1),
            nn.Conv2d(1536,1536,3,1,1)
        )#公用的卷积
        self.block71 = nn.Sequential(
            nn.Conv2d(1536,1536,3,1,1),
            nn.Conv2d(1536,204,3,1,1),#前面两步是卷积
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,1,1),
#三次扩张卷积，说实话我感觉做不到扩张率为16。。。
        )
        self.block72 = nn.Sequential(
            nn.Conv2d(1536,1536,3,1,1),
            nn.Conv2d(1536,204,3,1,1),#前面两步是卷积
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,1,1),
#三次扩张卷积，说实话我感觉做不到扩张率为16。。。
        )
        self.block73 = nn.Sequential(
            nn.Conv2d(1536,1536,3,1,1),
            nn.Conv2d(1536,204,3,1,1),#前面两步是卷积
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,1,1),
#三次扩张卷积，说实话我感觉做不到扩张率为16。。。
        )
        self.block74 = nn.Sequential(
            nn.Conv2d(1536,1536,3,1,1),
            nn.Conv2d(1536,204,3,1,1),#前面两步是卷积
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,2,2),
            nn.Conv2d(204,204,3,1,1,1),
#三次扩张卷积，说实话我感觉做不到扩张率为16。。。
        )

    def softmax(self,x):
        x = x.view(-1,4,51,14,14)
        x1 = x[:,:,3:17,:,:]
        x1 = nn.functional.softmax(x1,dim=2)
        x[:,:,3:17,:,:]=x1#这里是对20种物体概率进行归一化

        x1 = x[:,:,17:31,:,:]
        x1 = nn.functional.softmax(x1,dim=2)
        x[:,:,17:31,:,:]=x1#这里是对lx物体概率进行归一化

        x1 = x[:,:,31:51,:,:]
        x1 = nn.functional.softmax(x1,dim=2)
        x[:,:,31:51,:,:]=x1#这里是对lx物体概率进行归一化
        x = x.view(-1,204,14,14)
        return x
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)#前面五个属于基本的inceptionv2，接下来依次添加四条分支的卷积，扩张，推理与nms
        x = self.block6(x)
        out1 = self.block71(x)
        out1 = self.softmax(out1)
        out2 = self.block72(x)
        out2 = self.softmax(out2)
        out3 = self.block73(x)
        out3 = self.softmax(out3)
        out4 = self.block74(x)
        out4 = self.softmax(out4)#这里输出通道为204，预测为4*51，4为2B，（51为1+20+1+1+14+14，分别为存在概率，20个种类概率，xy坐标，两个链接向量）
        x = torch.stack([out1,out2,out3,out4],dim=1)
        x = x.permute(1,0,2,3,4)
        return x
    

