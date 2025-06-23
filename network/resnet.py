import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from dropblock import DropBlock2D
from opts import parse_opts
args = parse_opts()
AT = args.AT
DB = args.DB#DropBlock
GD = args.GD#固定图像层

checkpoint_path = './resNet50_AffectNet.pth'
# checkpoint_path = './resnet50_ImageNet.pth'

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):#
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#(1,1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()#源码就有
        )

    def forward(self,x): # torch.Size([1, 256, 64, 64])
        b,c,_,_ = x.size()
        x1 = self.avg_pool(x) # 全局平均池化[b,c,h,w] => torch.Size([1, 256, 1, 1])
        print("x1",x1.size())
        x2 = x1.view(b,c) # 调整维度[b,c,1,1] =>torch.Size([1, 256])
        print("x2",x2.size())
        y1 = self.fc(x2) # 模型，torch.Size([1, 256])
        print("y1",y1.size())
        y2 = y1.view(b,c,1,1)  # 调整维度torch.Size([1, 256]) =》torch.Size([1, 256, 1, 1])
        print("y2", y2.size())
        # y3 = y2.expand_as(x)#扩展张量中某维数据的尺寸，括号内的输入参数是另一个张量，作用是将输入tensor的维度扩展为与指定tensor相同的size。
        # print(y3)
        y4 = x * y2#y3？？？？
        # print(y4.shape) # torch.Size([1, 256, 64, 64])
        return y4

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#平均池化torch.Size([B, 256, 224, 224])->[B, 256, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)#最大池化

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)#
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)#输入通道2 输出通道1  卷积核大小(kernel_size)，卷积步长 (stride)，特征图填充宽度 (padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)#torch.Size([B, 256, 224, 224])->[B, x, y]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width,track_running_stats=True)#norm_layer
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width,track_running_stats=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d#
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes,track_running_stats=True)#   affine=False,momentum=0.9,eps=1e-4,
        self.relu = nn.ReLU(inplace=True)

        # 网络的第一层加入注意力机制
        # self.se = SELayer(planes*self.expansion, 16)######加入注意力机制reduction=16
        self.ca = ChannelAttention(self.inplanes)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 网络的卷积层的最后一层加入注意力机制
        # self.se = SELayer(planes*self.expansion, 16)######加入注意力机制reduction=16
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.drop_block = DropBlock2D(block_size=7, drop_prob=0.3)  # block_size为所有的特征图设置一个恒定的块大小

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion,track_running_stats=True),############################
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print(("输入x", x.size()))#torch.Size([256, 3, 171, 171]))
        # print(x.size())#([8, 3, 224, 224])
        x = self.conv1(x)#[8, 64, 112, 112])
        # print(x.size())#([8, 256, 56, 56])
        x = self.bn1(x)
        x = self.relu(x)
        if AT:
            # x = self.se(x)  ###SE 通道
            x = self.ca(x) * x#cbam  通道
            # print(("ca", x.size()))
            x = self.sa(x) * x# 空间
            # print(("sa", x.size()))

        x = self.maxpool(x)
        x = self.layer1(x)
        # print("x1",x.size())
        x = self.layer2(x)
        # print("x2", x.size())
        if DB:
            x = self.drop_block(x)
        x = self.layer3(x)
        # print("x3",x.size())
        if DB:
            x = self.drop_block(x)

        x = self.layer4(x)
        # print("x4",x.size())#([8, 2048, 7, 7])
        x = self.avgpool(x)
        # print("xavg",x.size())
        x = torch.flatten(x, 1)#将张量拉成一维的向量（B，2048）
        # print("xflatten",x.size())
        # x = self.fc(x)#不要原fc层

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('加载预训练模型:',checkpoint_path)
        model_dict = torch.load(checkpoint_path)
        # for key in model_dict:
        #     print(key)
        s_dict = model.state_dict()
        # print(model_dict["linear.bias"])
        # 筛选掉不需要的网络层
        for name in s_dict:
            # print(name)
            if name not in model_dict:# ResNet网络中没有fc层
                print(name,"预训练模型没有这个网络层")#预训练模型有sa ca层
                continue
            s_dict[name] = model_dict[name]
        model.load_state_dict(s_dict)
    if GD:
        print("固定网络层参数")
        i = 0
        # ====== 固定除了注意力的其他卷积层 ======
        for (name, param) in model.named_parameters():
            if i < 132:#+57
                param.requires_grad = False#固定
            else:
                pass
            i = i+1
        # print(i)#162层
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:

    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,**kwargs)



if __name__ == "__main__":
    import torch
    inputs = torch.rand(8, 512, 224, 224)#6维
    net = ChannelAttention(in_planes=512)
    outputs = net(inputs)
    print(outputs.size())