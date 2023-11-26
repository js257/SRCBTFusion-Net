import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath

class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out



class RGM(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,in_channels,out_channels):
        super(RGM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.root = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
    def forward(self, EM,x):
        EM = self.conv1(EM)
        _, _, h, w = x.size()
        EM = F.interpolate(EM, size=(h, w), mode='bilinear', align_corners=True)
        x = torch.cat([x * EM, x], dim=1)
        x = self.root(x)
        return x



class cAttention(nn.Module):  # 通道注意力
    def __init__(self, in_planes=None, ratio=16):
        super(cAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio,1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SIEM(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super(SIEM, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, 1,bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.gelu = nn.ReLU(inplace=True)
        self.bn = nn.GroupNorm(out_channels//2, out_channels, eps=1e-6)
        self.f_conv = DEPTHWISECONV(in_channels, out_channels)
        self.f_conv1 = DEPTHWISECONV(out_channels, out_channels)

        self.sigmoid = nn.Sigmoid()
        self.ac = cAttention(out_channels)
    def forward(self, x):

        x = self.f_conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        h = x
        h = self.ac(h)
        h = self.f_conv1(h)
        h = self.bn(h)
        h = self.gelu(h)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = avg_out+max_out
        sum = x*h
        return self.sigmoid(sum)



class Conv2d_r(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=3,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            # bias=not (use_batchnorm),
        )
        relu =nn.SiLU()

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2d_r, self).__init__(conv, bn, relu)

class MultiReceptiveFieldConvModule(nn.Module):
    def __init__(self, in_channels, out_channels,dp_rate,p,d,a):
        super(MultiReceptiveFieldConvModule, self).__init__()
        out_r = out_channels//16
        # 两个串联的3x3的卷积核
        self.extra_conv1 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[0],dilation=p[0]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[0],dilation=d[0]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[0],dilation=a[0]),
            DropPath(dp_rate)
        )

        self.extra_conv2 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[1], dilation=p[1]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[1], dilation=d[1]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[1], dilation=a[1]),
            DropPath(dp_rate)

        )

        self.extra_conv3 = nn.Sequential(
            Conv2d_r(in_channels, out_r, kernel_size=3, padding=p[2], dilation=p[2]),
            Conv2d_r(out_r, out_r, kernel_size=3, padding=d[2], dilation=d[2]),
            Conv2d_r(out_r, out_channels, kernel_size=3, padding=a[2], dilation=a[2]),
            DropPath(dp_rate)
        )

    def forward(self, x):
        # 第一个3x3卷积
        extra_out1 = self.extra_conv1(x)
        # 两个额外的3x3卷积
        extra_out2 = self.extra_conv2(x+extra_out1)
        extra_out3 = self.extra_conv3(x+extra_out1)
        output = extra_out1+extra_out2+extra_out3

        return output


class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels, out_channels ,hight):
        super(AdaptiveAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义自适应注意力的全连接层
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv1d(hight, hight, kernel_size=3,stride=1,padding=1, bias=False)  # 一维卷积

        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
        )
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1= nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(in_channels,out_channels, kernel_size=3,  bias=False)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        # 计算query、key和value
        query = self.channel_attention(F.adaptive_avg_pool2d(self.conv_3x3(x),[height,1]))
        key = self.channel_attention(F.adaptive_avg_pool2d(x,[1,width]))
        # value = self.value_conv(x)
        # value1 = self.value_conv()

        value1 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//2, width//2]))))     #2,6,8
        value1 = F.interpolate(value1, size=x.size()[2:], mode='bilinear', align_corners=True)

        # value2 = self.value_conv()
        value2 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//6, width//6]))))
        value2 = F.interpolate(value2, size=x.size()[2:], mode='bilinear', align_corners=True)

        # value3 = self.value_conv()
        value3 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//8, width//8]))))
        value3 = F.interpolate(value3, size=x.size()[2:], mode='bilinear', align_corners=True)

        value = torch.cat([value1,value2,value3],dim=1)
        value = self.conv_1x1(value)
        # 计算注意力分数
        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_channels//2, dtype=torch.float32))

        # 计算注意力权重
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # 使用注意力权重对value进行加权平均
        attended_value = torch.matmul(attention_weights, value)
        # attended_value = self.channel(attended_value )
        # attended_value = F.interpolate(attended_value, size=x.size()[2:], mode='bilinear', align_corners=True)
        # 使用注意力机制融合原始输入和加权平均后的值
        output = x + self.gamma * attended_value

        return output




class MASM(nn.Module):
    def __init__(self, in_channels, out_channels,height,p,d,a):
        super(MASM, self).__init__()
        #################################
        #1*1j卷积
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1)
        num_dep = [2, 4, 4, 6, 9, 15, 0]

        rate = [x.item() for x in torch.linspace(0, 0.2, sum(num_dep))]
        self.duo = MultiReceptiveFieldConvModule(in_channels, out_channels,rate[5],p,d,a)

        self.SH = AdaptiveAttention(in_channels,out_channels,height)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, bias=False)  # 一维卷积
    def forward(self, x):
        x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = self.duo(x)
        x3 = self.SH(x)
        x = torch.cat((x1,x2,x3), dim=1)
        x = self.conv2(x)

        return x