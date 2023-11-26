'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math




class CONV(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,in_channels,out_channels):
        super(CONV, self).__init__()
        identity = torch.nn.Identity()
        self.root = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding=1)),
            # ('gn', nn.GroupNorm(out_channels // 2, out_channels, eps=1e-6)),
            ('bn',nn.BatchNorm2d(out_channels, eps=1e-6)),
            ('relu', nn.SiLU()),
            # ('Drop', nn.Dropout(0.5)),
        ]))
    def forward(self, x):
        x = self.root(x)
        return x

# channels shuffle增加组间交流
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class DCM(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,in_channels,out_channels,r):
        super(DCM, self).__init__()
        # identity = torch.nn.Identity()
        self.root = nn.Sequential(
            # SoftPool2d(kernel_size=(128, 128), stride=(128, 128)),
            nn.GroupNorm(in_channels//2, in_channels),
            nn.Conv2d(in_channels, in_channels//r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels//r),
            nn.GELU(),
            nn.Conv2d(in_channels//r, out_channels, kernel_size=1, stride=2),
        )
        # self.down =nn.AdaptiveMaxPool2d()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        # self.avg_pool = SoftPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        x = self.root(self.avg_pool(x))
        return x

class MFAM(nn.Module):

    def __init__(self,num_classes):
        super(MFAM, self).__init__()
        self.in_planes = 64
        ####################################


        self.num_classes = num_classes


        # Top layer
        self.toplayer1_1 = CONV(512, 256)
        self.toplayer1_2 = CONV(256, 128)
        self.toplayer1_3 = CONV(128, 128)


        # Lateral layers
        self.latlayer1_1 = CONV(256, 128)
        self.latlayer1_2 = CONV(128, 128)


        self.latlayer2 = CONV(128, 128)



        self.conv3 = nn.Conv2d(128 * 3, self.num_classes, kernel_size=1, stride=1,padding=0)

        ######################################
        self.bian1 = DCM(512,128,8)
        self.bian2 = DCM(256,128,4)
        self.bian3 = DCM(128,128,2)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)+y

    def forward(self, low_level_features):

        c3 = low_level_features[2] #torch.Size([16, 128, 64, 64])
        c2 = low_level_features[1] #torch.Size([16, 256, 32, 32])
        c1 = low_level_features[0] #torch.Size([16, 512, 16, 16])


        _,_,_,H1= c2.size()
        # Top-down
        p5_1 = F.interpolate(self.toplayer1_1(c1),size=(H1, H1), mode='bilinear', align_corners=True)

        _, _, _, H2 = c3.size()
        p5_2 = F.interpolate(self.toplayer1_2(p5_1),size=(H2, H2), mode='bilinear', align_corners=True)

        H3 = 128
        p1 = F.interpolate(self.toplayer1_3(p5_2),size=(H3, H3), mode='bilinear', align_corners=True)

        add = self.bian1(c1)
        p1 = p1+add

        _, _, _, H2 = c3.size()
        p4_1 = F.interpolate(self.latlayer1_1(c2),size=(H2, H2), mode='bilinear', align_corners=True)
        H3 = 128
        p2 = F.interpolate(self.latlayer1_2(p4_1), size=(H3, H3), mode='bilinear', align_corners=True)

        add = self.bian2(c2)
        p2 = p2+add

        H3 = 128
        p3 = F.interpolate(self.latlayer2(c3), size=(H3, H3), mode='bilinear', align_corners=True)
        add = self.bian3(c3)
        p3 = p3+add


        sum = torch.cat([p1,p2,p3],dim=1)

        return self.up(self.conv3(sum))



class FPN(nn.Module):

    def __init__(self,num_classes):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)


        # Top layer
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        # Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128)
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, low_level_features):
        #high-low
        # c2 = low_level_features[3] #torch.Size([16, 64, 128, 128])
        c3 = low_level_features[2] #torch.Size([16, 128, 64, 64])
        c4 = low_level_features[1] #torch.Size([16, 256, 32, 32])
        c1 = low_level_features[0] #torch.Size([16, 512, 16, 16])


        # Top-down
        p5 = self.toplayer(c1)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        # p2 = self.smooth3(p2)

        # Semantic
        h, w = 128,128
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        s5 = F.relu(self.gn2(self.conv2(s5)))
        # 256->128
        s5 =F.relu(self.gn1(self.semantic_branch(s5)))

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = F.relu(self.gn1(self.semantic_branch(s4)))

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        # s2 = F.relu(self.gn1(self.semantic_branch(p2)))
        return self._upsample(self.conv3(s3 + s4 + s5), 2 * h, 2 * w)


if __name__ == "__main__":
    model = FCM([2, 4, 23, 3], 32, back_bone="resnet")
    input = torch.rand(1, 3, 512, 1024)
    output = model(input)
    print(output.size())