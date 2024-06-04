import torch
from torch import nn
import torch.nn.functional as F


class LAMME(nn.Module):
    def __init__(self, net, n_segment=8):
        super(LAMME, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.in_channels = net.in_channels
        self.out_channels = net.out_channels
        self.LAM = LAM(in_channels=self.out_channels, n_segment=n_segment)
        self.ME = ME(n_segment=self.n_segment)

        # self.me = ops.action.ME(in_channels=self.out_channels)
        print("LAMME==>")

    def forward(self, x):
        out = self.net(x)
        out = self.LAM(out)
        out = self.ME(out) * out
        print(out.size())
        # out = self.me(out) * out + out
        return out


class ME(nn.Module):
    """
    Motion exexcitation
    Constructs a ME module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3, n_segment=8):
        super(ME, self).__init__()
        self.n_segment = n_segment
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        print("Using ME===> kernel_size={}".format(k_size))

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x3_plus0, _ = x.view(n_batch, self.n_segment, c, h, w).split([self.n_segment - 1, 1], dim=1)
        x3_plus1 = x

        _, x3_plus1 = x3_plus1.view(n_batch, self.n_segment, c, h, w).split([1, self.n_segment - 1], dim=1)
        diff = x3_plus1 - x3_plus0
        diff = F.pad(diff, self.pad, mode="constant", value=0)
        diff = diff.view(nt, c, h, w)

        y = self.avg_pool(diff)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # y = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return y.expand_as(x)


class LAM(nn.Module):
    #    Long-term Time Sequence Aggregation Module
    def __init__(self,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 init_std=0.001):
        super(LAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.GlobalPool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0)
        self.MLP = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2),
            nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False),
            nn.Softmax(-1))

        print("Using LAM==>")

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        x_g = self.GlobalPool(new_x).view(n_batch, c, 1)
        x_g = self.conv(x_g)
        x_g = x_g.view(n_batch, c, 1, 1, 1)
        new_x = new_x + x_g

        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t).contiguous()
        weight = self.MLP(out)
        weight = weight.view(n_batch * c, 1, -1, 1)

        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       weight=weight,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)

        out = out.view(n_batch, c, t, h, w).contiguous()
        out = out.permute(0, 2, 1, 3, 4).contiguous().view(nt, c, h, w)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        print("Using SE==>")

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

        print("Using CBAM==>")
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        print("Using ECA==>")
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)