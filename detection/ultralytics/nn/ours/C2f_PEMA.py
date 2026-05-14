import torch
import torch.nn as nn

__all__ = ['C2f_PEMA','C2f_PConv','C2f_EMA']


class EMA(nn.Module):
    """高效多尺度注意力机制 (EMA)"""

    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class PConv(nn.Module):
    """部分卷积 (PConv) - 极限降低参数量"""

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div  # 只对 1/4 通道卷积
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.dim_conv3)
        self.act = nn.SiLU()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.act(self.bn(self.partial_conv3(x1)))
        return torch.cat((x1, x2), 1)


class PEMA_Bottleneck(nn.Module):
    """结合 PConv 与 EMA 的微型瓶颈层"""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.pconv = PConv(c_)
        self.ema = EMA(c_)  # 引入 EMA 进行多尺度特征互动

        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        out = self.ema(self.pconv(out))  # 核心：PConv 降参 + EMA 提效
        out = self.bn2(self.cv2(out))
        return x + out if self.add else out


class C2f_PEMA(nn.Module):
    """你的专属水下轻量化特征融合模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.m = nn.ModuleList(PEMA_Bottleneck(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.act(self.bn2(self.cv2(torch.cat(y, 1))))


# ==========================================
# 以下为消融实验专属模块 (Ablation Modules)
# ==========================================

class PConv_Bottleneck_Only(nn.Module):
    """消融实验 B：只包含 PConv，没有 EMA"""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.pconv = PConv(c_)  # 只有 PConv，没有 EMA 兜底
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        out = self.pconv(out)
        out = self.bn2(self.cv2(out))
        return x + out if self.add else out


class C2f_PConv(nn.Module):
    """消融实验 B 的专属 Neck 模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.m = nn.ModuleList(PConv_Bottleneck_Only(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.act(self.bn2(self.cv2(torch.cat(y, 1))))


class EMA_Bottleneck_Only(nn.Module):
    """消融实验 C：使用标准 3x3 卷积，加上 EMA"""

    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        # 恢复极其耗费参数的标准 3x3 卷积 (不使用 PConv)
        self.cv_standard = nn.Conv2d(c_, c_, 3, 1, 1, bias=False)
        self.bn_s = nn.BatchNorm2d(c_)
        self.ema = EMA(c_)  # 加上 EMA 注意力
        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        out = self.act(self.bn_s(self.cv_standard(out)))  # 标准卷积
        out = self.ema(out)  # 串联 EMA
        out = self.bn2(self.cv2(out))
        return x + out if self.add else out


class C2f_EMA(nn.Module):
    """消融实验 C 的专属 Neck 模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.m = nn.ModuleList(EMA_Bottleneck_Only(self.c, self.c, shortcut, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.act(self.bn2(self.cv2(torch.cat(y, 1))))
