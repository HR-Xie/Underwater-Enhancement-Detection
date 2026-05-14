import torch
import torch.nn as nn

__all__ = ['C2f_PStar', 'LEMA','SimAM']


# ==========================================
# 1. 核心提纯引擎：PStar_Bottleneck
# ==========================================
class PStar_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5, use_star=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

        # PConv 局部空间解耦
        self.dim_conv = c_ // 4
        self.dim_untouched = c_ - self.dim_conv
        self.pconv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)
        self.bn_pconv = nn.BatchNorm2d(self.dim_conv)

        self.use_star = use_star
        if self.use_star:
            # Star 双子空间映射
            self.f1 = nn.Conv2d(self.dim_conv, self.dim_conv, 1, 1, bias=False)
            self.f2 = nn.Conv2d(self.dim_conv, self.dim_conv, 1, 1, bias=False)

        self.cv2 = nn.Conv2d(c_, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        x1, x2 = torch.split(out, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.act(self.bn_pconv(self.pconv(x1)))

        if self.use_star:
            x1 = self.f1(x1) * self.f2(x1)

        out = self.bn2(self.cv2(torch.cat((x1, x2), 1)))
        return x + out if self.add else out


# ==========================================
# 2. 颈部提纯架构：C2f_PStar
# ==========================================
class C2f_PStar(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.m = nn.ModuleList(PStar_Bottleneck(self.c, self.c, shortcut, e=1.0, use_star=True) for _ in range(n))

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.act(self.bn2(self.cv2(torch.cat(y, 1))))


# ==========================================
# 3. 宏观漏检雷达：LEMA (Lightweight EMA)
# ==========================================
class LEMA(nn.Module):
    def __init__(self, c1, c2, factor=8):
        super(LEMA, self).__init__()
        assert c1 == c2
        self.groups = factor
        assert c1 // self.groups > 0

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        self.conv1x1 = nn.Conv2d(c1 // self.groups, c1 // self.groups, kernel_size=1, stride=1, padding=0)

        # 深度可分离卷积 (极致轻量化核心)
        self.conv3x3 = nn.Conv2d(c1 // self.groups, c1 // self.groups, kernel_size=3, stride=1, padding=1,
                                 groups=c1 // self.groups)

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

class SimAM(nn.Module):
    # 🚀 这里的 c1, c2 是专门给 YOLOv8 底层解析器留的“占位符”，内部不参与参数计算
    def __init__(self, c1, c2, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)