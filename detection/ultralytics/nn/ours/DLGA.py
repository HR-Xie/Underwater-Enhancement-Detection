import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['DLGA', 'DLG']

class DLG(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1):
        super(DLG, self).__init__()
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # 确保整除
        k = t if t % 2 else t + 1

        # 卷积层保持不变，但添加实例归一化
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.norm_global = nn.InstanceNorm1d(1, affine=False)
        self.norm_local = nn.InstanceNorm1d(1, affine=False)

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)


        b, c, m, n = x.shape
        c_local = local_arv.size(1)

        # 处理局部特征
        temp_local = local_arv.view(b, c_local, -1).transpose(1, 2).reshape(b, 1, -1)
        y_local = self.norm_local(self.conv_local(temp_local))
        y_local = y_local.view(b, -1, c_local).transpose(1, 2).view(b, c, self.local_size, self.local_size)
        att_local = y_local.sigmoid()

        # 处理全局特征
        temp_global = global_arv.view(b, c, -1).transpose(1, 2)
        y_global = self.norm_global(self.conv(temp_global))
        y_global = y_global.view(b, c, 1, 1)
        att_global = F.adaptive_avg_pool2d(y_global.sigmoid(), [self.local_size, self.local_size])

        # 动态计算融合权重
        alpha = torch.sigmoid(global_arv.mean(dim=(1,2,3), keepdim=True))  # 基于全局特征动态调整
        att_combined = att_global * (1 - alpha) + att_local * alpha

        # 调整注意力图尺寸并应用
        att_all = F.interpolate(att_combined, size=(m, n), mode='bilinear', align_corners=False)
        return x * att_all


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    return p if p is not None else k // 2 if isinstance(k, int) else [x // 2 for x in k]

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DLGA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(FeatureRefinement(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FeatureRefinement(nn.Module):
    """优化后的Bottleneck，集成改进的DLG并保持参数量"""
    def __init__(self, c1, c2, shortcut=True, g=1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = Conv(c2, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.attn = DLG(c2)  # 使用改进的注意力模块

    def forward(self, x):
        return x + self.attn(self.cv2(self.cv1(x))) if self.add else self.attn(self.cv2(self.cv1(x)))

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 64, 16, 16)
    model = DLG(64)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {model(x).shape}")
    print("参数量统计:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # 应保持与原版相同