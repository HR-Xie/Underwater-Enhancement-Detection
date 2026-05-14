import torch.nn as nn
import torch
import torch.nn.functional as F
import numbers
import numpy as np
from einops import rearrange
import copy
import time
from thop import profile
def evaluate_model_performance(model, input_size=(1, 3, 256, 256), device='cuda', num_warmup=50, num_iterations=100):
    """
    综合评估模型的 Params, GFLOPs, Latency, FPS
    :param model: 需要测试的 PyTorch 模型
    :param input_size: 模拟输入的尺寸 (Batch_size, Channels, Height, Width)
    :param device: 测试所在的设备 ('cuda' 或 'cpu')
    :param num_warmup: 预热次数 (避免显卡冷启动导致的时间偏差)
    :param num_iterations: 正式计算延迟的迭代次数
    """
    print(f"\n{'=' * 40}")
    print(f"🚀 开始评估模型性能...")
    print(f"📦 Input size: {input_size}")
    print(f"🖥️  Device: {device}")

    # 将模型移动到对应设备并设置为评估模式
    model = model.to(device)
    model.eval()

    # 构造模拟输入
    dummy_input = torch.randn(*input_size).to(device)

    # ==========================================
    # 1. 计算 Params 和 GFLOPs
    # ==========================================
    # 为了防止 thop 改变原模型状态，深拷贝一个模型到 CPU 进行统计
    model_for_flops = copy.deepcopy(model).cpu()
    dummy_input_cpu = torch.randn(*input_size).cpu()

    try:
        flops, params = profile(model_for_flops, inputs=(dummy_input_cpu,), verbose=False)
        params_m = params / 1e6  # 转换为百万 (M)
        gflops = flops / 1e9  # 转换为十亿 (G)
        print(f"📊 Params:  {params_m:.4f} M")
        print(f"🧮 GFLOPs:  {gflops:.4f} G")
    except Exception as e:
        print(f"⚠️ FLOPs 计算失败: {e}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📊 Params (Manual): {total_params / 1e6:.4f} M")

    # ==========================================
    # 2. 计算 Latency 和 FPS
    # ==========================================
    print("⏳ 正在进行模型预热 (Warm-up)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    print(f"⏱️ 正在测速 (Iterations={num_iterations})...")

    if device == 'cuda' and torch.cuda.is_available():
        # GPU 测速 (必须使用同步)
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        torch.cuda.synchronize()  # 确保所有 GPU 任务完成
        end_time = time.perf_counter()
    else:
        # CPU 测速
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.perf_counter()

    # 计算指标
    total_time = end_time - start_time
    latency_ms = (total_time / num_iterations) * 1000  # 毫秒/帧
    fps = num_iterations / total_time  # 帧/秒

    print(f"⚡ Latency: {latency_ms:.2f} ms")
    print(f"🎬 FPS:     {fps:.2f}")
    print(f"{'=' * 40}\n")

    return params_m, gflops, latency_ms, fps
# ==============================================================================
# 核心组件: 深度可分离卷积 (DWConv) - 降低参数量的关键
# ==============================================================================
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride,
                                   padding=kernel_size // 2, groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# ==============================================================================
# 核心组件: 标准化层 (兼容性修复版)
# ==============================================================================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        # x shape: [N, C, H, W]
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # 兼容性写法，不使用自定义 autograd function
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ==============================================================================
# 核心组件: 大核注意力模块 (LKA Block) - 替代复杂的 SWPSA
# ==============================================================================
class LKALayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. 局部特征提取 (5x5)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 2. 空间长距离依赖 (7x7 dilation 3 => RF 21x21)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 3. 通道混合 (1x1)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class LKABlock(nn.Module):
    def __init__(self, dim, bias=False):
        super(LKABlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LKALayer(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ==============================================================================
# 核心组件: 轻量级门控融合 (Gated Fusion)
# ==============================================================================


# ==============================================================================
# 基础组件
# ==============================================================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * 2.66)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(DWConv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(DWConv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class LutGate(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features * 2)

    def forward(self, x):
        x = self.linear(x)
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NILUT(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=3, out_features=3, res=True, drop=0.):
        super().__init__()
        self.res = res
        self.sg = LutGate(hidden_features)
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(self.sg)
        self.net.append(nn.Dropout(drop))
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.Tanh())
            self.net.append(nn.Dropout(drop))
        self.net.append(nn.Linear(hidden_features, out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, inp):
        original_shape = inp.shape
        inp = inp.view(-1, inp.shape[1])
        output = self.net(inp)
        if self.res:
            output = output + inp
            output = torch.clamp(output, 0., 1.)
        output = output.view(original_shape)
        return output


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(*[NAFBlock(n_feats) for _ in range(2)])

    def forward(self, x):
        return self.body(x) + x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = DWConv(in_size, out_size, kernel_size=3, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = DWConv(out_size, out_size, kernel_size=3, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample_layer = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out

        if self.downsample:
            out_down = self.downsample_layer(out)
            return out_down, out
        else:
            return out


class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(in_size=n_feat + scale_unetfeats * i, out_size=n_feat + scale_unetfeats * (i + 1),
                              downsample=True, relu_slope=0.2, use_csff=csff, use_HIN=True))
        self.body.append(UNetConvBlock(in_size=n_feat + scale_unetfeats * (depth - 1),
                                       out_size=n_feat + scale_unetfeats * (depth - 1), downsample=False,
                                       relu_slope=0.2, use_csff=csff, use_HIN=True))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()
        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat + scale_unetfeats * (depth - i - 1),
                                         out_size=n_feat + scale_unetfeats * (depth - i - 2), relu_slope=0.2))
            self.skip_conv.append(
                nn.Conv2d(n_feat + scale_unetfeats * (depth - i - 1), n_feat + scale_unetfeats * (depth - i - 2), 3, 1,
                          1))

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i - 1]))
            res.append(x)
        return res


# ==============================================================================
# 主模型: LightAquaNet (DGUNet + NILUT + LKA + DWConv)
# ==============================================================================

# ==============================================================================
# 核心组件: 轻量级门控融合 (加入消融开关)
# ==============================================================================
class EfficientMergeBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=False, use_gate=True):
        super(EfficientMergeBlock, self).__init__()
        self.use_gate = use_gate

        if self.use_gate:
            self.att = nn.Sequential(
                nn.Conv2d(n_feat * 2, n_feat, 1, bias=bias),
                nn.ReLU(True),
                nn.Conv2d(n_feat, n_feat, 1, bias=bias),
                nn.Sigmoid()
            )
        self.conv_out = DWConv(n_feat * 2, n_feat, kernel_size, bias=bias)

    def forward(self, x, bridge):
        if self.use_gate:
            combined = torch.cat([x, bridge], dim=1)
            gate = self.att(combined)
            bridge_gated = bridge * gate
            out = torch.cat([x, bridge_gated], dim=1)
        else:
            # 消融实验：退化为最普通的 Concat
            out = torch.cat([x, bridge], dim=1)

        out = self.conv_out(out)
        return out + x


# ==============================================================================
# 核心组件: 大核注意力模块 (加入消融开关)
# ==============================================================================
class EfficientSAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, use_lka=True):
        super(EfficientSAM, self).__init__()
        self.use_lka = use_lka

        if self.use_lka:
            self.conv1 = LKABlock(n_feat, bias=bias)
        else:
            # 消融实验：退化为 1x1 卷积，不使用 LKA 空间注意力
            self.conv1 = nn.Conv2d(n_feat, n_feat, 1, bias=bias)

        self.conv2 = DWConv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img


# ==============================================================================
# 主模型: LightAquaNet (加入总体消融开关)
# ==============================================================================
class LightAquaNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, num_cab=8, kernel_size=3, reduction=4,
                 bias=False, use_prior=True, use_lka=True, use_gate=True):
        super(LightAquaNet, self).__init__()

        # 记录消融实验配置
        self.use_prior = use_prior

        act = nn.PReLU()

        if self.use_prior:
            self.lut = NILUT()
            self.output1 = nn.Conv2d(3, 3, 3, 1, padding=1)

        self.shallow_feat1 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat6 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat7 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))

        self.phi_0 = ResBlock(3)
        self.phit_0 = ResBlock(3)
        self.phi_5 = ResBlock(3)
        self.phit_5 = ResBlock(3)
        self.phi_6 = ResBlock(3)
        self.phit_6 = ResBlock(3)

        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r5 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)

        self.stage6_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
        self.stage6_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)

        # 传递消融开关给子模块
        self.sam12 = EfficientSAM(n_feat, kernel_size=1, bias=bias, use_lka=use_lka)
        self.merge56 = EfficientMergeBlock(n_feat, 3, bias, use_gate=use_gate)
        self.sam67 = EfficientSAM(n_feat, kernel_size=1, bias=bias, use_lka=use_lka)
        self.merge67 = EfficientMergeBlock(n_feat, 3, bias, use_gate=use_gate)

        self.tail = DWConv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, img):
        _, _, h, w = img.size()
        factor = 16
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor

        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        # --- Prior Guided (消融控制) ---
        if self.use_prior:
            tone_x = self.lut(img)
            prior1 = self.output1(tone_x)
        else:
            prior1 = 0.0  # 切断颜色先验

        # --- Stage 1 ---
        phixsy_1 = self.phi_0(img) - img + prior1
        x1_img = img - self.r0 * self.phit_0(phixsy_1)
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # --- Stage 6 ---
        phixsy_6 = self.phi_5(stage1_img) - img + prior1
        x6_img = stage1_img - self.r5 * self.phit_5(phixsy_6)
        x6 = self.shallow_feat6(x6_img)
        x6_cat = self.merge56(x6, x2_samfeats)
        feat6, feat_fin6 = self.stage6_encoder(x6_cat, feat1, res1)
        res6 = self.stage6_decoder(feat_fin6, feat6)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        # --- Stage 7 ---
        phixsy_7 = self.phi_6(stage6_img) - img + prior1
        x7_img = stage6_img - self.r6 * self.phit_6(phixsy_7)
        x7 = self.shallow_feat7(x7_img)
        x7_cat = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat) + img

        if pad_h > 0 or pad_w > 0:
            stage7_img = stage7_img[:, :, :h, :w]
            stage6_img = stage6_img[:, :, :h, :w]
            stage1_img = stage1_img[:, :, :h, :w]

        return [stage7_img, stage6_img, stage1_img]


# ... (此处省略基础组件代码如 DWConv, LayerNorm2d, LKABlock, NAFBlock 等，保持原样即可) ...
# 为了展示清晰，直接从高效融合模块开始展示重构部分

# ==============================================================================
# 核心组件: 轻量级门控融合 (加入消融开关)
# [AI Diagram: Block 'EfficientMergeBlock']
# [Flow: Input(x) & Input(bridge) -> Concat -> Attention Gate -> Multiply with bridge -> Concat with x -> DWConv -> Add x]
# # ==============================================================================
# class EfficientMergeBlock(nn.Module):
#     def __init__(self, n_feat, kernel_size=3, bias=False, use_gate=True):
#         super(EfficientMergeBlock, self).__init__()
#         self.use_gate = use_gate
#
#         if self.use_gate:
#             self.att = nn.Sequential(
#                 nn.Conv2d(n_feat * 2, n_feat, 1, bias=bias),
#                 nn.ReLU(True),
#                 nn.Conv2d(n_feat, n_feat, 1, bias=bias),
#                 nn.Sigmoid()
#             )
#         self.conv_out = DWConv(n_feat * 2, n_feat, kernel_size, bias=bias)
#
#     def forward(self, x, bridge):
#         if self.use_gate:
#             combined = torch.cat([x, bridge], dim=1)
#             gate = self.att(combined)
#             bridge_gated = bridge * gate
#             out = torch.cat([x, bridge_gated], dim=1)
#         else:
#             # 消融实验：退化为最普通的 Concat
#             out = torch.cat([x, bridge], dim=1)
#
#         out = self.conv_out(out)
#         return out + x
#
#
# # ==============================================================================
# # 核心组件: 大核注意力模块 (加入消融开关)
# # [AI Diagram: Block 'EfficientSAM' (Spatial Attention Module)]
# # [Flow: Input(x) -> LKA_Block -> Output(Features)]
# # [Flow: Input(x) -> DWConv -> Add(x_img) -> Output(Image)]
# # ==============================================================================
# class EfficientSAM(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias, use_lka=True):
#         super(EfficientSAM, self).__init__()
#         self.use_lka = use_lka
#
#         if self.use_lka:
#             self.conv1 = LKABlock(n_feat, bias=bias)
#         else:
#             # 消融实验：退化为 1x1 卷积，不使用 LKA 空间注意力
#             self.conv1 = nn.Conv2d(n_feat, n_feat, 1, bias=bias)
#
#         self.conv2 = DWConv(n_feat, 3, kernel_size, bias=bias)
#
#     def forward(self, x, x_img):
#         x1 = self.conv1(x)
#         img = self.conv2(x) + x_img
#         x1 = x1 + x
#         return x1, img
#
#
# # ==============================================================================
# # 主模型: LightAquaNet
# # [AI Diagram: Global Architecture 'LightAquaNet' with 3 Stages]
# # [Style: CVPR style, clean, structured layout with clear skip connections]
# # ==============================================================================
# class LightAquaNet(nn.Module):
#     def __init__(self, in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, num_cab=8, kernel_size=3, reduction=4,
#                  bias=False, use_prior=True, use_lka=True, use_gate=True):
#         super(LightAquaNet, self).__init__()
#
#         self.use_prior = use_prior
#         act = nn.PReLU()
#
#         # [AI Diagram: Component 'Color Prior Guidance']
#         # [Visual: A dedicated box extracting prior P from Input Image using NILUT]
#         if self.use_prior:
#             self.lut = NILUT()
#             self.output1 = nn.Conv2d(3, 3, 3, 1, padding=1)
#
#         # 浅层特征提取 (统一命名为 Stage 1, 2, 3)
#         self.shallow_feat1 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
#         self.shallow_feat2 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
#         self.shallow_feat3 = nn.Sequential(DWConv(3, n_feat, kernel_size, bias=bias),
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
#
#         # 图像先验提取网络 (统一命名为 1, 2, 3)
#         self.phi_1 = ResBlock(3)
#         self.phit_1 = ResBlock(3)
#         self.phi_2 = ResBlock(3)
#         self.phit_2 = ResBlock(3)
#         self.phi_3 = ResBlock(3)
#         self.phit_3 = ResBlock(3)
#
#         # 残差缩放系数 (统一命名为 1, 2, 3)
#         self.r1 = nn.Parameter(torch.Tensor([0.5]))
#         self.r2 = nn.Parameter(torch.Tensor([0.5]))
#         self.r3 = nn.Parameter(torch.Tensor([0.5]))
#
#         # [AI Diagram: Component 'Stage 1 U-Net']
#         self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=False)
#         self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
#
#         # [AI Diagram: Component 'Stage 2 U-Net' - Notice CSFF=True indicating cross-stage feature fusion]
#         self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
#         self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
#
#         # 阶段间连接模块 (统一命名为 1->2 和 2->3 的桥梁)
#         # [AI Diagram: Connections between stages using EfficientSAM and EfficientMergeBlock]
#         self.sam12 = EfficientSAM(n_feat, kernel_size=1, bias=bias, use_lka=use_lka)
#         self.merge12 = EfficientMergeBlock(n_feat, 3, bias, use_gate=use_gate)
#
#         self.sam23 = EfficientSAM(n_feat, kernel_size=1, bias=bias, use_lka=use_lka)
#         self.merge23 = EfficientMergeBlock(n_feat, 3, bias, use_gate=use_gate)
#
#         self.tail = DWConv(n_feat, 3, kernel_size, bias=bias)
#
#     def forward(self, img):
#         _, _, h, w = img.size()
#         factor = 16
#         pad_h = (factor - h % factor) % factor
#         pad_w = (factor - w % factor) % factor
#
#         if pad_h > 0 or pad_w > 0:
#             img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
#
#         # ==========================================================
#         # Prior Guided (消融控制)
#         # ==========================================================
#         if self.use_prior:
#             tone_x = self.lut(img)
#             prior_color = self.output1(tone_x)
#         else:
#             prior_color = 0.0
#
#             # ==========================================================
#         # Stage 1
#         # [AI Diagram: Flow 'Stage 1']
#         # [Nodes: Image -> Prior Update -> ShallowFeat -> Encoder -> Decoder -> SAM]
#         # ==========================================================
#         phixsy_1 = self.phi_1(img) - img + prior_color
#         x1_img = img - self.r1 * self.phit_1(phixsy_1)
#
#         x1 = self.shallow_feat1(x1_img)
#         feat1, feat_fin1 = self.stage1_encoder(x1)
#         res1 = self.stage1_decoder(feat_fin1, feat1)
#
#         # 提取 Stage 1 的增强图像和传递给 Stage 2 的特征
#         x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)
#
#         # ==========================================================
#         # Stage 2
#         # [AI Diagram: Flow 'Stage 2']
#         # [Nodes: Stage1_Image -> Prior Update -> ShallowFeat -> Merge(with SAM12_Feats) -> Encoder(CSFF with Stage1) -> Decoder -> SAM]
#         # ==========================================================
#         phixsy_2 = self.phi_2(stage1_img) - img + prior_color
#         x2_img = stage1_img - self.r2 * self.phit_2(phixsy_2)
#
#         x2 = self.shallow_feat2(x2_img)
#         x2_cat = self.merge12(x2, x2_samfeats)  # 融合本阶段浅层特征与上一阶段深层特征
#
#         # CSFF (Cross Stage Feature Fusion): 融合阶段1的编码器特征(feat1)和解码器特征(res1)
#         feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
#         res2 = self.stage2_decoder(feat_fin2, feat2)
#
#         # 提取 Stage 2 的增强图像和传递给 Stage 3 的特征
#         x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
#
#         # ==========================================================
#         # Stage 3
#         # [AI Diagram: Flow 'Stage 3' (Lightweight Refinement)]
#         # [Nodes: Stage2_Image -> Prior Update -> ShallowFeat -> Merge(with SAM23_Feats) -> Tail Conv -> Output Image]
#         # ==========================================================
#         phixsy_3 = self.phi_3(stage2_img) - img + prior_color
#         x3_img = stage2_img - self.r3 * self.phit_3(phixsy_3)
#
#         x3 = self.shallow_feat3(x3_img)
#         x3_cat = self.merge23(x3, x3_samfeats)
#
#         stage3_img = self.tail(x3_cat) + img
#
#         if pad_h > 0 or pad_w > 0:
#             stage3_img = stage3_img[:, :, :h, :w]
#             stage2_img = stage2_img[:, :, :h, :w]
#             stage1_img = stage1_img[:, :, :h, :w]
#
#         # 返回三个阶段的图像结果，便于计算多阶段 Loss
#         return [stage3_img, stage2_img, stage1_img]


if __name__ == '__main__':
    # print("Initializing LightAquaNet...")
    # model = LightAquaNet()
    #
    # # 模拟输入
    # x = torch.randn(1, 3, 256, 256)
    #
    # # ==========================================
    # # 1. 先验证前向传播 (避免 thop 污染模型状态)
    # # ==========================================
    # try:
    #     y = model(x)
    #     print(f"Output shapes: {[out.shape for out in y]}")
    #     print("Forward pass successful! ✅")
    # except Exception as e:
    #     print(f"Forward pass failed: {e}")
    #
    # # ==========================================
    # # 2. 再使用 thop 计算 FLOPs 和 Params
    # # ==========================================
    # try:
    #     from thop import profile
    #     import copy
    #
    #     # 为了绝对安全，深拷贝一个模型专门给 thop 霍霍
    #     model_for_flops = copy.deepcopy(model)
    #
    #     flops, params = profile(model_for_flops, inputs=(x,), verbose=False)
    #     print("\n" + "=" * 40)
    #     print(f"Total Parameters: {params / 1e6:.4f} M")  # 转换为百万 (Million)
    #     print(f"Total GFLOPs:     {flops / 1e9:.4f} G")  # 转换为十亿 (Giga)
    #     print("=" * 40)
    # except ImportError:
    #     print("【提示】请安装 thop 库来精确计算 FLOPs: pip install thop")
    #     print("正在进行简易参数计数...")
    #     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"Total Parameters: {total_params / 1e6:.4f} M")
    # except Exception as e:
    #     print(f"计算 FLOPs 时发生错误: {e}")
    # -----------------------------
    # 示例 1：测试你的 LightAquaNet (图像增强)
    # -----------------------------


    print("--- 正在加载 LightAquaNet ---")
    net_enhance = LightAquaNet()
    # 图像增强常用输入尺寸 256x256
    evaluate_model_performance(net_enhance, input_size=(1, 3, 256, 256), device='cuda')