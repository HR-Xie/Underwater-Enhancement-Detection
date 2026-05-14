import torch
import time
from thop import profile
import torch.nn as nn
from models import *
import copy


# ==========================================
# 这里放你自己的模型结构，或者从你的文件导入
# 例如: from your_model_file import UWModel1, UWModel2
# ==========================================

# 这是一个示例的 Dummy Model，用来测试代码能否跑通
class DummyUnderwaterModel(nn.Module):
    def __init__(self):
        super(DummyUnderwaterModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


def evaluate_model(model, input_size=(1, 3, 256, 256), device='cuda'):
    """
    评估模型的 Params, GFLOPs 和 FPS
    :param model: PyTorch 模型
    :param input_size: 输入张量的维度 (Batch_size, Channels, Height, Width)
    :param device: 运行设备 ('cuda' 或 'cpu')
    """
    model = model.to(device)
    model.eval()

    # 1. 创建虚拟输入
    dummy_input = torch.randn(input_size).to(device)

    # ================= 修改的部分开始 =================
    # 2. 计算 Params 和 GFLOPs (使用 thop)
    # 制作一个模型的深拷贝，专门给 thop 霍霍，避免它的残留 hook 污染原模型
    model_for_profile = copy.deepcopy(model)

    # 注意: thop 默认计算的是 MACs (乘加操作数)。通常 1 MAC ≈ 2 FLOPs
    macs, params = profile(model_for_profile, inputs=(dummy_input,), verbose=False)
    gflops = (macs * 2) / 1e9  # 转换为 GFLOPs
    params_m = params / 1e6  # 转换为 M (百万) 级别参数量

    print(f"--- 模型评估结果 ---")
    print(f"输入分辨率: {input_size[2]}x{input_size[3]}")
    print(f"参数量 (Params): {params_m:.3f} M")
    print(f"计算量 (GFLOPs): {gflops:.3f} G")

    # 删掉替身，释放显存
    del model_for_profile
    # ================= 修改的部分结束 =================

    # 3. 计算 FPS (测试推理耗时) - 这里使用的是原始的、没被污染的 model
    if device == 'cuda':
        print("正在进行 GPU 预热，请稍候...")
        # GPU 预热 (Warm-up) - 避免首次推理初始化带来的时间误差
        for _ in range(50):
            _ = model(dummy_input)

        torch.cuda.synchronize()  # 等待所有 GPU 任务完成

        # 开始计时
        iterations = 100  # 循环测试 100 次取平均值
        start_time = time.time()

        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)

        torch.cuda.synchronize()  # 再次等待所有 GPU 任务完成
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_image = total_time / iterations
        fps = 1.0 / avg_time_per_image

        print(f"平均单张耗时: {avg_time_per_image * 1000:.2f} ms")
        print(f"推理帧率 (FPS): {fps:.2f} frames/s")
    else:
        print("注意: 建议在 CUDA (GPU) 环境下测试 FPS 以获得准确的工业界标准性能数据。")
    print("-" * 20 + "\n")


if __name__ == "__main__":
    # 检查是否有 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 实例化模型
    model = LightAquaNet()

    # 水下图像增强的常用测试分辨率有 256x256, 512x512, 或高清分辨率等
    # 请根据你论文或实际使用情况修改尺寸
    test_resolution = (1, 3, 256, 256)

    print("正在评估 LightAquaNet ...")
    evaluate_model(model, input_size=test_resolution, device=device)
