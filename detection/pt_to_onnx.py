import torch
import os
import sys

# Add project root for imports from detection/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'detection'))

from ultralytics import YOLO
from WIModel import LightAquaNet

# Resolve paths relative to this script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS_DIR = os.path.join(_BASE_DIR, '..', 'weights')


def export_yolo_to_onnx():
    print("⏳ [1/2] 开始转换 YOLO 目标检测模型...")
    yolo_path = os.path.join(_WEIGHTS_DIR, 'best.pt')

    if not os.path.exists(yolo_path):
        print(f"❌ 找不到 YOLO 权重文件，请检查路径: {yolo_path}")
        return

    # 加载模型并导出
    model = YOLO(yolo_path)
    # format='onnx' 指定导出格式，dynamic=True 允许推理时输入不同尺寸的图像
    model.export(format='onnx', dynamic=True)
    print("✅ YOLO 转换完成！默认保存在原 .pt 文件同级目录下 (best.onnx)\n")


def export_lightaquanet_to_onnx():
    print("⏳ [2/2] 开始转换 LightAquaNet 水下增强模型...")
    weight_path = os.path.join(_WEIGHTS_DIR, 'UW_epoch_277.pth')

    if not os.path.exists(weight_path):
        print(f"❌ 找不到 LightAquaNet 权重文件: {weight_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并加载权重
    model = LightAquaNet().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 伪造一个占位输入张量 (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, 3, 480, 640).to(device)
    output_path = os.path.join(_WEIGHTS_DIR, 'LightAquaNet.onnx')

    print("   正在进行计算图追踪和常量折叠优化，请稍候...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                          'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
    print(f"✅ LightAquaNet 转换完成！已保存至: {output_path}\n")


if __name__ == "__main__":
    print("🚀 开始批量转换为 ONNX 格式...\n")

    export_yolo_to_onnx()
    export_lightaquanet_to_onnx()

    print("🎉 所有模型转换任务已结束！可以去检查生成的 .onnx 文件了。")