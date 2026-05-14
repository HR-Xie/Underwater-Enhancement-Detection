import warnings
import sys
import os
import torch

# Add detection module root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 提取你要测试的模型列表（与训练时保持一致）
    yaml_list = [
        'yolov8n.yaml',
        'yolov9t.yaml',
        'yolov9s.yaml',
        'yolov10n.yaml',
        'yolov10s.yaml',
        'yolo11.yaml',
        'yolov12.yaml',
        'Final_Ours.yaml'
    ]

    # 设置 Result 根目录并确保其存在
    result_dir = os.path.join(os.path.dirname(__file__), '..', 'Train', 'Result')
    os.makedirs(result_dir, exist_ok=True)

    # 定义 txt 数据文件路径
    txt_path = os.path.join(result_dir, 'DUO_data.txt')

    # 测试开始前，先初始化 txt 文件并写入表头
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 使用制表符 \t 分隔，方便直接复制到 Excel
        f.write("Model\tLatency_Infer(ms)\tLatency_Total(ms)\tFPS_Infer\tFPS_Total\tPGM(MB)\n")

    for yaml_path in yaml_list:
        model_name = yaml_path.split('.')[0]
        full_model_name = f"DUO-{model_name}"
        print(f"\n\n{'=' * 20} 正在测试模型: {full_model_name} {'=' * 20}")

        # 动态构建权重路径
        weight_path = f'{result_dir}/{full_model_name}/weights/best.pt'

        # 容错处理：如果某个消融模型没训练完找不到权重，直接跳过
        if not os.path.exists(weight_path):
            print(f"⚠️ 警告: 未找到 {full_model_name} 的权重文件，路径: {weight_path}")
            print("跳过该模型，继续测试下一个...")
            continue

        # 加载当前循环的最优权重
        model = YOLO(weight_path)

        # 每次验证新模型前，清空 GPU 缓存并重置峰值显存统计，确保 PGM 独立且准确
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 2. 开始验证/测试
        print(f"开始计算 {full_model_name} 评估指标...")
        metrics = model.val(
            data=os.path.join(os.path.dirname(__file__), '..', 'data2.yaml'),  # DUO对应data2
            split='val',
            imgsz=640,
            batch=1,
            device='0',
            project='Result',
            name=f'{full_model_name}_val',
            save=True
        )

        # ---------------- 3. 提取指标 ----------------
        # 精度指标
        p = metrics.box.mp
        r = metrics.box.mr
        map50 = metrics.box.map50
        map95 = metrics.box.map
        f1 = 2 * (p * r) / (p + r + 1e-16)

        # 轻量化指标
        params_m = sum(x.numel() for x in model.model.parameters()) / 1e6
        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        try:
            gflops = model.info(detailed=False)[1]
        except:
            gflops = "未知"

        # 部署与实时推理指标提取
        speed = metrics.speed
        latency_total_ms = speed['preprocess'] + speed['inference'] + speed['postprocess']
        latency_infer_ms = speed['inference']
        fps_total = 1000.0 / latency_total_ms if latency_total_ms > 0 else 0
        fps_infer = 1000.0 / latency_infer_ms if latency_infer_ms > 0 else 0

        if torch.cuda.is_available():
            pgm_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            pgm_mb = 0.0

        # ---------------- 4. 终端直观打印汇总 ----------------
        print(f"\n================ {full_model_name} 论文核心指标总结 ================")
        print(f"【部署与实时性指标 (Inference & Deployment)】")
        print(f"  - Latency (纯推理): {latency_infer_ms:.2f} ms/img")
        print(f"  - Latency (端到端): {latency_total_ms:.2f} ms/img")
        print(f"  - FPS (纯推理)  : {fps_infer:.2f} fps")
        print(f"  - FPS (端到端)  : {fps_total:.2f} fps")
        print(f"  - PGM (峰值显存): {pgm_mb:.2f} MB")
        print("===============================================================\n")

        # ---------------- 5. 写入 TXT 文件 ----------------
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{full_model_name}\t{latency_infer_ms:.2f}\t{latency_total_ms:.2f}\t{fps_infer:.2f}\t{fps_total:.2f}\t{pgm_mb:.2f}\n")