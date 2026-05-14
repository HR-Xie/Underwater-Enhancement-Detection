import os
import cv2
import random
import yaml
import numpy as np


def draw_and_stitch(yaml_path, num_images=8, target_size=(640, 640)):
    # 1. 解析 YAML 配置文件
    if not os.path.exists(yaml_path):
        print(f"❌ 找不到配置文件: {yaml_path}")
        return

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)

    # 获取训练集图片路径和类别名称
    images_dir = data_cfg.get('train', '')
    class_names = data_cfg.get('names', [])

    # 自动推导 labels 文件夹路径 (将路径中的 images 替换为 labels)
    labels_dir = images_dir.replace('images', 'labels')

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ 路径不存在，请检查:\n图片路径: {images_dir}\n标签路径: {labels_dir}")
        return

    # 提取数据集名称（默认使用 yaml 文件的名称，转换为小写）
    dataset_name = os.path.splitext(os.path.basename(yaml_path))[0].lower()

    # 为不同类别定义特定的好看的颜色 (BGR格式: 蓝, 绿, 红)
    # holothurian(海参)-绿色, echinus(海胆)-橙色/红色, scallop(扇贝)-蓝色
    class_colors = [
        (0, 255, 0),  # class 0: 绿色
        (0, 165, 255),  # class 1: 橙色
        (255, 0, 0)  # class 2: 蓝色
    ]

    # 2. 获取所有有效图片
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]

    if len(all_images) < num_images:
        print(f"❌ 图片数量不足 {num_images} 张！")
        return

    # 随机抽取 8 张图片
    random.seed(42)  # 固定随机种子，确保每次抽出来的图一样，方便写论文。若想换图可以修改或注释这行
    selected_images = random.sample(all_images, num_images)
    processed_images = []

    print(f"🚀 正在处理数据集，抽取 {num_images} 张图片...")

    # 3. 逐张处理
    for img_name in selected_images:
        img_path = os.path.join(images_dir, img_name)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # 画真实标注框
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, box_w, box_h = map(float, parts[1:5])

                        # YOLO (归一化) 转 像素坐标
                        x1 = int((x_center - box_w / 2) * w)
                        y1 = int((y_center - box_h / 2) * h)
                        x2 = int((x_center + box_w / 2) * w)
                        y2 = int((y_center + box_h / 2) * h)

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)

                        # 匹配颜色和标签
                        color = class_colors[cls_id % len(class_colors)]
                        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

                        # 绘制矩形框和半透明标签底色，提升高级感
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 统一尺寸，防止图片大小不一致导致拼接失败
        img_resized = cv2.resize(img, target_size)
        processed_images.append(img_resized)

    # 4. 拼接大图 (2行4列)
    if len(processed_images) == 8:
        row1 = np.hstack(processed_images[0:4])
        row2 = np.hstack(processed_images[4:8])
        final_grid = np.vstack([row1, row2])

        # 5. 保存至当前目录
        output_filename = f"{dataset_name}.png"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
        cv2.imwrite(output_path, final_grid)
        print(f"✅ 完美！拼接图已成功保存至: {output_path}")
    else:
        print("❌ 处理失败，有效图片不足 8 张。")


if __name__ == "__main__":
    # 假设你将刚刚发我的那段配置保存为了 data.yaml 放在同一目录下
    # 如果你的文件名叫 uod.yaml，请将下面的 "data.yaml" 修改为 "uod.yaml"
    yaml_file_path = "data2.yaml"
    draw_and_stitch(yaml_file_path)