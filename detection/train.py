"""Train YOLO-based underwater object detection model.

Usage:
    cd detection
    python train.py --data data1.yaml --model ../weights/yolov8n.pt --epochs 100
    python train.py --data data2.yaml --model ../weights/best.pt --epochs 200 --imgsz 640
"""

from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description='Train YOLO detection model')
    parser.add_argument('--data', type=str, default='data1.yaml', help='Dataset config yaml file')
    parser.add_argument('--model', type=str, default='../weights/yolov8n.pt', help='Pretrained model weights')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (e.g. 0, 0,1, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
