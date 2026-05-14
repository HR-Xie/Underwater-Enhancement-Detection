"""Run YOLO-based object detection on images/videos.

Usage:
    cd detection
    python predict.py --source ../assets/test_image.jpg
    python predict.py --source ./Test/images --model ../weights/best.pt --conf 0.5
"""

from ultralytics import YOLO
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run YOLO object detection')
    parser.add_argument('--source', type=str, required=True, help='Image/video path or directory')
    parser.add_argument('--model', type=str, default='../weights/yolov8n.pt', help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--save', action='store_true', default=True, help='Save output')
    parser.add_argument('--project', type=str, default='runs/predict', help='Save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        project=args.project,
        name=args.name,
    )


if __name__ == '__main__':
    main()
