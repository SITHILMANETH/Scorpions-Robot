import argparse
from pathlib import Path

from ultralytics import YOLO

from pi_bundle import DEFAULT_BUNDLE_DIR_NAME, DEFAULT_EXPORT_OPSET, export_for_pi, resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILE = PROJECT_ROOT / "data.yaml"
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_RUN_NAME = "ball_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a red-ball detector with Ultralytics YOLO and export a Pi 5 bundle.")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to data.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Base YOLO weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
        help="Training image size. Smaller is faster and more Pi-friendly.",
    )
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Run name under runs/detect.")
    parser.add_argument("--device", default="cpu", help="Training device, for example cpu or 0.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data loader workers. Zero is usually safest on Windows.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--export-format",
        default="onnx",
        help="Pi export format after training. Use 'none' to skip export.",
    )
    parser.add_argument(
        "--export-imgsz",
        type=int,
        default=None,
        help="Export image size. Defaults to the training image size.",
    )
    parser.add_argument(
        "--export-opset",
        type=int,
        default=DEFAULT_EXPORT_OPSET,
        help="ONNX opset version for Pi export.",
    )
    parser.add_argument(
        "--export-nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include NMS inside the exported ONNX model for a simpler Pi runtime.",
    )
    parser.add_argument(
        "--bundle-dir-name",
        default=DEFAULT_BUNDLE_DIR_NAME,
        help="Folder created inside the training run with the files to copy to the Pi.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str) -> Path:
    return resolve_project_path(path_arg)


def main() -> None:
    args = parse_args()
    data_file = resolve_path(args.data)
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset config: {data_file}")

    model = YOLO(args.model)
    model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
    )

    best_weights = Path(model.trainer.best).resolve()
    print(f"Training done. Best model is at: {best_weights}")

    if args.export_format.lower() == "none":
        return

    export_imgsz = args.export_imgsz or args.imgsz
    exported_model, bundle_dir = export_for_pi(
        weights_path=best_weights,
        data_file=data_file,
        export_imgsz=export_imgsz,
        export_format=args.export_format,
        export_nms=args.export_nms,
        export_opset=args.export_opset,
        bundle_dir_name=args.bundle_dir_name,
    )
    print(f"Pi 5 model exported to: {exported_model}")
    print(f"Pi 5 bundle ready at: {bundle_dir}")


if __name__ == "__main__":
    main()
