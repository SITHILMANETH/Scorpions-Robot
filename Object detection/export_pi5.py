import argparse
from pathlib import Path

from pi_bundle import DEFAULT_BUNDLE_DIR_NAME, DEFAULT_EXPORT_OPSET, PROJECT_ROOT, export_for_pi


DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "ball_detector" / "weights" / "best.pt"
DATA_FILE = PROJECT_ROOT / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an existing YOLO checkpoint into a Raspberry Pi 5 bundle.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a trained best.pt checkpoint.",
    )
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to data.yaml.")
    parser.add_argument("--imgsz", type=int, default=512, help="Export image size for the Pi runtime.")
    parser.add_argument("--format", default="onnx", help="Export format. Pi packaging currently supports onnx.")
    parser.add_argument(
        "--export-opset",
        type=int,
        default=DEFAULT_EXPORT_OPSET,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--export-nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include NMS in the exported ONNX model so the Pi runtime stays simple.",
    )
    parser.add_argument(
        "--bundle-dir-name",
        default=DEFAULT_BUNDLE_DIR_NAME,
        help="Folder created next to the run with the files to copy to the Pi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported_model, bundle_dir = export_for_pi(
        weights_path=args.weights,
        data_file=args.data,
        export_imgsz=args.imgsz,
        export_format=args.format,
        export_nms=args.export_nms,
        export_opset=args.export_opset,
        bundle_dir_name=args.bundle_dir_name,
    )

    print(f"Exported Pi model: {Path(exported_model).resolve()}")
    print(f"Pi 5 bundle ready: {Path(bundle_dir).resolve()}")


if __name__ == "__main__":
    main()
