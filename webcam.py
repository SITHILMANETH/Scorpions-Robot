import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "ball_detector" / "weights" / "best.pt"
WINDOW_NAME = "Red Ball Detector"
WINDOW_NAME_RAW = "Camera Preview"
CAMERA_BACKENDS = [
    ("default", None),
    ("dshow", cv2.CAP_DSHOW),
    ("msmf", cv2.CAP_MSMF),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ball detection on a webcam feed.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to a trained YOLO weights file.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Preferred camera index. If omitted, available cameras are tried automatically.",
    )
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=5,
        help="Highest camera index to probe during auto-detection.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--raw-preview",
        action="store_true",
        help="Open the camera without running detection. Useful for webcam testing.",
    )
    return parser.parse_args()


def resolve_model_path(model_arg: str) -> Path:
    model_path = Path(model_arg)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


def open_camera(index: int, backend: int | None) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None

    return cap


def find_camera(preferred_index: int | None, max_camera_index: int) -> tuple[cv2.VideoCapture, int, str]:
    if preferred_index is None:
        indices = list(range(max_camera_index + 1))
    else:
        indices = [preferred_index]

    for backend_name, backend in CAMERA_BACKENDS:
        for index in indices:
            cap = open_camera(index, backend)
            if cap is not None:
                return cap, index, backend_name

    tried = ", ".join(str(index) for index in indices)
    raise RuntimeError(
        f"Could not open a webcam. Tried camera indices: {tried}. "
        "Close other apps using the camera, then try again with --camera 0, 1, or 2."
    )


def load_model_or_raise(model_path: Path) -> YOLO:
    if model_path.exists():
        return YOLO(str(model_path))

    local_fallbacks = [
        PROJECT_ROOT / "best.onnx",
        PROJECT_ROOT / "best.pt",
    ]
    for fallback in local_fallbacks:
        if fallback.exists():
            print(f"Default model not found. Using {fallback} instead.")
            return YOLO(str(fallback))

    discovered_fallbacks = sorted(PROJECT_ROOT.glob("runs/detect/*/weights/best.onnx"))
    discovered_fallbacks += sorted(PROJECT_ROOT.glob("runs/detect/*/weights/best.pt"))
    if discovered_fallbacks:
        fallback = discovered_fallbacks[-1]
        print(f"Default model not found. Using {fallback} instead.")
        return YOLO(str(fallback))

    raise FileNotFoundError(
        "No trained model was found.\n"
        f"Expected: {model_path}\n"
        "Run training first with:\n"
        "  .\\yolo_env\\Scripts\\python train.py\n"
        "If you only want to test the camera, run:\n"
        "  .\\yolo_env\\Scripts\\python webcam.py --raw-preview"
    )


def main() -> None:
    args = parse_args()
    model = None if args.raw_preview else load_model_or_raise(resolve_model_path(args.model))
    cap, camera_index, backend_name = find_camera(args.camera, args.max_camera_index)
    window_name = WINDOW_NAME_RAW if args.raw_preview else WINDOW_NAME

    print(f"Using camera index {camera_index} via {backend_name}.")
    if args.raw_preview:
        print("Raw preview started. Press Q to quit.")
    else:
        print("Detection started. Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame received from webcam. Stopping.")
                break

            if model is None:
                output_frame = frame
            else:
                results = model(frame, conf=args.conf, verbose=False)
                output_frame = results[0].plot()

            cv2.imshow(window_name, output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
