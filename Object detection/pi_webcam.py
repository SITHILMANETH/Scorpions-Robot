import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "Pi 5 Ball Detector"


def find_default_model_path() -> Path:
    bundled_model = PROJECT_ROOT / "model.onnx"
    if bundled_model.exists():
        return bundled_model

    exported_models = sorted(PROJECT_ROOT.glob("runs/detect/*/weights/best.onnx"), key=lambda path: path.stat().st_mtime)
    if exported_models:
        return exported_models[-1]

    return bundled_model


DEFAULT_MODEL_PATH = find_default_model_path()
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "classes.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exported ONNX detector on a Raspberry Pi 5.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the exported ONNX model.")
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES_PATH), help="Path to classes.txt.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for OpenCV.")
    parser.add_argument("--image", help="Optional image path for one-shot testing instead of webcam mode.")
    parser.add_argument("--save", help="Optional output path for an annotated frame or image.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override model input size for dynamic ONNX models. Fixed-size exports detect this automatically.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested capture width in webcam mode.")
    parser.add_argument("--height", type=int, default=720, help="Requested capture height in webcam mode.")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="ONNX Runtime CPU threads.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the preview window. Defaults to on for webcam mode and off for image mode.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str | None) -> Path | None:
    if path_arg is None:
        return None

    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_class_names(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_path}")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not class_names:
        raise ValueError(f"No class names found in {classes_path}")
    return class_names


def create_session(model_path: Path, threads: int) -> tuple[ort.InferenceSession, str]:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, threads)
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    return session, input_name


def infer_input_size(session: ort.InferenceSession, requested_imgsz: int | None) -> int:
    input_shape = session.get_inputs()[0].shape
    height = input_shape[2] if len(input_shape) > 2 else None
    width = input_shape[3] if len(input_shape) > 3 else None

    if isinstance(height, int) and isinstance(width, int):
        if height != width:
            raise ValueError(f"Expected a square model input, got {height}x{width}")
        return height

    if requested_imgsz is None:
        raise ValueError("The ONNX model has a dynamic input shape. Pass --imgsz with the desired size.")

    return requested_imgsz


def letterbox(image: np.ndarray, new_shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    shape = image.shape[:2]
    gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * gain)), int(round(shape[0] * gain)))

    resized = image
    if shape[::-1] != new_unpad:
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape[1] - new_unpad[0]
    pad_h = new_shape[0] - new_unpad[1]
    pad_w /= 2
    pad_h /= 2

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, gain, (pad_w, pad_h)


def preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    letterboxed, gain, pad = letterbox(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), gain, pad


def scale_box(box: np.ndarray, gain: float, pad: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = (x1 - pad[0]) / gain
    y1 = (y1 - pad[1]) / gain
    x2 = (x2 - pad[0]) / gain
    y2 = (y2 - pad[1]) / gain

    height, width = shape
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    return x1, y1, x2, y2


def run_detector(
    session: ort.InferenceSession,
    input_name: str,
    frame: np.ndarray,
    imgsz: int,
    conf_threshold: float,
) -> list[dict[str, float | int | tuple[int, int, int, int]]]:
    tensor, gain, pad = preprocess(frame, imgsz)
    predictions = session.run(None, {input_name: tensor})[0][0]

    detections = []
    for prediction in predictions:
        confidence = float(prediction[4])
        if confidence < conf_threshold:
            continue

        box = scale_box(prediction[:4], gain, pad, frame.shape[:2])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        detections.append(
            {
                "box": box,
                "confidence": confidence,
                "class_id": int(prediction[5]),
            }
        )

    return detections


def draw_detections(frame: np.ndarray, detections: list[dict], class_names: list[str], fps: float | None = None) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]
        label_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        label = f"{label_name} {confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(output, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

    if fps is not None:
        cv2.putText(output, f"FPS {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 220), 2)

    return output


def save_output_image(output_path: Path | None, image: np.ndarray) -> None:
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved annotated output to: {output_path}")


def run_image_mode(
    session: ort.InferenceSession,
    input_name: str,
    image_path: Path,
    save_path: Path | None,
    class_names: list[str],
    imgsz: int,
    conf_threshold: float,
    show_window: bool,
) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detections = run_detector(session, input_name, frame, imgsz, conf_threshold)
    annotated = draw_detections(frame, detections, class_names)

    print(f"Detections found: {len(detections)}")
    for detection in detections:
        label_name = class_names[detection['class_id']] if 0 <= detection["class_id"] < len(class_names) else detection["class_id"]
        print(f"{label_name}: {detection['confidence']:.3f} {detection['box']}")

    save_output_image(save_path, annotated)

    if show_window:
        cv2.imshow(WINDOW_NAME, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam_mode(
    session: ort.InferenceSession,
    input_name: str,
    camera_index: int,
    width: int,
    height: int,
    save_path: Path | None,
    class_names: list[str],
    imgsz: int,
    conf_threshold: float,
    show_window: bool,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    writer = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, 20.0, (width, height))

    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame received from the camera. Stopping.")
                break

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now

            detections = run_detector(session, input_name, frame, imgsz, conf_threshold)
            annotated = draw_detections(frame, detections, class_names, fps=fps)

            if writer is not None:
                writer.write(annotated)

            if show_window:
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    classes_path = resolve_path(args.classes)
    image_path = resolve_path(args.image)
    save_path = resolve_path(args.save)
    show_window = args.show if args.show is not None else image_path is None

    class_names = load_class_names(classes_path)
    session, input_name = create_session(model_path, args.threads)
    imgsz = infer_input_size(session, args.imgsz)

    print(f"Using model: {model_path}")
    print(f"Input size: {imgsz}")

    if image_path is not None:
        run_image_mode(
            session=session,
            input_name=input_name,
            image_path=image_path,
            save_path=save_path,
            class_names=class_names,
            imgsz=imgsz,
            conf_threshold=args.conf,
            show_window=show_window,
        )
        return

    print("Webcam mode started. Press Q to quit.")
    run_webcam_mode(
        session=session,
        input_name=input_name,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        save_path=save_path,
        class_names=class_names,
        imgsz=imgsz,
        conf_threshold=args.conf,
        show_window=show_window,
    )


if __name__ == "__main__":
    main()
