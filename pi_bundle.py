import json
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPORT_FORMAT = "onnx"
DEFAULT_EXPORT_OPSET = 12
DEFAULT_BUNDLE_DIR_NAME = "pi5_bundle"
PI_RUNTIME_SCRIPT = PROJECT_ROOT / "pi_webcam.py"
PI_REQUIREMENTS_FILE = PROJECT_ROOT / "requirements-pi.txt"


def resolve_project_path(path_arg: str | Path) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_class_names(data_file: Path) -> list[str]:
    data = yaml.safe_load(data_file.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])

    if isinstance(names, dict):
        items = sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]))
        class_names = [str(name).strip() for _, name in items]
    elif isinstance(names, list):
        class_names = [str(name).strip() for name in names]
    else:
        class_names = []

    class_names = [name for name in class_names if name]
    if not class_names:
        raise ValueError(f"No class names found in {data_file}")

    return class_names


def export_model_for_pi(
    weights_path: Path,
    export_imgsz: int,
    export_format: str = DEFAULT_EXPORT_FORMAT,
    export_nms: bool = True,
    export_opset: int = DEFAULT_EXPORT_OPSET,
) -> Path:
    if export_format != DEFAULT_EXPORT_FORMAT:
        raise ValueError(f"Pi 5 packaging currently supports only '{DEFAULT_EXPORT_FORMAT}' export.")

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format=export_format,
        imgsz=export_imgsz,
        dynamic=False,
        nms=export_nms,
        simplify=False,
        opset=export_opset,
    )
    return resolve_project_path(exported_path)


def create_pi_bundle(
    run_dir: Path,
    exported_model_path: Path,
    class_names: list[str],
    export_imgsz: int,
    weights_path: Path,
    bundle_dir_name: str = DEFAULT_BUNDLE_DIR_NAME,
) -> Path:
    bundle_dir = run_dir / bundle_dir_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundled_model_path = bundle_dir / "model.onnx"
    shutil.copy2(exported_model_path, bundled_model_path)
    shutil.copy2(PI_RUNTIME_SCRIPT, bundle_dir / PI_RUNTIME_SCRIPT.name)
    shutil.copy2(PI_REQUIREMENTS_FILE, bundle_dir / PI_REQUIREMENTS_FILE.name)
    (bundle_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    metadata = {
        "model_format": "onnx",
        "model_file": bundled_model_path.name,
        "weights_source": str(weights_path),
        "exported_from": str(exported_model_path),
        "imgsz": export_imgsz,
        "classes": class_names,
        "recommended_confidence": 0.4,
    }
    (bundle_dir / "model_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (bundle_dir / "README_PI5.txt").write_text(build_bundle_readme(export_imgsz), encoding="utf-8")
    return bundle_dir.resolve()


def build_bundle_readme(export_imgsz: int) -> str:
    return (
        "Copy this whole folder to the Raspberry Pi 5.\n\n"
        "Quick start:\n"
        "1. Install the Python packages:\n"
        "   python3 -m pip install -r requirements-pi.txt\n"
        "2. Run the webcam detector:\n"
        "   python3 pi_webcam.py --model model.onnx --classes classes.txt --camera 0 --conf 0.4\n"
        "3. If you want to test a single image instead:\n"
        "   python3 pi_webcam.py --model model.onnx --classes classes.txt --image test.jpg --save result.jpg --no-show\n\n"
        f"Notes:\n- The exported model uses {export_imgsz}x{export_imgsz} input.\n"
        "- Press Q to quit webcam mode.\n"
        "- The default runtime uses a USB webcam through OpenCV.\n"
    )


def export_for_pi(
    weights_path: str | Path,
    data_file: str | Path,
    export_imgsz: int,
    export_format: str = DEFAULT_EXPORT_FORMAT,
    export_nms: bool = True,
    export_opset: int = DEFAULT_EXPORT_OPSET,
    bundle_dir_name: str = DEFAULT_BUNDLE_DIR_NAME,
) -> tuple[Path, Path]:
    resolved_weights = resolve_project_path(weights_path)
    resolved_data = resolve_project_path(data_file)

    if not resolved_weights.exists():
        raise FileNotFoundError(f"Missing trained weights: {resolved_weights}")
    if not resolved_data.exists():
        raise FileNotFoundError(f"Missing dataset config: {resolved_data}")

    class_names = load_class_names(resolved_data)
    run_dir = resolved_weights.parent.parent if resolved_weights.parent.name == "weights" else resolved_weights.parent
    exported_model_path = export_model_for_pi(
        weights_path=resolved_weights,
        export_imgsz=export_imgsz,
        export_format=export_format,
        export_nms=export_nms,
        export_opset=export_opset,
    )
    bundle_dir = create_pi_bundle(
        run_dir=run_dir,
        exported_model_path=exported_model_path,
        class_names=class_names,
        export_imgsz=export_imgsz,
        weights_path=resolved_weights,
        bundle_dir_name=bundle_dir_name,
    )
    return exported_model_path, bundle_dir
