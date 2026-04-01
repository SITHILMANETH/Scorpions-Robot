# Object Detection

This folder is prepared for GitHub upload.

Included:
- Training and inference scripts
- YOLO dataset in `images/` and `labels/`
- Trained model files in `runs/detect/ball_detector/weights/`

Quick start:

```bash
python -m pip install ultralytics opencv-python pyyaml onnxruntime
python train.py
python webcam.py
```

Dataset config:
- `data.yaml` uses a relative path so it works after upload or clone.

Main files:
- `train.py`
- `webcam.py`
- `pi_webcam.py`
- `pi_bundle.py`
- `export_pi5.py`
