Copy these files to the Raspberry Pi 5 if you want to run the original PyTorch model instead of ONNX.

Quick start:
1. Install the Python packages:
   python3 -m pip install -r requirements-pi-pt.txt
2. Run the detector:
   python3 webcam.py --model best.pt --camera 0 --conf 0.4

Notes:
- This uses the original Ultralytics .pt model.
- It is usually slower on the Pi than the ONNX version.
