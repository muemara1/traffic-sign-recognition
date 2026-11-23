# 🚦 Traffic Sign Recognition: from training to deployment

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005ced?logo=onnx&logoColor=white)

Training a good classifier is the easy part. Car makers care about **size, speed and running without a big framework**. This project goes all the way from data to a standalone app.

| Data | Key result |
|---|---|
| GTSRB, 43 classes, 39k images | **98.6%** test accuracy, INT8 model **3.6× smaller**, **0.21 ms** per image on 1 CPU thread |

📓 [`traffic_signs.ipynb`](traffic_signs.ipynb) · 📦 [`deploy/`](deploy/)

![The 43 GTSRB classes](assets/traffic_sign_classes.png)

## What I did

- **Leak-free validation split by track.** GTSRB films every real sign 30 times from a moving car. A random split puts almost identical photos in both train and validation, so I split by *physical sign* instead.
- **Augmentation that keeps the meaning:** small rotations, shifts and light changes, but **no horizontal flips**, because a flipped "Keep right" *is* "Keep left".
- **A small CNN** (293k parameters, 3 conv blocks + global average pooling) with label smoothing and a cosine learning-rate schedule.
- **Deployment pipeline:** PyTorch → **ONNX** → **INT8 static quantization** (calibrated on training images), measured like an embedded device: 1 CPU thread, batch size 1, with warm-up.
- **A standalone predictor** ([`deploy/predict_sign.py`](deploy/predict_sign.py)) that needs only `onnxruntime`, `numpy` and `Pillow`. The preprocessing settings are saved in `labels.json`, so the device does exactly the same steps as training.

![Deployment results](assets/traffic_deployment.png)

| Version | Size | Latency (1 thread) | Test accuracy |
|---|---|---|---|
| PyTorch FP32 | 1161 KB | 5.45 ms | 98.57% |
| ONNX FP32 | 1175 KB | 0.50 ms | 98.57% |
| **ONNX INT8** | **325 KB** | **0.21 ms** | **98.58%** |

## Limits & next steps

GTSRB images are already cropped around the sign. A real car first needs a **detector** (e.g. YOLO) to find signs in the full camera frame. Latency was measured on a laptop CPU; the next step is measuring on a Raspberry Pi.

## Try the deployed model

No PyTorch needed:

```bash
pip install onnxruntime numpy pillow
python deploy/predict_sign.py path/to/sign.png
```

## Run the notebook

```bash
git clone https://github.com/muemara1/traffic-sign-recognition.git
cd traffic-sign-recognition
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# For an NVIDIA GPU, install the CUDA build of torch first (see pytorch.org)
pip install -r requirements.txt
jupyter notebook traffic_signs.ipynb
```

The data is downloaded automatically by `torchvision.datasets.GTSRB` into `data/gtsrb/` (not stored in the repo).

> **AMD GPU note:** on ROCm under Windows, MIOpen may fail to compile its BatchNorm kernel.
> The notebook sets `torch.backends.cudnn.enabled = False` to use PyTorch's built-in kernels instead.

## Credits

Dataset: [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html), Ruhr-Universität Bochum.

---

More of my work: [github.com/muemara1](https://github.com/muemara1)
