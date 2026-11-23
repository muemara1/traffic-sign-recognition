"""Traffic sign DETECTION + CLASSIFICATION for a small device.
Needs ONLY: onnxruntime, numpy, Pillow (no PyTorch, no ultralytics!).

Stage 1: sign_detector.onnx (YOLO)        -> where are the signs?
Stage 2: traffic_sign_int8.onnx (our CNN)  -> which sign is it?

Usage:
    python detect_signs.py road.jpg [--out result.jpg] [--conf 0.25]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import predict_sign                                   # stage 2: the classifier (same folder)

CONF = 0.25        # minimum detection confidence
IOU_NMS = 0.5      # boxes overlapping more than this are duplicates
MARGIN = 0.10      # crop margin, like the border around GTSRB training images

_session = None


def set_threads(n):
    """(Re)load the detector with n CPU threads (also used by the classifier)."""
    global _session
    options = ort.SessionOptions()
    options.intra_op_num_threads = n
    _session = ort.InferenceSession(str(HERE / "sign_detector.onnx"), options,
                                    providers=["CPUExecutionProvider"])
    predict_sign.session = ort.InferenceSession(str(HERE / "traffic_sign_int8.onnx"), options,
                                                providers=["CPUExecutionProvider"])


def letterbox(img, out_h, out_w):
    """Resize keeping the aspect ratio, then pad with gray (114) to (out_h, out_w), centered.
    Returns the image array and how to map boxes back (scale, pad_x, pad_y)."""
    scale = min(out_h / img.height, out_w / img.width)
    new_w, new_h = round(img.width * scale), round(img.height * scale)
    pad_x, pad_y = (out_w - new_w) / 2, (out_h - new_h) / 2
    canvas = Image.new("RGB", (out_w, out_h), (114, 114, 114))
    canvas.paste(img.resize((new_w, new_h), Image.BILINEAR), (round(pad_x - 0.1), round(pad_y - 0.1)))
    return np.asarray(canvas), scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold=IOU_NMS):
    """Non-Maximum Suppression: keep the best box, delete boxes that overlap it too much, repeat."""
    order = scores.argsort()[::-1]                   # best first
    keep = []
    while len(order) > 0:
        best, rest = order[0], order[1:]
        keep.append(best)
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[best] + area[rest] - inter)
        order = rest[iou < iou_threshold]
    return np.array(keep, dtype=int)


def detect(img, conf=CONF):
    """Stage 1. A PIL image -> boxes [N, 4] (x1, y1, x2, y2 in ORIGINAL pixels) and scores [N]."""
    if _session is None:
        set_threads(4)
    _, _, in_h, in_w = _session.get_inputs()[0].shape          # e.g. [1, 3, 608, 1024]
    arr, scale, pad_x, pad_y = letterbox(img, in_h, in_w)
    x = arr.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0   # [1, 3, H, W], 0..1

    out = _session.run(None, {_session.get_inputs()[0].name: x})[0][0]   # [5, N]: cx, cy, w, h, score
    cx, cy, w, h, scores = out
    keep = scores >= conf
    cx, cy, w, h, scores = cx[keep], cy[keep], w[keep], h[keep], scores[keep]

    # center + size (letterboxed pixels) -> corners (original pixels)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img.width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img.height)

    if len(boxes) == 0:
        return boxes, scores
    keep = nms(boxes, scores)
    return boxes[keep], scores[keep]


def detect_and_classify(img, conf=CONF):
    """Stage 1 + 2. Returns a list of dicts: box, det_conf, class_id, name, cls_conf."""
    img = img.convert("RGB")
    results = []
    for box, score in zip(*detect(img, conf)):
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        crop = img.crop((max(0, x1 - mx), max(0, y1 - my), min(img.width, x2 + mx), min(img.height, y2 + my)))
        class_id, name, cls_conf = predict_sign.classify_image(crop)
        results.append({"box": box, "det_conf": float(score), "class_id": class_id,
                        "name": name, "cls_conf": cls_conf})
    return results


def draw(img, results):
    img = img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for r in results:
        x1, y1, x2, y2 = r["box"]
        d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = f"{r['name']} {r['cls_conf']:.0%}"
        d.rectangle([x1, y1 - 14, x1 + 6 * len(label), y1], fill=(0, 255, 0))
        d.text((x1 + 2, y1 - 13), label, fill=(0, 0, 0))
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--out", default="detections.jpg")
    parser.add_argument("--conf", type=float, default=CONF)
    args = parser.parse_args()

    img = Image.open(args.image)
    start = time.perf_counter()
    results = detect_and_classify(img, args.conf)
    ms = (time.perf_counter() - start) * 1000

    print(f"{Path(args.image).name}: {len(results)} sign(s) in {ms:.0f} ms")
    for r in results:
        x1, y1, x2, y2 = r["box"].round().astype(int)
        print(f"  {r['name']:30s} classifier {r['cls_conf']:4.0%}  detector {r['det_conf']:4.0%}  box ({x1}, {y1}, {x2}, {y2})")
    draw(img, results).save(args.out)
    print(f"saved: {args.out}")
