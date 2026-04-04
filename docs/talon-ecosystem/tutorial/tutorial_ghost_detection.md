---
sidebar_position: 15
---

# Tutorial: Ghost Architecture & Detection Pipeline

This tutorial covers the GhostNet-derived spiking primitives and the object detection pipeline in TALON IR. These primitives power efficient architectures like SU-YOLO for neuromorphic object detection.

## What You'll Learn

1. SGhostConv — spiking ghost convolution
2. GhostBasicBlock1 — stride-2 CSP-ELAN ghost bottleneck (backbone)
3. GhostBasicBlock2 — no-stride CSP-ELAN ghost bottleneck (FPN head)
4. SGhostEncoderLite — stem encoder (Layer 0)
5. ChannelSplit and Concat — channel manipulation
6. Detection pipeline: SDDetect → DFLDecode → Dist2BBox → NMS

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. SGhostConv

A Spiking Ghost Convolution generates features efficiently by combining a primary 1×1 convolution with a cheap depthwise 3×3 convolution. The output channels are `2 × primary_out_channels` (primary features concatenated with ghost features).

```python
import numpy as np
from talon import ir

ghost_conv = ir.SGhostConv(
    primary_weight=np.random.randn(16, 8, 1, 1).astype(np.float32) * 0.1,
    primary_bias=np.zeros(16, dtype=np.float32),
    cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1,
    cheap_bias=np.zeros(16, dtype=np.float32),
)

print(f"SGhostConv:")
print(f"  Primary weight: {ghost_conv.primary_weight.shape}")
print(f"  Cheap weight: {ghost_conv.cheap_weight.shape}")
print(f"  Output channels: {ghost_conv.primary_weight.shape[0] * 2}")
```

**Output:**

```
SGhostConv:
  Primary weight: (16, 8, 1, 1)
  Cheap weight: (16, 1, 3, 3)
  Output channels: 32
```

---

## 2. GhostBasicBlock1 (Stride-2 Backbone)

The stride-2 block downsamples spatial dimensions while expanding channels. It is the core building block of the SU-YOLO backbone. Internally it uses three SGhostConv operations:

- **cvres**: residual path (stride-2 depthwise downsample)
- **cv0**: main path first ghost conv
- **cv2**: main path second ghost conv (processes split features)

### Channel Constraints

```
cv0 ghost output = 2 * cp0   (primary + cheap)
split: Z1 = cp0 channels, Z2 = cp0 channels
cv2_in  = cp0   (cv2 processes Z2, which is the primary-only half)
cvres_out = cp0 / 2  (= cp_r)
cv2_out   = cp_r
final output = cat(Z1[cp0], cv2_out[2*cp_r]) = cp0 + 2*cp_r channels
```

```python
in_ch = 32
cp0 = 16    # cv0 primary output
cp_r = 8    # cvres primary = cp0 / 2
cp2 = 8     # cv2 primary = cp_r

gb1 = ir.GhostBasicBlock1(
    cvres_primary_weight=np.random.randn(cp_r, in_ch, 1, 1).astype(np.float32) * 0.1,
    cvres_primary_bias=np.zeros(cp_r, dtype=np.float32),
    cvres_cheap_weight=np.random.randn(cp_r, 1, 3, 3).astype(np.float32) * 0.1,
    cvres_cheap_bias=np.zeros(cp_r, dtype=np.float32),
    cv0_primary_weight=np.random.randn(cp0, in_ch, 1, 1).astype(np.float32) * 0.1,
    cv0_primary_bias=np.zeros(cp0, dtype=np.float32),
    cv0_cheap_weight=np.random.randn(cp0, 1, 3, 3).astype(np.float32) * 0.1,
    cv0_cheap_bias=np.zeros(cp0, dtype=np.float32),
    cv2_primary_weight=np.random.randn(cp2, cp0, 1, 1).astype(np.float32) * 0.1,
    cv2_primary_bias=np.zeros(cp2, dtype=np.float32),
    cv2_cheap_weight=np.random.randn(cp2, 1, 3, 3).astype(np.float32) * 0.1,
    cv2_cheap_bias=np.zeros(cp2, dtype=np.float32),
    stride=2,
)

print(f"GhostBasicBlock1 (stride-2):")
print(f"  Input channels: {in_ch}")
print(f"  cv0 primary out: {cp0} (total: {cp0 * 2})")
print(f"  cvres primary out: {cp_r} (total: {cp_r * 2})")
print(f"  cv2 primary out: {cp2} (total: {cp2 * 2})")
print(f"  stride: {gb1.stride}")
```

**Output:**

```
GhostBasicBlock1 (stride-2):
  Input channels: 32
  cv0 primary out: 16 (total: 32)
  cvres primary out: 8 (total: 16)
  cv2 primary out: 8 (total: 16)
  stride: 2
```

---

## 3. GhostBasicBlock2 (No-Stride FPN Head)

The no-stride block maintains spatial dimensions and is used in the FPN (Feature Pyramid Network) head. It uses two SGhostConv operations (cv0 and cv2) without a residual downsample path.

### Channel Constraints

```
cv0 ghost output = 2 * cp0   (primary + cheap)
split: Z1 = cp0 channels, Z2 = cp0 channels
cv2_in = cp0   (cv2 processes Z2, the primary-only half of cv0's output)
cv2_out = 2 * cp2
final output = cat(Z1[cp0], cv2_out[2*cp2]) = cp0 + 2*cp2 channels
```

```python
in_ch = 32
cp0 = 16    # cv0 primary output
cp2 = 8     # cv2 primary = cp0 / 2

gb2 = ir.GhostBasicBlock2(
    cv0_primary_weight=np.random.randn(cp0, in_ch, 1, 1).astype(np.float32) * 0.1,
    cv0_primary_bias=np.zeros(cp0, dtype=np.float32),
    cv0_cheap_weight=np.random.randn(cp0, 1, 3, 3).astype(np.float32) * 0.1,
    cv0_cheap_bias=np.zeros(cp0, dtype=np.float32),
    cv2_primary_weight=np.random.randn(cp2, cp0, 1, 1).astype(np.float32) * 0.1,
    cv2_primary_bias=np.zeros(cp2, dtype=np.float32),
    cv2_cheap_weight=np.random.randn(cp2, 1, 3, 3).astype(np.float32) * 0.1,
    cv2_cheap_bias=np.zeros(cp2, dtype=np.float32),
)

print(f"GhostBasicBlock2 (no stride):")
print(f"  Input channels: {in_ch}")
print(f"  cv0 primary out: {cp0} (total: {cp0 * 2})")
print(f"  cv2 primary out: {cp2} (total: {cp2 * 2})")
```

**Output:**

```
GhostBasicBlock2 (no stride):
  Input channels: 32
  cv0 primary out: 16 (total: 32)
  cv2 primary out: 8 (total: 16)
```

---

## 4. SGhostEncoderLite

The stem encoder (Layer 0) combines a standard 3×3 convolution with a ghost convolution. It is the entry point of the SU-YOLO backbone, processing the raw input into initial feature maps with stride-2 spatial reduction.

```python
encoder = ir.SGhostEncoderLite(
    conv1_weight=np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1,
    conv1_bias=np.zeros(16, dtype=np.float32),
    ghost_primary_weight=np.random.randn(16, 16, 1, 1).astype(np.float32) * 0.1,
    ghost_primary_bias=np.zeros(16, dtype=np.float32),
    ghost_cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1,
    ghost_cheap_bias=np.zeros(16, dtype=np.float32),
    stride=2,
)

print(f"SGhostEncoderLite:")
print(f"  conv1 weight: {encoder.conv1_weight.shape}")
print(f"  ghost primary: {encoder.ghost_primary_weight.shape}")
print(f"  ghost cheap: {encoder.ghost_cheap_weight.shape}")
print(f"  stride: {encoder.stride}")
```

**Output:**

```
SGhostEncoderLite:
  conv1 weight: (16, 3, 3, 3)
  ghost primary: (16, 16, 1, 1)
  ghost cheap: (16, 1, 3, 3)
  stride: 2
```

---

## 5. ChannelSplit and Concat

These primitives handle channel manipulation within the CSP-ELAN structure.

```python
split = ir.ChannelSplit(split_sections=(16, 16), dim=0)
print(f"ChannelSplit: sections={split.split_sections}, dim={split.dim}")

concat = ir.Concat(num_inputs=2, dim=0)
print(f"Concat: num_inputs={concat.num_inputs}, dim={concat.dim}")
```

**Output:**

```
ChannelSplit: sections=(16, 16), dim=0
Concat: num_inputs=2, dim=0
```

---

## 6. Detection Pipeline

The detection head processes feature maps through four stages:

```
SDDetect → DFLDecode → Dist2BBox → NMS
```

| Stage | Description |
|-------|-------------|
| **SDDetect** | Per-scale spiking detection head (regression + classification branches) |
| **DFLDecode** | Distribution Focal Loss — decodes discrete distributions into distances |
| **Dist2BBox** | Converts distances (top/bottom/left/right) to bounding box coordinates |
| **NMS** | Non-Maximum Suppression — filters overlapping detections |

### SDDetect

```python
detect = ir.SDDetect(
    num_classes=80,
    reg_max=16,
    stride=8,
    cv2_w0=np.random.randn(64, 32, 3, 3).astype(np.float32) * 0.01,
    cv2_b0=np.zeros(64, dtype=np.float32),
    cv2_w1=np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01,
    cv2_b1=np.zeros(64, dtype=np.float32),
    cv2_w2=np.random.randn(64, 64, 1, 1).astype(np.float32) * 0.01,
    cv2_b2=np.zeros(64, dtype=np.float32),
    cv3_w0=np.random.randn(80, 32, 3, 3).astype(np.float32) * 0.01,
    cv3_b0=np.zeros(80, dtype=np.float32),
    cv3_w1=np.random.randn(80, 80, 3, 3).astype(np.float32) * 0.01,
    cv3_b1=np.zeros(80, dtype=np.float32),
    cv3_w2=np.random.randn(80, 80, 1, 1).astype(np.float32) * 0.01,
    cv3_b2=np.zeros(80, dtype=np.float32),
)
print(f"SDDetect: classes={detect.num_classes}, reg_max={detect.reg_max}, stride={detect.stride}")
```

**Output:**

```
SDDetect: classes=80, reg_max=16, stride=8
```

### DFLDecode, Dist2BBox, NMS

```python
dfl = ir.DFLDecode(reg_max=16)
print(f"DFLDecode: reg_max={dfl.reg_max}")

d2b = ir.Dist2BBox(stride=8)
print(f"Dist2BBox: stride={d2b.stride}")

nms = ir.NMS(iou_threshold=0.45, score_threshold=0.25, max_detections=300)
print(f"NMS: iou={nms.iou_threshold}, score={nms.score_threshold}, max_det={nms.max_detections}")
```

**Output:**

```
DFLDecode: reg_max=16
Dist2BBox: stride=8
NMS: iou=0.45, score=0.25, max_det=300
```

### Full Detection Graph

```python
det_nodes = {
    "features": ir.Input(np.array([32, 80, 80])),
    "detect": detect,
    "dfl": dfl,
    "d2b": d2b,
    "nms": nms,
    "output": ir.Output(np.array([300, 6])),
}

det_edges = [
    ("features", "detect"),
    ("detect", "dfl"),
    ("dfl", "d2b"),
    ("d2b", "nms"),
    ("nms", "output"),
]

det_graph = ir.Graph(nodes=det_nodes, edges=det_edges)
print(f"Detection pipeline graph:")
print(f"  Nodes: {len(det_graph.nodes)}")
print(f"  Edges: {len(det_graph.edges)}")
print(f"  Is DAG: {det_graph.is_dag}")
```

**Output:**

```
Detection pipeline graph:
  Nodes: 6
  Edges: 5
  Is DAG: True
```

### Serialize and Verify

```python
ir.write("models/detection_pipeline.t1c", det_graph)
loaded = ir.read("models/detection_pipeline.t1c")
print(f"Saved & reloaded: {len(loaded.nodes)} nodes, {len(loaded.edges)} edges")
```

**Output:**

```
Saved & reloaded: 6 nodes, 5 edges
```

---

## 7. Building a Mini SU-YOLO Backbone

Combining the ghost primitives into a simplified backbone:

```python
backbone_nodes = {
    "input": ir.Input(np.array([3, 640, 640])),
    "encoder": ir.SGhostEncoderLite(
        conv1_weight=np.random.randn(16, 3, 3, 3).astype(np.float32) * 0.1,
        conv1_bias=np.zeros(16, dtype=np.float32),
        ghost_primary_weight=np.random.randn(16, 16, 1, 1).astype(np.float32) * 0.1,
        ghost_primary_bias=np.zeros(16, dtype=np.float32),
        ghost_cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1,
        ghost_cheap_bias=np.zeros(16, dtype=np.float32),
        stride=2,
    ),
    "stage1": ir.GhostBasicBlock1(
        cvres_primary_weight=np.random.randn(8, 32, 1, 1).astype(np.float32) * 0.1,
        cvres_primary_bias=np.zeros(8, dtype=np.float32),
        cvres_cheap_weight=np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1,
        cvres_cheap_bias=np.zeros(8, dtype=np.float32),
        cv0_primary_weight=np.random.randn(16, 32, 1, 1).astype(np.float32) * 0.1,
        cv0_primary_bias=np.zeros(16, dtype=np.float32),
        cv0_cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1,
        cv0_cheap_bias=np.zeros(16, dtype=np.float32),
        cv2_primary_weight=np.random.randn(8, 16, 1, 1).astype(np.float32) * 0.1,
        cv2_primary_bias=np.zeros(8, dtype=np.float32),
        cv2_cheap_weight=np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1,
        cv2_cheap_bias=np.zeros(8, dtype=np.float32),
        stride=2,
    ),
    "output": ir.Output(np.array([16, 160, 160])),
}

backbone_edges = [
    ("input", "encoder"),
    ("encoder", "stage1"),
    ("stage1", "output"),
]

backbone = ir.Graph(nodes=backbone_nodes, edges=backbone_edges)
print(f"Mini SU-YOLO backbone:")
print(f"  Nodes: {len(backbone.nodes)}")
print(f"  Edges: {len(backbone.edges)}")
for name, node in backbone.nodes.items():
    print(f"  {name}: {type(node).__name__}")
```

**Output:**

```
Mini SU-YOLO backbone:
  Nodes: 4
  Edges: 3
  input: Input
  encoder: SGhostEncoderLite
  stage1: GhostBasicBlock1
  output: Output
```

---

## Summary

| Primitive | Role | Key Parameters |
|-----------|------|----------------|
| `SGhostConv` | Ghost convolution | `primary_weight`, `cheap_weight` |
| `GhostBasicBlock1` | Stride-2 backbone block | `cvres_*`, `cv0_*`, `cv2_*`, `stride=2` |
| `GhostBasicBlock2` | No-stride FPN block | `cv0_*`, `cv2_*` |
| `SGhostEncoderLite` | Stem encoder | `conv1_*`, `ghost_*`, `stride` |
| `ChannelSplit` | Split channels | `split_sections`, `dim` |
| `Concat` | Concatenate channels | `num_inputs`, `dim` |
| `SDDetect` | Detection head | `num_classes`, `reg_max`, `stride` |
| `DFLDecode` | DFL distance decode | `reg_max` |
| `Dist2BBox` | Distance to bbox | `stride` |
| `NMS` | Non-Maximum Suppression | `iou_threshold`, `score_threshold`, `max_detections` |

## What's Next

- [Tutorial: Hardware Mapping](./tutorial_hardware_mapping) — Partition and place on hardware
- [Tutorial: Backend](./tutorial_backend) — Simulate and profile on CPU/FPGA
- [Primitives Reference](../primitives) — All 36 TALON IR primitives
