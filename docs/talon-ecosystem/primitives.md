---
sidebar_position: 5
---

# Primitives Reference

TALON IR defines 36 primitives for representing spiking neural networks and hybrid ANN-SNN architectures. Each primitive maps directly to hardware operations on Type 1 Compute chips.

## Primitive Overview

| Primitive | Category | Description |
|-----------|----------|-------------|
| Affine | Linear | Linear transform y = Wx + b |
| SpikingAffine | Linear | Quantized affine with spike hints |
| Conv1d | Convolution | 1D convolution (temporal/sequential) |
| Conv2d | Convolution | 2D convolution |
| SepConv2d | Convolution | Depthwise separable convolution |
| SConv | Convolution | Spiking standard 2D convolution (Conv2d alias with SNN semantics) |
| SDConv | Convolution | Spiking depthwise 2D convolution (groups=in_channels) |
| MaxPool2d | Spatial | 2D max pooling (downsampling) |
| AvgPool2d | Spatial | 2D average pooling (downsampling) |
| Upsample | Spatial | 2D upsampling (nearest/bilinear) |
| Flatten | Reshape | Reshape to 1D |
| LIF | Neuron | Leaky integrate-and-fire |
| IF | Neuron | Integrate-and-fire (no leak) |
| Skip | Skip | Residual/skip connection |
| ReLU | ANN Activation | Rectified linear unit |
| Sigmoid | ANN Activation | Logistic sigmoid |
| Tanh | ANN Activation | Hyperbolic tangent |
| Softmax | ANN Activation | Softmax for classification |
| GELU | ANN Activation | Gaussian error linear unit |
| ELU | ANN Activation | Exponential linear unit |
| PReLU | ANN Activation | Parametric ReLU |
| BatchNorm1d | Normalization | Batch norm for linear layers |
| BatchNorm2d | Normalization | Batch norm for conv layers |
| LayerNorm | Normalization | Layer normalization |
| Dropout | Regularization | Dropout regularization |
| HybridRegion | Marker | ANN/SNN region marker |
| ChannelSplit | Routing | Channel splitting for CSP-ELAN |
| Concat | Routing | Channel concatenation |
| SGhostConv | Ghost Conv | Spiking ghost convolution |
| SGhostEncoderLite | Ghost Conv | Ghost encoder stem (Layer 0) |
| GhostBasicBlock1 | Ghost Block | CSP-ELAN backbone (stride-2) |
| GhostBasicBlock2 | Ghost Block | CSP-ELAN FPN head (no stride) |
| SDDetect | Detection | Per-scale detection head |
| DFLDecode | Detection | DFL box decoder |
| Dist2BBox | Detection | Distance-to-bounding-box |
| NMS | Detection | Non-Maximum Suppression (post-model) |

## Graph Containers

### Input

Marks the entry point of the graph.

```python
import numpy as np
from talon import ir

# Shape excludes batch dimension
input_node = ir.Input(np.array([784]))          # 1D: 784 features
input_node = ir.Input(np.array([3, 32, 32]))    # 3D: 3x32x32 image
```

### Output

Marks the exit point of the graph.

```python
output_node = ir.Output(np.array([10]))         # 10 classes
```

## Linear Layers

### Affine

Linear transformation: `y = Wx + b`

```python
W = np.random.randn(128, 784).astype(np.float32)  # (out, in)
b = np.zeros(128, dtype=np.float32)

affine = ir.Affine(weight=W, bias=b)

# Properties
print(f"Input shape: {affine.input_type}")   # {'input': array([784])}
print(f"Output shape: {affine.output_type}") # {'output': array([128])}
```

### SpikingAffine

Affine layer with hardware compilation hints.

```python
W = np.random.randn(128, 784).astype(np.float32)
b = np.zeros(128, dtype=np.float32)

spiking_affine = ir.SpikingAffine(
    weight=W,
    bias=b,
    spike_mode='binary',      # 'binary', 'graded', or 'rate'
    weight_bits=8,            # Quantization precision (1-32)
    accumulator_bits=16       # MAC accumulator bits (1-64)
)
```

#### Spike Modes

- **binary**: 0/1 spikes, standard SNN
- **graded**: Multi-level spike values
- **rate**: Rate-coded activations

#### Quantization

`weight_bits` and `accumulator_bits` hint to the compiler:
- Lower bits = smaller memory footprint, faster computation
- `accumulator_bits` must be >= `weight_bits`

## Convolution

### Conv1d

1D convolution for temporal and sequential data processing.

```python
# (out_channels, in_channels, kernel_size)
W = np.random.randn(32, 16, 3).astype(np.float32)
b = np.zeros(32, dtype=np.float32)

conv = ir.Conv1d(
    weight=W,
    bias=b,
    stride=1,
    padding=1,
    dilation=1,
    groups=1
)
```

### Conv2d

2D convolution with configurable stride, padding, dilation, and groups.

```python
# (out_channels, in_channels, kH, kW)
W = np.random.randn(32, 3, 3, 3).astype(np.float32)
b = np.zeros(32, dtype=np.float32)

conv = ir.Conv2d(
    weight=W,
    bias=b,
    stride=(1, 1),
    padding=(1, 1),
    dilation=(1, 1),
    groups=1
)
```

### SepConv2d

Depthwise separable convolution (efficient alternative to Conv2d).

```python
in_ch, out_ch = 32, 64

# Depthwise: (in_ch, 1, kH, kW) - each channel convolved separately
dw = np.random.randn(in_ch, 1, 3, 3).astype(np.float32)
# Pointwise: (out_ch, in_ch, 1, 1) - 1x1 conv to mix channels
pw = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32)

sepconv = ir.SepConv2d(
    depthwise_weight=dw,
    pointwise_weight=pw,
    depthwise_bias=np.zeros(in_ch, dtype=np.float32),
    pointwise_bias=np.zeros(out_ch, dtype=np.float32),
    stride=(1, 1),
    padding=(1, 1)
)
```

### SConv

Spiking Standard 2D Convolution — a `Conv2d` subclass with explicit SNN semantics. Functionally identical to `Conv2d` but marks the convolution as operating in a spiking domain, enabling the compiler to apply spike-aware optimizations.

```python
W = np.random.randn(64, 32, 3, 3).astype(np.float32)
b = np.zeros(64, dtype=np.float32)

sconv = ir.SConv(
    weight=W,
    bias=b,
    stride=(1, 1),
    padding=(1, 1)
)
```

### SDConv

Spiking Depthwise 2D Convolution — a `Conv2d` subclass where `groups=in_channels` is enforced. Each input channel is convolved independently with its own filter, standard in Ghost modules and MobileNet-style architectures.

```python
in_ch = 32
# Depthwise: (in_ch, 1, kH, kW) with groups=in_ch
W = np.random.randn(in_ch, 1, 3, 3).astype(np.float32)
b = np.zeros(in_ch, dtype=np.float32)

sdconv = ir.SDConv(
    weight=W,
    bias=b,
    stride=(1, 1),
    padding=(1, 1)
)
# groups is automatically set to in_channels
```

## Spatial Operations

### MaxPool2d

2D max pooling for downsampling.

```python
pool = ir.MaxPool2d(
    kernel_size=(2, 2),
    stride=(2, 2),      # Defaults to kernel_size if None
    padding=(0, 0),
    dilation=(1, 1),
    ceil_mode=False,
    input_type={'input': np.array([64, 32, 32])}
)
# Input: [64, 32, 32] -> Output: [64, 16, 16]
```

### AvgPool2d

2D average pooling for downsampling with smoother output.

```python
pool = ir.AvgPool2d(
    kernel_size=(2, 2),
    stride=(2, 2),
    padding=(0, 0),
    ceil_mode=False,
    count_include_pad=True,  # Include padding in average calculation
    input_type={'input': np.array([64, 32, 32])}
)
# Input: [64, 32, 32] -> Output: [64, 16, 16]
```

**Pooling Parameters:**
- `kernel_size`: Pooling window `(kH, kW)`
- `stride`: Stride (defaults to `kernel_size`)
- `padding`: Zero padding `(pH, pW)`
- `ceil_mode`: Use ceiling for output size calculation
- `count_include_pad`: Include padding zeros in average calculation (AvgPool2d only)

### Upsample

2D spatial upsampling using interpolation. Commonly used in FPN (Feature Pyramid Network) architectures for multi-scale feature fusion.

```python
# 2x upsampling with nearest neighbor
up = ir.Upsample(
    scale_factor=2,
    mode='nearest',
    input_type={'input': np.array([64, 20, 20])}
)
# Input: [64, 20, 20] -> Output: [64, 40, 40]

# Upsample to explicit size with bilinear interpolation
up = ir.Upsample(
    size=(80, 80),
    mode='bilinear',
    align_corners=True,
    input_type={'input': np.array([128, 40, 40])}
)
# Input: [128, 40, 40] -> Output: [128, 80, 80]
```

**Parameters:**
- `scale_factor`: Multiplier for height and width (e.g., 2 doubles spatial dimensions)
- `size`: Target output size as `(H, W)`. Takes precedence over `scale_factor`.
- `mode`: Interpolation mode - `'nearest'` (faster) or `'bilinear'` (smoother)
- `align_corners`: Align corners for bilinear mode (only applies when `mode='bilinear'`)

#### Upsample Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `nearest` | Nearest neighbor interpolation | Fast, no new values created |
| `bilinear` | Bilinear interpolation | Smoother gradients |

## Reshape

### Flatten

Flattens dimensions in range `[start_dim, end_dim]`.

```python
flatten = ir.Flatten(start_dim=0, end_dim=-1)

# With explicit input type
flatten = ir.Flatten(
    start_dim=0,
    end_dim=-1,
    input_type={'input': np.array([32, 8, 8])}
)
# Output: [2048] (32 * 8 * 8)
```

## Neurons

### LIF

Leaky integrate-and-fire neuron implementing NIR-compliant dynamics:

```
tau * dv/dt = (v_leak - v) + r * I
spike when v >= v_threshold
reset to 0 on spike
```

```python
n_neurons = 128

lif = ir.LIF(
    tau=np.ones(n_neurons) * 10.0,      # Time constant
    r=np.ones(n_neurons) * 10.0,        # Membrane resistance
    v_leak=np.zeros(n_neurons),          # Leak potential
    v_threshold=np.ones(n_neurons)       # Spike threshold
)
```

#### snnTorch Conversion

When exporting `snn.Leaky(beta=0.9)`:

```python
# snnTorch: beta = 0.9
# TALON IR: tau = 1/(1-beta) = 10, r = tau = 10, v_leak = 0

# This ensures identical dynamics:
# snnTorch: mem = beta*mem + x
# TALON IR: mem = beta*mem + (1-beta)*(v_leak + r*x)
#      = 0.9*mem + 0.1*(0 + 10*x)
#      = 0.9*mem + x  ✓
```

### IF

Integrate-and-Fire neuron (no leak). Simpler than LIF — membrane potential integrates input without decay. Spikes deterministically when `v >= v_threshold`, then resets to zero.

```
dv/dt = i
spike when v >= v_threshold
reset to 0 on spike
```

```python
if_neuron = ir.IF(v_threshold=np.ones(128))
```

## Skip

Residual/skip connections for multi-branch architectures. The `skip_type` determines how multiple inputs are merged.

```python
skip = ir.Skip(
    skip_type='residual',  # 'residual', 'concatenate', or 'passthrough'
    input_type={'input': np.array([128])}
)
```

#### Skip Types

| Type | Operation | Description |
|------|-----------|-------------|
| `passthrough` | Identity | Single input passes through unchanged |
| `residual` | Add | Element-wise addition of all inputs |
| `concatenate` | Concat | Channel concatenation (dim=1 for NCHW) |

#### ResNet-style Residual Block

```python
nodes = {
    'input': ir.Input(np.array([64, 32, 32])),
    'conv1': ir.Conv2d(weight=W1, bias=b1),
    'lif1': ir.LIF(...),
    'conv2': ir.Conv2d(weight=W2, bias=b2),
    'skip': ir.Skip(skip_type='residual'),  # Adds conv2 output + input
    'lif2': ir.LIF(...),
    'output': ir.Output(...),
}

edges = [
    ('input', 'conv1'),
    ('conv1', 'lif1'),
    ('lif1', 'conv2'),
    ('conv2', 'skip'),   # Main path
    ('input', 'skip'),   # Residual path (element-wise add)
    ('skip', 'lif2'),
    ('lif2', 'output'),
]
```

#### SPP-style Concatenation

```python
nodes = {
    'input': ir.Input(np.array([64, 20, 20])),
    'pool5': ir.MaxPool2d(kernel_size=(5,5), stride=(1,1), padding=(2,2)),
    'pool9': ir.MaxPool2d(kernel_size=(9,9), stride=(1,1), padding=(4,4)),
    'concat': ir.Skip(skip_type='concatenate'),  # Channel concat
    'output': ir.Output(np.array([192, 20, 20])),  # 64*3 = 192 channels
}

edges = [
    ('input', 'pool5'),
    ('input', 'pool9'),
    ('input', 'concat'),    # Original features (64ch)
    ('pool5', 'concat'),    # Pooled features (64ch)
    ('pool9', 'concat'),    # Pooled features (64ch)
    ('concat', 'output'),   # Concatenated (192ch)
]
```

#### RepConv Multi-Branch

```python
# RepConv: 3 parallel branches merged via residual add
nodes = {
    'input': ir.Input(np.array([64, 32, 32])),
    'conv3x3': ir.Conv2d(weight=W3, bias=b3, padding=(1,1)),
    'conv1x1': ir.Conv2d(weight=W1, bias=b1, padding=(0,0)),
    'identity': ir.Skip(skip_type='passthrough'),
    'merge': ir.Skip(skip_type='residual'),  # Element-wise add all branches
    'output': ir.Output(np.array([64, 32, 32])),
}

edges = [
    ('input', 'conv3x3'),
    ('input', 'conv1x1'),
    ('input', 'identity'),
    ('conv3x3', 'merge'),
    ('conv1x1', 'merge'),
    ('identity', 'merge'),
    ('merge', 'output'),
]
```

#### FPN (Feature Pyramid Network) Neck

Upsample + Skip for multi-scale feature fusion:

```python
# FPN neck: upsample high-level features and fuse with low-level features
nodes = {
    'p4': ir.Input(np.array([256, 40, 40])),    # High-level features
    'p3': ir.Input(np.array([128, 80, 80])),    # Low-level features
    'reduce': ir.Conv2d(weight=W_reduce, bias=b_reduce),  # 256 -> 128 channels
    'up': ir.Upsample(scale_factor=2, mode='nearest'),    # 40x40 -> 80x80
    'concat': ir.Skip(skip_type='concatenate'),            # 128 + 128 = 256 channels
    'out_conv': ir.Conv2d(weight=W_out, bias=b_out),      # Process fused features
    'output': ir.Output(np.array([128, 80, 80])),
}

edges = [
    ('p4', 'reduce'),
    ('reduce', 'up'),
    ('up', 'concat'),      # Upsampled high-level features
    ('p3', 'concat'),      # Low-level features
    ('concat', 'out_conv'),
    ('out_conv', 'output'),
]
```

## ANN Activations

For hybrid ANN-SNN architectures, TALON IR supports common ANN activation functions. These are used in encoder/decoder layers that operate in rate-based mode.

### ReLU

Rectified Linear Unit activation. Supports LeakyReLU via negative_slope parameter.

```python
# Standard ReLU
relu = ir.ReLU(features=128)

# Leaky ReLU with negative slope
leaky = ir.ReLU(features=128, negative_slope=0.01)
```

### Sigmoid

Sigmoid activation, outputs values in (0, 1).

```python
sigmoid = ir.Sigmoid(features=128)
```

### Tanh

Hyperbolic tangent activation, outputs values in (-1, 1).

```python
tanh = ir.Tanh(features=128)
```

### Softmax

Softmax activation for classification outputs.

```python
# 10-class classification
softmax = ir.Softmax(features=10)

# Softmax along specific dimension
softmax = ir.Softmax(features=100, dim=1)
```

### GELU

Gaussian Error Linear Unit, common in transformers.

```python
gelu = ir.GELU(features=256)
gelu_exact = ir.GELU(features=256, approximate=False)
```

### ELU

Exponential Linear Unit with configurable alpha.

```python
elu = ir.ELU(features=128)
elu_scaled = ir.ELU(features=128, alpha=0.5)
```

### PReLU

Parametric ReLU with learnable negative slope.

```python
# Shared weight
prelu = ir.PReLU(features=128, weight=np.array([0.25]))

# Per-channel weights
prelu = ir.PReLU(features=128, weight=np.full(128, 0.25))
```

## Normalization

Normalization layers for hybrid architectures.

### BatchNorm1d

Batch normalization for linear layers.

```python
bn = ir.BatchNorm1d(
    num_features=128,
    weight=np.ones(128),           # gamma
    bias=np.zeros(128),            # beta
    running_mean=np.zeros(128),
    running_var=np.ones(128),
    eps=1e-5,
    momentum=0.1
)
```

### BatchNorm2d

Batch normalization for convolutional layers.

```python
bn = ir.BatchNorm2d(
    num_features=64,
    weight=np.ones(64),
    bias=np.zeros(64),
    running_mean=np.zeros(64),
    running_var=np.ones(64)
)
```

### LayerNorm

Layer normalization, common in transformers.

```python
ln = ir.LayerNorm(
    normalized_shape=[256],
    weight=np.ones(256),
    bias=np.zeros(256)
)
```

## Regularization

### Dropout

Dropout regularization layer.

```python
dropout = ir.Dropout(features=256, p=0.1)
```

Note: Dropout is typically a no-op during inference.

## Hybrid Architecture Support

### HybridRegion

Marker node to identify ANN vs SNN regions in hybrid architectures.

```python
from talon import ir
from talon.ir import NeuronMode

# Mark the start of an ANN encoder
encoder_start = ir.HybridRegion(
    mode='ann',
    features=256,
    name='encoder'
)

# Mark transition to SNN processing
snn_start = ir.HybridRegion(
    mode='snn',
    features=256,
    name='snn_core'
)
```

### Hybrid ANN-SNN Example

A typical encoder-SNN-decoder architecture:

```python
nodes = {
    'input': ir.Input(np.array([784])),
    # ANN Encoder
    'fc1': ir.Affine(weight=w1, bias=b1),
    'bn1': ir.BatchNorm1d(num_features=256, weight=g1, bias=b1, ...),
    'relu1': ir.ReLU(features=256),
    'snn_region': ir.HybridRegion(mode='snn', features=256),
    # SNN Core
    'fc2': ir.Affine(weight=w2, bias=b2),
    'lif': ir.LIF(tau=tau, r=r, v_leak=vl, v_threshold=vt),
    'ann_region': ir.HybridRegion(mode='ann', features=128),
    # ANN Decoder
    'fc3': ir.Affine(weight=w3, bias=b3),
    'softmax': ir.Softmax(features=10),
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'fc1'), ('fc1', 'bn1'), ('bn1', 'relu1'),
    ('relu1', 'snn_region'), ('snn_region', 'fc2'),
    ('fc2', 'lif'), ('lif', 'ann_region'),
    ('ann_region', 'fc3'), ('fc3', 'softmax'),
    ('softmax', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Serialization

### Write Graph

```python
ir.write('model.t1c', graph)
```

### Read Graph

```python
graph = ir.read('model.t1c')
```

### Check Version

```python
version = ir.read_version('model.t1c')
print(version)  # '0.0.1'
```

## Ghost / Detect Primitives (SU-YOLO Mid-Ghost)

These primitives support the SU-YOLO spiking object detection architecture built on Ghost modules. Ghost convolutions generate feature maps cheaply by performing a primary convolution followed by a depthwise "cheap" convolution, then concatenating both outputs to double the channel count at minimal cost.

### ChannelSplit

Splits a tensor along the channel dimension into multiple chunks.

```python
split = ir.ChannelSplit(
    split_sections=[32, 32],   # Two equal chunks
    dim=1,                      # Channel dimension
    input_type={'input': np.array([64, 40, 40])}
)
# Output: 2 tensors of shape [32, 40, 40]
```

### Concat

Concatenates multiple tensors along the channel dimension.

```python
concat = ir.Concat(
    num_inputs=2,
    dim=1,
    input_type={'input': np.array([32, 40, 40])}
)
```

### SGhostConv

Spiking Ghost Convolution: primary conv followed by cheap depthwise conv, concatenated.

```python
Cp = 32  # Primary output channels (total output = 2*Cp = 64)
C_in = 16

sghost = ir.SGhostConv(
    primary_weight=np.random.randn(Cp, C_in, 3, 3).astype(np.float32),
    primary_bias=np.zeros(Cp, dtype=np.float32),
    cheap_weight=np.random.randn(Cp, 1, 3, 3).astype(np.float32),
    cheap_bias=np.zeros(Cp, dtype=np.float32),
    primary_stride=(1, 1),
    primary_padding=(1, 1),
)
# Output channels = 2 * Cp = 64
```

### SGhostEncoderLite

Stem encoder (Layer 0) in SU-YOLO Mid-Ghost. Performs a standard conv followed by an SGhostConv with halved concatenation for better parameter efficiency.

```python
encoder = ir.SGhostEncoderLite(
    conv1_weight=np.random.randn(16, 3, 3, 3).astype(np.float32),
    conv1_bias=np.zeros(16, dtype=np.float32),
    ghost_primary_weight=np.random.randn(16, 16, 3, 3).astype(np.float32),
    ghost_primary_bias=np.zeros(16, dtype=np.float32),
    ghost_cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32),
    ghost_cheap_bias=np.zeros(16, dtype=np.float32),
    stride=2,
)
```

### GhostBasicBlock1

CSP-ELAN Ghost bottleneck block for backbone (stride-2). Contains three SGhostConv sub-modules (cv0, cvres, cv2) with a channel split and residual path.

```python
C1, C2 = 32, 64
Cp0 = C2 // 2   # Primary channels for cv0

gbb1 = ir.GhostBasicBlock1(
    cv0_primary_weight=np.random.randn(Cp0, C1, 3, 3).astype(np.float32),
    cv0_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
    cvres_primary_weight=np.random.randn(C2 // 4, C2, 1, 1).astype(np.float32),
    cvres_cheap_weight=np.random.randn(C2 // 4, 1, 3, 3).astype(np.float32),
    cv2_primary_weight=np.random.randn(Cp0, Cp0, 1, 1).astype(np.float32),
    cv2_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
    stride=2,
)
# Input: [C1, H, W] -> Output: [C2, H/2, W/2]
```

### GhostBasicBlock2

CSP-ELAN Ghost bottleneck block for FPN head (no stride). Same structure as Block1 but without the residual downsampling branch.

```python
C1, C2 = 64, 64
Cp0 = C2 // 2

gbb2 = ir.GhostBasicBlock2(
    cv0_primary_weight=np.random.randn(Cp0, C1, 3, 3).astype(np.float32),
    cv0_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
    cv2_primary_weight=np.random.randn(Cp0, Cp0, 1, 1).astype(np.float32),
    cv2_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
)
# Input: [C1, H, W] -> Output: [C2, H, W]
```

### SDDetect

Per-scale spiking object detection head. Contains parallel branches for bounding box regression (cv2) and classification (cv3).

```python
detect = ir.SDDetect(
    num_classes=80,
    reg_max=16,
    stride=8,
    cv2_w0=np.random.randn(64, 64, 3, 3).astype(np.float32),
    cv2_b0=np.zeros(64, dtype=np.float32),
    cv2_w1=np.random.randn(64, 64, 3, 3).astype(np.float32),
    cv2_b1=np.zeros(64, dtype=np.float32),
    cv2_w2=np.random.randn(64, 64, 1, 1).astype(np.float32),
    cv2_b2=np.zeros(64, dtype=np.float32),
    cv3_w0=np.random.randn(80, 64, 3, 3).astype(np.float32),
    cv3_b0=np.zeros(80, dtype=np.float32),
    cv3_w1=np.random.randn(80, 80, 3, 3).astype(np.float32),
    cv3_b1=np.zeros(80, dtype=np.float32),
    cv3_w2=np.random.randn(80, 80, 1, 1).astype(np.float32),
    cv3_b2=np.zeros(80, dtype=np.float32),
)
```

### DFLDecode

Distribution Focal Loss decoding. Converts DFL logits into box distance values using a weighted softmax.

```python
dfl = ir.DFLDecode(reg_max=16)
```

### Dist2BBox

Converts distance predictions (top, left, bottom, right) to bounding box format (x_center, y_center, w, h) with optional anchor point generation.

```python
d2b = ir.Dist2BBox(stride=8)
```

| Primitive | Category | In Architecture |
|-----------|----------|-----------------|
| ChannelSplit | Routing | Feature channel splitting for CSP-ELAN |
| Concat | Routing | Feature channel concatenation |
| SGhostConv | Ghost Conv | Primary + cheap depthwise concatenation |
| SGhostEncoderLite | Ghost Conv | Stem encoder (Layer 0) |
| GhostBasicBlock1 | Ghost Block | Backbone block (stride-2) |
| GhostBasicBlock2 | Ghost Block | FPN head block (no stride) |
| SDDetect | Detection | Per-scale detection head |
| DFLDecode | Detection | DFL box decoder |
| Dist2BBox | Detection | Distance to bounding box |
| NMS | Detection | Non-Maximum Suppression (post-model) |

### NMS

Non-Maximum Suppression for filtering overlapping detections. Runs on the ARM CPU post-accelerator (not synthesized to HLS). Carries no learned weights — purely parametric.

Algorithm: O(N log N) score sort + O(N * K) greedy IoU suppression.

```python
nms = ir.NMS(
    score_threshold=0.25,   # Minimum confidence to keep
    iou_threshold=0.45,     # IoU overlap threshold for suppression
    max_detections=300,     # Cap on output count (0 = unlimited)
)

# Apply to detections: (N, 5+num_classes) where columns are
# [cx, cy, w, h, objectness, cls_0, cls_1, ...]
detections = np.random.randn(100, 85).astype(np.float32)
detections[:, 4] = np.random.rand(100)  # objectness scores
kept = nms(detections)
```

### SU-YOLO Detection Pipeline

The full detection pipeline chains these primitives:

```
GhostBasicBlock1/2 → SDDetect → DFLDecode → Dist2BBox → NMS
```

1. **SDDetect**: Splits features into box regression (cv2) and classification (cv3) branches
2. **DFLDecode**: Converts DFL logits to distance values via weighted softmax
3. **Dist2BBox**: Converts (l, t, r, b) distances + anchors to (cx, cy, w, h) pixel boxes
4. **NMS**: Filters overlapping boxes, keeps highest-confidence per region

## Custom Primitives

Register custom node types:

```python
from talon.ir import Node, register_node
from dataclasses import dataclass

@register_node
@dataclass(eq=False)
class CustomNeuron(Node):
    tau: np.ndarray
    custom_param: float = 1.0
    input_type: dict = None
    output_type: dict = None

# Now usable
ir.str_to_node('CustomNeuron')  # Returns CustomNeuron class
```


