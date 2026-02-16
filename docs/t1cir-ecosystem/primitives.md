---
sidebar_position: 5
---

# Primitives Reference

T1C-IR defines 22+ primitives for representing spiking neural networks and hybrid ANN-SNN architectures. Each primitive maps directly to hardware operations on Type 1 Compute chips.

## Primitive Overview

| Primitive | Category | Description |
|-----------|----------|-------------|
| Affine | Linear | Linear transform y = Wx + b |
| SpikingAffine | Linear | Quantized affine with spike hints |
| Conv2d | Convolution | 2D convolution |
| SepConv2d | Convolution | Depthwise separable convolution |
| MaxPool2d | Spatial | 2D max pooling (downsampling) |
| AvgPool2d | Spatial | 2D average pooling (downsampling) |
| Upsample | Spatial | 2D upsampling (nearest/bilinear) |
| Flatten | Reshape | Reshape to 1D |
| LIF | Neuron | Leaky integrate-and-fire |
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

## Graph Containers

### Input

Marks the entry point of the graph.

```python
import numpy as np
from t1c import ir

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
# T1C-IR: tau = 1/(1-beta) = 10, r = tau = 10, v_leak = 0

# This ensures identical dynamics:
# snnTorch: mem = beta*mem + x
# T1C-IR: mem = beta*mem + (1-beta)*(v_leak + r*x)
#      = 0.9*mem + 0.1*(0 + 10*x)
#      = 0.9*mem + x  ✓
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

For hybrid ANN-SNN architectures, T1C-IR supports common ANN activation functions. These are used in encoder/decoder layers that operate in rate-based mode.

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
from t1c import ir
from t1c.ir import NeuronMode

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

## Custom Primitives

Register custom node types:

```python
from t1c.ir import Node, register_node
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


