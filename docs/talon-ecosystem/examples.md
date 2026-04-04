---
sidebar_position: 9
---

# Examples

Real-world TALON IR patterns and architectures. For step-by-step walkthroughs, see the [Tutorials](#tutorials) section below.

## Tutorials

| Tutorial | Topics |
|----------|--------|
| [TALON IR Basics](./tutorial/tutorial_t1cir_basics) | Primitives, graphs, serialization, ghost & detection |
| [T1CViz](./tutorial/tutorial_t1cviz) | Graph visualization, event processing, pattern detection |
| [TALON SDK](./tutorial/tutorial_talon) | Analysis, profiling, linting, fingerprinting |
| [Bridge](./tutorial/tutorial_bridge) | Export/import, stateful LIF, CyclicGraphExecutor |
| [Ghost & Detection](./tutorial/tutorial_ghost_detection) | GhostNet architecture, detection pipeline |
| [Hardware Mapping](./tutorial/tutorial_hardware_mapping) | Partition, allocate, route, place |
| [Backend](./tutorial/tutorial_backend) | Simulate, profile, energy presets, FPGA |
| [Event I/O](./tutorial/tutorial_event_io) | Neural encoding, HDF5, throughput |
| [snnTorch Integration](./tutorial/tutorial_snntorch_integration) | snnTorch export/import round-trip with SDK analysis |
| [End-to-End Pipeline](./tutorial/tutorial_end_to_end) | Full workflow: model → export → analyze → partition → simulate → profile |

---

## Simple Feedforward SNN

Two-layer fully connected SNN for classification.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([784])),
    'fc1': ir.Affine(
        weight=np.random.randn(128, 784).astype(np.float32),
        bias=np.zeros(128, dtype=np.float32)
    ),
    'lif1': ir.LIF(
        tau=np.ones(128) * 10.0,
        r=np.ones(128) * 10.0,
        v_leak=np.zeros(128),
        v_threshold=np.ones(128)
    ),
    'fc2': ir.Affine(
        weight=np.random.randn(10, 128).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    'lif2': ir.LIF(
        tau=np.ones(10) * 10.0,
        r=np.ones(10) * 10.0,
        v_leak=np.zeros(10),
        v_threshold=np.ones(10)
    ),
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'fc1'),
    ('fc1', 'lif1'),
    ('lif1', 'fc2'),
    ('fc2', 'lif2'),
    ('lif2', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Convolutional SNN

Conv-pool-flatten-FC architecture for image classification.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([1, 32, 32])),  # 1-channel 32x32
    
    # Conv block
    'conv1': ir.Conv2d(
        weight=np.random.randn(16, 1, 3, 3).astype(np.float32),
        bias=np.zeros(16, dtype=np.float32),
        stride=(1, 1),
        padding=(1, 1)
    ),
    'lif1': ir.LIF(
        tau=np.ones(16) * 10.0,
        r=np.ones(16) * 10.0,
        v_leak=np.zeros(16),
        v_threshold=np.ones(16)
    ),
    'pool1': ir.MaxPool2d(
        kernel_size=(2, 2),
        stride=(2, 2)
    ),
    
    # Flatten and classify
    'flatten': ir.Flatten(start_dim=0, end_dim=-1),
    'fc1': ir.Affine(
        weight=np.random.randn(10, 16*16*16).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    'lif2': ir.LIF(
        tau=np.ones(10) * 10.0,
        r=np.ones(10) * 10.0,
        v_leak=np.zeros(10),
        v_threshold=np.ones(10)
    ),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'conv1'),
    ('conv1', 'lif1'),
    ('lif1', 'pool1'),
    ('pool1', 'flatten'),
    ('flatten', 'fc1'),
    ('fc1', 'lif2'),
    ('lif2', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Deep 3-Layer SNN

Multi-layer feedforward for complex classification.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([784])),
    
    # Layer 1
    'fc1': ir.Affine(
        weight=np.random.randn(256, 784).astype(np.float32),
        bias=np.zeros(256, dtype=np.float32)
    ),
    'lif1': ir.LIF(
        tau=np.ones(256) * 10.0,
        r=np.ones(256) * 10.0,
        v_leak=np.zeros(256),
        v_threshold=np.ones(256)
    ),
    
    # Layer 2
    'fc2': ir.Affine(
        weight=np.random.randn(128, 256).astype(np.float32),
        bias=np.zeros(128, dtype=np.float32)
    ),
    'lif2': ir.LIF(
        tau=np.ones(128) * 10.0,
        r=np.ones(128) * 10.0,
        v_leak=np.zeros(128),
        v_threshold=np.ones(128)
    ),
    
    # Layer 3
    'fc3': ir.Affine(
        weight=np.random.randn(10, 128).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    'lif3': ir.LIF(
        tau=np.ones(10) * 10.0,
        r=np.ones(10) * 10.0,
        v_leak=np.zeros(10),
        v_threshold=np.ones(10)
    ),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'fc1'), ('fc1', 'lif1'),
    ('lif1', 'fc2'), ('fc2', 'lif2'),
    ('lif2', 'fc3'), ('fc3', 'lif3'),
    ('lif3', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Skip/Residual Connection

Residual architecture with skip connection.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([128])),
    
    # Main path
    'fc1': ir.Affine(
        weight=np.random.randn(128, 128).astype(np.float32),
        bias=np.zeros(128, dtype=np.float32)
    ),
    'lif1': ir.LIF(
        tau=np.ones(128) * 10.0,
        r=np.ones(128) * 10.0,
        v_leak=np.zeros(128),
        v_threshold=np.ones(128)
    ),
    
    # Skip path
    'skip': ir.Skip(
        skip_type='residual',
        input_type={'input': np.array([128])}
    ),
    
    'output': ir.Output(np.array([128]))
}

# Both paths merge at output
edges = [
    ('input', 'fc1'),
    ('fc1', 'lif1'),
    ('lif1', 'output'),     # Main path
    ('input', 'skip'),
    ('skip', 'output')       # Skip path (adds to main)
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Depthwise Separable Conv

Efficient convolution using SepConv2d.

```python
import numpy as np
from talon import ir

in_ch, out_ch = 32, 64

nodes = {
    'input': ir.Input(np.array([in_ch, 16, 16])),
    
    'sepconv': ir.SepConv2d(
        depthwise_weight=np.random.randn(in_ch, 1, 3, 3).astype(np.float32),
        pointwise_weight=np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32),
        depthwise_bias=np.zeros(in_ch, dtype=np.float32),
        pointwise_bias=np.zeros(out_ch, dtype=np.float32),
        stride=(1, 1),
        padding=(1, 1)
    ),
    'lif': ir.LIF(
        tau=np.ones(out_ch) * 10.0,
        r=np.ones(out_ch) * 10.0,
        v_leak=np.zeros(out_ch),
        v_threshold=np.ones(out_ch)
    ),
    
    'output': ir.Output(np.array([out_ch, 16, 16]))
}

edges = [
    ('input', 'sepconv'),
    ('sepconv', 'lif'),
    ('lif', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## RepConv (Multi-Branch)

Multi-branch architecture with parallel 3x3, 1x1, and identity paths.

```python
import numpy as np
from talon import ir

ch = 8

nodes = {
    'input': ir.Input(np.array([ch, 35, 35])),
    
    # Branch 1: 3x3 conv
    'conv3x3': ir.Conv2d(
        weight=np.random.randn(ch, ch, 3, 3).astype(np.float32),
        bias=np.zeros(ch, dtype=np.float32),
        padding=(1, 1)
    ),
    
    # Branch 2: 1x1 conv
    'conv1x1': ir.Conv2d(
        weight=np.random.randn(ch, ch, 1, 1).astype(np.float32),
        bias=np.zeros(ch, dtype=np.float32),
        padding=(0, 0)
    ),
    
    # Branch 3: identity skip
    'identity': ir.Skip(
        skip_type='residual',
        input_type={'input': np.array([ch, 35, 35])}
    ),
    
    # Merge point
    'merge': ir.Skip(
        skip_type='residual',
        input_type={'input': np.array([ch, 35, 35])}
    ),
    
    # LIF after merge
    'lif': ir.LIF(
        tau=np.ones(ch) * 10.0,
        r=np.ones(ch) * 10.0,
        v_leak=np.zeros(ch),
        v_threshold=np.ones(ch)
    ),
    
    'output': ir.Output(np.array([ch, 35, 35]))
}

# Multi-branch topology
edges = [
    ('input', 'conv3x3'),    # Branch 1
    ('input', 'conv1x1'),    # Branch 2
    ('input', 'identity'),   # Branch 3
    ('conv3x3', 'merge'),
    ('conv1x1', 'merge'),
    ('identity', 'merge'),
    ('merge', 'lif'),
    ('lif', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## SPP Design Pattern (using MaxPool2d)

Multi-scale feature extraction using parallel MaxPool2d operations. SPP is a design pattern, not a primitive - it's composed from existing primitives.

```python
# SPP = concat(original, pool_5x5, pool_9x9, pool_13x13)
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([64, 20, 20])),
    
    # Branch 1: Identity (original features)
    'identity': ir.Skip(
        input_type={'input': np.array([64, 20, 20])},
        skip_type='passthrough'
    ),
    
    # Branch 2: 5x5 pooling (padding=2 preserves spatial dims)
    'pool5': ir.MaxPool2d(
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        input_type={'input': np.array([64, 20, 20])}
    ),
    
    # Branch 3: 9x9 pooling
    'pool9': ir.MaxPool2d(
        kernel_size=(9, 9),
        stride=(1, 1),
        padding=(4, 4),
        input_type={'input': np.array([64, 20, 20])}
    ),
    
    # Branch 4: 13x13 pooling
    'pool13': ir.MaxPool2d(
        kernel_size=(13, 13),
        stride=(1, 1),
        padding=(6, 6),
        input_type={'input': np.array([64, 20, 20])}
    ),
    
    # Merge point (channel concatenation)
    'merge': ir.Skip(
        input_type={'input': np.array([256, 20, 20])},  # 64*4
        skip_type='concatenate'
    ),
    
    'output': ir.Output(np.array([256, 20, 20]))
}

# Multi-branch topology
edges = [
    ('input', 'identity'),
    ('input', 'pool5'),
    ('input', 'pool9'),
    ('input', 'pool13'),
    ('identity', 'merge'),
    ('pool5', 'merge'),
    ('pool9', 'merge'),
    ('pool13', 'merge'),
    ('merge', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## SPPF Design Pattern (using MaxPool2d)

Efficient sequential alternative to SPP. Applies the same kernel repeatedly.

```python
# SPPF = concat(x, pool(x), pool(pool(x)), pool(pool(pool(x))))
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([64, 20, 20])),
    
    # Original features (branch 0)
    'identity': ir.Skip(
        input_type={'input': np.array([64, 20, 20])},
        skip_type='passthrough'
    ),
    
    # Sequential pooling chain
    'pool1': ir.MaxPool2d(
        kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
        input_type={'input': np.array([64, 20, 20])}
    ),
    'pool2': ir.MaxPool2d(
        kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
        input_type={'input': np.array([64, 20, 20])}
    ),
    'pool3': ir.MaxPool2d(
        kernel_size=(5, 5), stride=(1, 1), padding=(2, 2),
        input_type={'input': np.array([64, 20, 20])}
    ),
    
    # Merge point (channel concatenation)
    'merge': ir.Skip(
        input_type={'input': np.array([256, 20, 20])},
        skip_type='concatenate'
    ),
    
    'output': ir.Output(np.array([256, 20, 20]))
}

# Sequential chain with all outputs merged
edges = [
    ('input', 'identity'),   # Branch 0: original
    ('input', 'pool1'),      # Branch 1: 5x5 effective
    ('pool1', 'pool2'),      # Chain
    ('pool2', 'pool3'),      # Chain
    ('identity', 'merge'),
    ('pool1', 'merge'),      # Branch 1 output
    ('pool2', 'merge'),      # Branch 2: 9x9 effective
    ('pool3', 'merge'),      # Branch 3: 13x13 effective
    ('merge', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)

# SPPF is more efficient than SPP:
# - SPP: 3 different kernel computations in parallel
# - SPPF: Same kernel reused sequentially (fewer unique ops)
```

## Using AvgPool2d

Average pooling for smoother downsampling.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([32, 16, 16])),
    
    'conv': ir.Conv2d(
        weight=np.random.randn(64, 32, 3, 3).astype(np.float32),
        bias=np.zeros(64, dtype=np.float32),
        padding=(1, 1)
    ),
    'lif': ir.LIF(
        tau=np.ones(64) * 10.0,
        r=np.ones(64) * 10.0,
        v_leak=np.zeros(64),
        v_threshold=np.ones(64)
    ),
    
    # AvgPool2d instead of MaxPool2d for smoother output
    'pool': ir.AvgPool2d(
        kernel_size=(2, 2),
        stride=(2, 2),
        count_include_pad=True
    ),
    
    'output': ir.Output(np.array([64, 8, 8]))
}

edges = [
    ('input', 'conv'),
    ('conv', 'lif'),
    ('lif', 'pool'),
    ('pool', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Hardware-Optimized Export

Using SpikingAffine for quantization hints.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([784])),
    
    'fc1': ir.SpikingAffine(
        weight=np.random.randn(128, 784).astype(np.float32),
        bias=np.zeros(128, dtype=np.float32),
        spike_mode='binary',
        weight_bits=8,
        accumulator_bits=16
    ),
    'lif1': ir.LIF(
        tau=np.ones(128) * 10.0,
        r=np.ones(128) * 10.0,
        v_leak=np.zeros(128),
        v_threshold=np.ones(128)
    ),
    
    'fc2': ir.SpikingAffine(
        weight=np.random.randn(10, 128).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32),
        spike_mode='binary',
        weight_bits=8,
        accumulator_bits=16
    ),
    'lif2': ir.LIF(
        tau=np.ones(10) * 10.0,
        r=np.ones(10) * 10.0,
        v_leak=np.zeros(10),
        v_threshold=np.ones(10)
    ),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'fc1'), ('fc1', 'lif1'),
    ('lif1', 'fc2'), ('fc2', 'lif2'),
    ('lif2', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)

# Serialize
ir.write('model_quantized.t1c', graph)
```

## Hybrid ANN-SNN Architecture

Encoder-SNN-Decoder pattern for hybrid processing. ANN layers handle preprocessing and output, SNN handles temporal processing.

```python
import numpy as np
from talon import ir

# Weight placeholders (in practice, these come from trained models)
w_enc = np.random.randn(256, 784).astype(np.float32)
b_enc = np.zeros(256, dtype=np.float32)
w_snn = np.random.randn(128, 256).astype(np.float32)
b_snn = np.zeros(128, dtype=np.float32)
w_dec = np.random.randn(10, 128).astype(np.float32)
b_dec = np.zeros(10, dtype=np.float32)

nodes = {
    'input': ir.Input(np.array([784])),
    
    # ANN Encoder
    'fc_enc': ir.Affine(weight=w_enc, bias=b_enc),
    'bn_enc': ir.BatchNorm1d(
        num_features=256,
        weight=np.ones(256, dtype=np.float32),
        bias=np.zeros(256, dtype=np.float32),
        running_mean=np.zeros(256, dtype=np.float32),
        running_var=np.ones(256, dtype=np.float32)
    ),
    'relu_enc': ir.ReLU(features=256),
    'dropout_enc': ir.Dropout(features=256, p=0.1),
    
    # Mark transition to SNN
    'snn_region': ir.HybridRegion(mode='snn', features=256, name='snn_core'),
    
    # SNN Core
    'fc_snn': ir.Affine(weight=w_snn, bias=b_snn),
    'lif': ir.LIF(
        tau=np.ones(128) * 10.0,
        r=np.ones(128) * 10.0,
        v_leak=np.zeros(128),
        v_threshold=np.ones(128)
    ),
    
    # Mark transition back to ANN
    'ann_region': ir.HybridRegion(mode='ann', features=128, name='decoder'),
    
    # ANN Decoder
    'fc_dec': ir.Affine(weight=w_dec, bias=b_dec),
    'softmax': ir.Softmax(features=10, dim=-1),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    # Encoder
    ('input', 'fc_enc'),
    ('fc_enc', 'bn_enc'),
    ('bn_enc', 'relu_enc'),
    ('relu_enc', 'dropout_enc'),
    ('dropout_enc', 'snn_region'),
    # SNN Core
    ('snn_region', 'fc_snn'),
    ('fc_snn', 'lif'),
    ('lif', 'ann_region'),
    # Decoder
    ('ann_region', 'fc_dec'),
    ('fc_dec', 'softmax'),
    ('softmax', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)

# Visualize to see the hybrid structure
from talon import viz
viz.visualize(graph, title="Hybrid ANN-SNN Architecture")
```

## Using Normalization Layers

Batch normalization for training stability.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([784])),
    
    # FC + BatchNorm1d + ReLU pattern
    'fc1': ir.Affine(
        weight=np.random.randn(256, 784).astype(np.float32),
        bias=np.zeros(256, dtype=np.float32)
    ),
    'bn1': ir.BatchNorm1d(
        num_features=256,
        weight=np.ones(256, dtype=np.float32),   # gamma
        bias=np.zeros(256, dtype=np.float32),    # beta
        running_mean=np.zeros(256, dtype=np.float32),
        running_var=np.ones(256, dtype=np.float32),
        eps=1e-5,
        momentum=0.1
    ),
    'relu1': ir.ReLU(features=256),
    
    # Second layer with LIF
    'fc2': ir.Affine(
        weight=np.random.randn(10, 256).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    'lif': ir.LIF(
        tau=np.ones(10) * 10.0,
        r=np.ones(10) * 10.0,
        v_leak=np.zeros(10),
        v_threshold=np.ones(10)
    ),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'fc1'),
    ('fc1', 'bn1'),
    ('bn1', 'relu1'),
    ('relu1', 'fc2'),
    ('fc2', 'lif'),
    ('lif', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Conv2d with BatchNorm2d

Convolutional architecture with batch normalization.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([3, 32, 32])),  # RGB 32x32
    
    # Conv + BatchNorm2d + ReLU
    'conv1': ir.Conv2d(
        weight=np.random.randn(64, 3, 3, 3).astype(np.float32),
        bias=np.zeros(64, dtype=np.float32),
        padding=(1, 1)
    ),
    'bn1': ir.BatchNorm2d(
        num_features=64,
        weight=np.ones(64, dtype=np.float32),
        bias=np.zeros(64, dtype=np.float32),
        running_mean=np.zeros(64, dtype=np.float32),
        running_var=np.ones(64, dtype=np.float32)
    ),
    'relu1': ir.ReLU(features=64*32*32),
    'pool1': ir.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    
    'flatten': ir.Flatten(start_dim=0, end_dim=-1),
    'fc': ir.Affine(
        weight=np.random.randn(10, 64*16*16).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    
    'output': ir.Output(np.array([10]))
}

edges = [
    ('input', 'conv1'),
    ('conv1', 'bn1'),
    ('bn1', 'relu1'),
    ('relu1', 'pool1'),
    ('pool1', 'flatten'),
    ('flatten', 'fc'),
    ('fc', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Transformer-style with LayerNorm

LayerNorm for attention-based architectures.

```python
import numpy as np
from talon import ir

nodes = {
    'input': ir.Input(np.array([256])),  # 256-dim embedding
    
    # Pre-norm pattern
    'ln1': ir.LayerNorm(
        normalized_shape=[256],
        weight=np.ones(256, dtype=np.float32),
        bias=np.zeros(256, dtype=np.float32)
    ),
    'fc1': ir.Affine(
        weight=np.random.randn(512, 256).astype(np.float32),
        bias=np.zeros(512, dtype=np.float32)
    ),
    'gelu': ir.GELU(features=512),
    'fc2': ir.Affine(
        weight=np.random.randn(256, 512).astype(np.float32),
        bias=np.zeros(256, dtype=np.float32)
    ),
    'dropout': ir.Dropout(features=256, p=0.1),
    
    # Residual skip
    'skip': ir.Skip(skip_type='residual'),
    
    'output': ir.Output(np.array([256]))
}

edges = [
    ('input', 'ln1'),
    ('ln1', 'fc1'),
    ('fc1', 'gelu'),
    ('gelu', 'fc2'),
    ('fc2', 'dropout'),
    ('dropout', 'skip'),
    ('input', 'skip'),  # Residual connection
    ('skip', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Full Pipeline: snnTorch to Hardware

Complete workflow from training to deployment.

```python
import torch
import torch.nn as nn
import snntorch as snn
from talon import ir, bridge, viz

# 1. Define and train model
class PokerSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1156, 256)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(256, 4)
        self.lif2 = snn.Leaky(beta=0.9)
    
    def forward(self, x, mem1=None, mem2=None):
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        return spk2, mem1, mem2

model = PokerSNN()
# ... training code ...

# 2. Export to TALON IR
sample = torch.randn(1, 1156)
graph = bridge.to_ir(model, sample)

# 3. Visualize
viz.visualize(graph, title="PokerDVS SNN")

# 4. Save for hardware
ir.write('poker_snn.t1c', graph)

# 5. Verify round-trip
executor = bridge.ir_to_torch('poker_snn.t1c', return_state=True)
test_input = torch.randn(1, 1156)
output, state = executor(test_input, {})
print(f"Output shape: {output.shape}")  # torch.Size([1, 4])
```

## Ghost Block (SU-YOLO Mid-Ghost)

GhostBasicBlock1 for backbone feature extraction with stride-2 downsampling.

```python
import numpy as np
from talon import ir

C_in, C_out = 32, 64
Cp0 = C_out // 2

gbb1 = ir.GhostBasicBlock1(
    cv0_primary_weight=np.random.randn(Cp0, C_in, 3, 3).astype(np.float32),
    cv0_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
    cvres_primary_weight=np.random.randn(C_out // 4, C_out, 1, 1).astype(np.float32),
    cvres_cheap_weight=np.random.randn(C_out // 4, 1, 3, 3).astype(np.float32),
    cv2_primary_weight=np.random.randn(Cp0, Cp0, 1, 1).astype(np.float32),
    cv2_cheap_weight=np.random.randn(Cp0, 1, 3, 3).astype(np.float32),
    stride=2,
)

nodes = {
    'input': ir.Input(np.array([C_in, 40, 40])),
    'ghost_bb1': gbb1,
    'lif': ir.LIF(
        tau=np.ones(C_out) * 10.0,
        r=np.ones(C_out) * 10.0,
        v_leak=np.zeros(C_out),
        v_threshold=np.ones(C_out)
    ),
    'output': ir.Output(np.array([C_out, 20, 20]))
}
edges = [
    ('input', 'ghost_bb1'),
    ('ghost_bb1', 'lif'),
    ('lif', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
# Input: [32, 40, 40] -> GhostBasicBlock1 -> [64, 20, 20]
```

## Detection Pipeline (SDDetect → DFLDecode → Dist2BBox → NMS)

Full SU-YOLO detection post-processing pipeline.

```python
import numpy as np
from talon import ir

# Per-scale detection head
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

# Post-processing chain
dfl = ir.DFLDecode(reg_max=16)
d2b = ir.Dist2BBox(stride=8)
nms = ir.NMS(score_threshold=0.25, iou_threshold=0.45, max_detections=300)

nodes = {
    'input': ir.Input(np.array([64, 80, 80])),
    'detect': detect,
    'dfl': dfl,
    'd2b': d2b,
    'nms': nms,
    'output': ir.Output(np.array([300, 85]))
}
edges = [
    ('input', 'detect'),
    ('detect', 'dfl'),
    ('dfl', 'd2b'),
    ('d2b', 'nms'),
    ('nms', 'output')
]

graph = ir.Graph(nodes=nodes, edges=edges)
```

## Hardware Partitioning and Placement

Partition a graph across neuromorphic cores and optimize placement.

```python
import numpy as np
from talon import ir
from talon.graph import HardwareSpec, partition, route, allocate, place

# Build a simple SNN
nodes = {
    'input': ir.Input(np.array([784])),
    'fc1': ir.Affine(
        weight=np.random.randn(128, 784).astype(np.float32),
        bias=np.zeros(128, dtype=np.float32)
    ),
    'lif1': ir.LIF(
        tau=np.ones(128) * 10.0, r=np.ones(128) * 10.0,
        v_leak=np.zeros(128), v_threshold=np.ones(128)
    ),
    'fc2': ir.Affine(
        weight=np.random.randn(10, 128).astype(np.float32),
        bias=np.zeros(10, dtype=np.float32)
    ),
    'lif2': ir.LIF(
        tau=np.ones(10) * 10.0, r=np.ones(10) * 10.0,
        v_leak=np.zeros(10), v_threshold=np.ones(10)
    ),
    'output': ir.Output(np.array([10]))
}
edges = [
    ('input', 'fc1'), ('fc1', 'lif1'),
    ('lif1', 'fc2'), ('fc2', 'lif2'),
    ('lif2', 'output')
]
graph = ir.Graph(nodes=nodes, edges=edges)

# Use a device preset
hw = HardwareSpec.zynq_us_plus()

# Full hardware mapping pipeline
graph = partition(graph, hw)
graph = route(graph, hw)
resources = allocate(graph, hw)
placement = place(graph, hw)

print(f"Cores used: {graph.partition_metadata['num_cores_used']}")
print(f"Fits hardware: {resources.fits_hardware}")
print(f"Total hop distance: {placement.total_hops}")
print(f"Placement improvement: {placement.improvement:.1f}%")
```

## Backend Compilation and Simulation

Compile a graph and run CPU simulation with energy profiling.

```python
import numpy as np
from talon import ir
from talon.backend import get_backend

# Build graph (reuse from above)
graph = ir.read('model.t1c')

# CPU backend
cpu = get_backend("cpu")

# Validate
validation = cpu.validate(graph)
print(f"Valid: {validation.is_valid}")

# Compile to hardware descriptor
descriptor = cpu.compile(graph)

# Simulate for 10 timesteps
result = cpu.simulate(graph, n_steps=10)

# Profile with device-specific energy preset
profile = cpu.profile(graph, n_steps=10, energy_preset="zynq_us_plus")
print(profile.summary())
```

## Event I/O and Neural Encoding

Process event camera data with talon.io.

```python
from talon.io import streaming, encoding, formats

# Generate random events for testing
events = streaming.generate_random_events(10_000, sensor_size=(32, 32))

# Neural encoding: convert analog data to spike trains
import numpy as np
analog_data = np.random.randn(100).astype(np.float32)
spikes = encoding.rate_encode(analog_data, n_steps=50)

# Convert events to frame tensor for visualization
frames = formats.events_to_frames(events, sensor_size=(32, 32), n_frames=10)
print(f"Frame tensor shape: {frames.shape}")  # (10, 2, 32, 32)

# Throughput benchmarking
from talon.io.throughput import benchmark_throughput
result = benchmark_throughput(n_events=1_000_000, sensor_size=(640, 480))
print(result.summary())
```

