---
sidebar_position: 11
---

# Tutorial: TALON IR Basics

This tutorial introduces TALON IR (Type-1 Compute Intermediate Representation), a hardware-agnostic format for representing spiking and hybrid neural networks as directed graphs.

## What You'll Learn

1. What TALON IR is and why it exists
2. Creating nodes (primitives) for your network
3. Building graphs by connecting nodes
4. Serializing graphs to disk (HDF5 format)
5. Graph validation and inspection
6. Building convolutional networks
7. Skip connections (residual blocks)
8. Ghost and detection primitives

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Introduction to TALON IR

TALON IR is designed to:

- **Represent** spiking neural networks as computational graphs
- **Bridge** PyTorch/snnTorch models to hardware deployment
- **Support** both SNN and hybrid ANN-SNN architectures
- **Enable** hardware-agnostic model exchange

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Graph** | Container for nodes and edges forming a DAG (or cyclic graph) |
| **Node** | A computational primitive (LIF, Conv2d, SGhostConv, etc.) |
| **Edge** | Connection between nodes (data flow) |
| **Primitive** | The type of operation a node performs |

---

## 2. Setup and Available Primitives

```python
import numpy as np
from talon import ir

print(f"TALON IR version: {ir.__version__}")
print(f"\nAvailable primitives: {len(ir.list_primitives())}")
for p in sorted(ir.list_primitives()):
    print(f"  - {p}")
```

**Output:**

```
TALON IR version: 0.0.1

Available primitives: 36
  - Affine
  - AvgPool2d
  - BatchNorm1d
  - BatchNorm2d
  - ChannelSplit
  - Concat
  - Conv1d
  - Conv2d
  - DFLDecode
  - Dist2BBox
  - Dropout
  - ELU
  - Flatten
  - GELU
  - GhostBasicBlock1
  - GhostBasicBlock2
  - HybridRegion
  - IF
  - LIF
  - LayerNorm
  - MaxPool2d
  - NMS
  - PReLU
  - ReLU
  - SConv
  - SDConv
  - SDDetect
  - SGhostConv
  - SGhostEncoderLite
  - SepConv2d
  - Sigmoid
  - Skip
  - Softmax
  - SpikingAffine
  - Tanh
  - Upsample
```

:::tip Primitive Categories
The 36 primitives span six categories: **Core** (Affine, Conv2d, Conv1d, SConv, SDConv, SepConv2d), **Neuron** (LIF, IF), **Activation** (ReLU, Sigmoid, Tanh, Softmax, GELU, ELU, PReLU), **Normalization** (BatchNorm1d/2d, LayerNorm, Dropout), **Pooling/Reshape** (MaxPool2d, AvgPool2d, Flatten, Upsample), **Ghost/Detection** (SGhostConv, SGhostEncoderLite, GhostBasicBlock1/2, ChannelSplit, Concat, SDDetect, DFLDecode, Dist2BBox, NMS), and **Structural** (Skip, HybridRegion, SpikingAffine).
:::

---

## 3. Creating Nodes (Primitives)

Every neural network is made of computational operations. In TALON IR, these are called **primitives**.

### Input and Output Nodes

```python
input_node = ir.Input(np.array([784]))
print(f"Input shape: {input_node.input_type}")
```

**Output:**

```
Input shape: {'input': array([784])}
```

### Affine (Fully-Connected) Layer

```python
fc1 = ir.Affine(
    weight=np.random.randn(128, 784).astype(np.float32) * 0.01,
    bias=np.zeros(128, dtype=np.float32),
)
print(f"Affine weight shape: {fc1.weight.shape}")
print(f"Affine bias shape: {fc1.bias.shape}")
```

**Output:**

```
Affine weight shape: (128, 784)
Affine bias shape: (128,)
```

### LIF (Leaky Integrate-and-Fire) Neuron

The LIF neuron is the core of spiking neural networks. Parameters control the membrane dynamics:

- **tau**: membrane time constant (higher = slower decay)
- **r**: membrane resistance
- **v_leak**: resting membrane potential
- **v_threshold**: spike threshold

```python
lif1 = ir.LIF(
    tau=np.ones(128, dtype=np.float32) * 10.0,
    r=np.ones(128, dtype=np.float32),
    v_leak=np.zeros(128, dtype=np.float32),
    v_threshold=np.ones(128, dtype=np.float32),
)
print(f"LIF neurons: {len(lif1.tau)}")
```

**Output:**

```
LIF neurons: 128
```

### IF (Integrate-and-Fire) Neuron

The IF neuron is a simpler variant with no leak — only a threshold parameter:

```python
if_neuron = ir.IF(v_threshold=np.ones(32, dtype=np.float32))
print(f"IF neurons: {len(if_neuron.v_threshold)}")
```

**Output:**

```
IF neurons: 32
```

### Output Layer

```python
fc2 = ir.Affine(
    weight=np.random.randn(10, 128).astype(np.float32) * 0.01,
    bias=np.zeros(10, dtype=np.float32),
)
lif2 = ir.LIF(
    tau=np.ones(10, dtype=np.float32) * 10.0,
    r=np.ones(10, dtype=np.float32),
    v_leak=np.zeros(10, dtype=np.float32),
    v_threshold=np.ones(10, dtype=np.float32),
)
output_node = ir.Output(np.array([10]))
print(f"Output shape: {output_node.output_type}")
```

**Output:**

```
Output shape: {'output': array([10])}
```

---

## 4. Building a Graph

A graph is created by defining a dictionary of named nodes and a list of edges (source, destination) tuples.

```
Input -> Affine -> LIF -> Affine -> LIF -> Output
```

```python
nodes = {
    "input": input_node,
    "fc1": fc1,
    "lif1": lif1,
    "fc2": fc2,
    "lif2": lif2,
    "output": output_node,
}

edges = [
    ("input", "fc1"),
    ("fc1", "lif1"),
    ("lif1", "fc2"),
    ("fc2", "lif2"),
    ("lif2", "output"),
]

graph = ir.Graph(nodes=nodes, edges=edges)

print(f"Graph created:")
print(f"  Nodes: {len(graph.nodes)}")
print(f"  Edges: {len(graph.edges)}")
print(f"  Is DAG: {graph.is_dag}")
```

**Output:**

```
Graph created:
  Nodes: 6
  Edges: 5
  Is DAG: True
```

### Inspecting the Graph

```python
print("Nodes:")
for name, node in graph.nodes.items():
    print(f"  {name}: {type(node).__name__}")

print("\nEdges:")
for src, dst in graph.edges:
    print(f"  {src} -> {dst}")
```

**Output:**

```
Nodes:
  input: Input
  fc1: Affine
  lif1: LIF
  fc2: Affine
  lif2: LIF
  output: Output

Edges:
  input -> fc1
  fc1 -> lif1
  lif1 -> fc2
  fc2 -> lif2
  lif2 -> output
```

---

## 5. Serialization (Save/Load)

TALON IR uses HDF5 format for serialization. This preserves graph structure, all parameters (weights, biases, neuron state), and metadata.

```python
output_path = "models/simple_snn.t1c"
ir.write(output_path, graph)
print(f"Saved to: {output_path}")
```

**Output:**

```
Saved to: models/simple_snn.t1c
```

```python
loaded_graph = ir.read(output_path)

print(f"Loaded graph:")
print(f"  Nodes: {len(loaded_graph.nodes)}")
print(f"  Edges: {len(loaded_graph.edges)}")

original_weights = fc1.weight
loaded_weights = loaded_graph.nodes["fc1"].weight
weight_diff = np.abs(original_weights - loaded_weights).max()
print(f"  Weight preservation: max diff = {weight_diff:.2e}")
```

**Output:**

```
Loaded graph:
  Nodes: 6
  Edges: 5
  Weight preservation: max diff = 0.00e+00
```

---

## 6. Graph Validation

TALON IR supports both DAGs (directed acyclic graphs) and cyclic graphs for recurrent architectures.

```python
print(f"Is DAG: {graph.is_dag}")
print(f"Node count: {len(graph.nodes)}")
print(f"Edge count: {len(graph.edges)}")
```

**Output:**

```
Is DAG: True
Node count: 6
Edge count: 5
```

Adding a feedback edge creates a cyclic graph:

```python
cyclic_edges = edges + [("lif2", "fc1")]
cyclic_graph = ir.Graph(nodes=nodes, edges=cyclic_edges)
print(f"Graph created (allows cycles)")
print(f"  Is DAG: {cyclic_graph.is_dag}")
```

**Output:**

```
Graph created (allows cycles)
  Is DAG: False
```

---

## 7. Building a Convolutional Network

For image processing, combine `Conv2d`, `MaxPool2d`, `Flatten`, and dense layers:

```
Input (1x28x28) -> Conv2d -> LIF -> MaxPool -> Conv2d -> LIF -> MaxPool -> Flatten -> FC -> LIF -> Output
```

```python
def create_conv_block(name, in_channels, out_channels, kernel_size=3):
    """Create a Conv2d + LIF block."""
    conv = ir.Conv2d(
        weight=np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * 0.1,
        bias=np.zeros(out_channels, dtype=np.float32),
        stride=(1, 1),
        padding=(kernel_size // 2, kernel_size // 2),
    )
    lif = ir.LIF(
        tau=np.ones(out_channels, dtype=np.float32) * 10.0,
        r=np.ones(out_channels, dtype=np.float32),
        v_leak=np.zeros(out_channels, dtype=np.float32),
        v_threshold=np.ones(out_channels, dtype=np.float32),
    )
    return {f"{name}_conv": conv, f"{name}_lif": lif}

cnn_nodes = {}
cnn_nodes["input"] = ir.Input(np.array([1, 28, 28]))

cnn_nodes.update(create_conv_block("block1", 1, 8))
cnn_nodes["pool1"] = ir.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

cnn_nodes.update(create_conv_block("block2", 8, 16))
cnn_nodes["pool2"] = ir.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

cnn_nodes["flatten"] = ir.Flatten(start_dim=0)
cnn_nodes["fc"] = ir.Affine(
    weight=np.random.randn(10, 16 * 7 * 7).astype(np.float32) * 0.01,
    bias=np.zeros(10, dtype=np.float32),
)
cnn_nodes["fc_lif"] = ir.LIF(
    tau=np.ones(10, dtype=np.float32) * 10.0,
    r=np.ones(10, dtype=np.float32),
    v_leak=np.zeros(10, dtype=np.float32),
    v_threshold=np.ones(10, dtype=np.float32),
)
cnn_nodes["output"] = ir.Output(np.array([10]))

print(f"CNN nodes: {len(cnn_nodes)}")
for name, node in cnn_nodes.items():
    print(f"  {name}: {type(node).__name__}")
```

**Output:**

```
CNN nodes: 11
  input: Input
  block1_conv: Conv2d
  block1_lif: LIF
  pool1: MaxPool2d
  block2_conv: Conv2d
  block2_lif: LIF
  pool2: MaxPool2d
  flatten: Flatten
  fc: Affine
  fc_lif: LIF
  output: Output
```

```python
cnn_edges = [
    ("input", "block1_conv"),
    ("block1_conv", "block1_lif"),
    ("block1_lif", "pool1"),
    ("pool1", "block2_conv"),
    ("block2_conv", "block2_lif"),
    ("block2_lif", "pool2"),
    ("pool2", "flatten"),
    ("flatten", "fc"),
    ("fc", "fc_lif"),
    ("fc_lif", "output"),
]

cnn_graph = ir.Graph(nodes=cnn_nodes, edges=cnn_edges)
print(f"CNN Graph:")
print(f"  Nodes: {len(cnn_graph.nodes)}")
print(f"  Edges: {len(cnn_graph.edges)}")
print(f"  Is DAG: {cnn_graph.is_dag}")
```

**Output:**

```
CNN Graph:
  Nodes: 11
  Edges: 10
  Is DAG: True
```

---

## 8. Skip Connections (Residual Blocks)

Skip connections allow gradients to flow through the network more easily.

```
      ┌─────────────────────────┐
      │                         │
      │                         ▼
Input -> Conv -> LIF -> Conv -> Skip -> LIF -> Output
```

```python
channels = 32

res_nodes = {
    "input": ir.Input(np.array([channels, 14, 14])),
    "conv1": ir.Conv2d(
        weight=np.random.randn(channels, channels, 3, 3).astype(np.float32) * 0.1,
        bias=np.zeros(channels, dtype=np.float32),
        stride=(1, 1), padding=(1, 1),
    ),
    "lif1": ir.LIF(
        tau=np.ones(channels, dtype=np.float32) * 10.0,
        r=np.ones(channels, dtype=np.float32),
        v_leak=np.zeros(channels, dtype=np.float32),
        v_threshold=np.ones(channels, dtype=np.float32),
    ),
    "conv2": ir.Conv2d(
        weight=np.random.randn(channels, channels, 3, 3).astype(np.float32) * 0.1,
        bias=np.zeros(channels, dtype=np.float32),
        stride=(1, 1), padding=(1, 1),
    ),
    "skip": ir.Skip(
        input_type={"input": np.array([channels, 14, 14])},
        skip_type="residual",
    ),
    "lif2": ir.LIF(
        tau=np.ones(channels, dtype=np.float32) * 10.0,
        r=np.ones(channels, dtype=np.float32),
        v_leak=np.zeros(channels, dtype=np.float32),
        v_threshold=np.ones(channels, dtype=np.float32),
    ),
    "output": ir.Output(np.array([channels, 14, 14])),
}

res_edges = [
    ("input", "conv1"),
    ("conv1", "lif1"),
    ("lif1", "conv2"),
    ("conv2", "skip"),
    ("input", "skip"),
    ("skip", "lif2"),
    ("lif2", "output"),
]

res_graph = ir.Graph(nodes=res_nodes, edges=res_edges)
print(f"Residual Block:")
print(f"  Nodes: {len(res_graph.nodes)}")
print(f"  Edges: {len(res_graph.edges)}")
print(f"  Is DAG: {res_graph.is_dag}")
```

**Output:**

```
Residual Block:
  Nodes: 7
  Edges: 7
  Is DAG: True
```

---

## 9. Ghost and Detection Primitives

TALON IR includes Ghost convolution and object detection primitives for efficient spiking architectures like SU-YOLO.

### SGhostConv

A Spiking Ghost Convolution performs a primary 1x1 convolution followed by a cheap depthwise convolution to generate "ghost" features:

```python
ghost_conv = ir.SGhostConv(
    primary_weight=np.random.randn(16, 8, 1, 1).astype(np.float32) * 0.1,
    primary_bias=np.zeros(16, dtype=np.float32),
    cheap_weight=np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.1,
    cheap_bias=np.zeros(16, dtype=np.float32),
)
print(f"SGhostConv primary weight: {ghost_conv.primary_weight.shape}")
print(f"SGhostConv cheap weight: {ghost_conv.cheap_weight.shape}")
```

**Output:**

```
SGhostConv primary weight: (16, 8, 1, 1)
SGhostConv cheap weight: (16, 1, 3, 3)
```

### NMS (Non-Maximum Suppression)

Post-processing for object detection that filters overlapping bounding boxes:

```python
nms = ir.NMS(iou_threshold=0.45, score_threshold=0.25, max_detections=100)
print(f"NMS:")
print(f"  IoU threshold: {nms.iou_threshold}")
print(f"  Score threshold: {nms.score_threshold}")
print(f"  Max detections: {nms.max_detections}")
```

**Output:**

```
NMS:
  IoU threshold: 0.45
  Score threshold: 0.25
  Max detections: 100
```

### Spiking Convolutions

`SConv` (standard 2D) and `SDConv` (depthwise 2D) are spiking-aware convolution variants:

```python
sconv = ir.SConv(
    weight=np.random.randn(16, 8, 3, 3).astype(np.float32) * 0.1,
    bias=np.zeros(16, dtype=np.float32),
    stride=(1, 1), padding=(1, 1),
)
print(f"SConv weight: {sconv.weight.shape}")

# SDConv is spiking depthwise conv: groups == in_channels == out_channels.
# For an 8-channel depthwise kernel, weight shape is (8, 1, kH, kW) with groups=8.
sdconv = ir.SDConv(
    weight=np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.1,
    bias=np.zeros(8, dtype=np.float32),
    stride=(1, 1), padding=(1, 1),
    groups=8,
)
print(f"SDConv weight: {sdconv.weight.shape}, groups={sdconv.groups}")
```

**Output:**

```
SConv weight: (16, 8, 3, 3)
SDConv weight: (8, 1, 3, 3), groups=8
```

---

## Summary

You've learned:

1. **36 Primitives**: Input, Output, Affine, LIF, IF, Conv2d, Conv1d, SConv, SDConv, MaxPool2d, SGhostConv, NMS, and more
2. **Graph construction**: nodes dict + edges list
3. **Serialization**: `ir.write()` and `ir.read()` (HDF5 format, lossless)
4. **Validation**: `graph.is_dag`, cyclic graph support
5. **Architectures**: Dense, CNN, Residual, Ghost/Detection

## What's Next

- [Tutorial: talonviz](./tutorial_talonviz) — Visualize graphs and spike events
- [Tutorial: TALON SDK](./tutorial_talon) — Analyze, profile, and deploy graphs
- [Tutorial: Bridge](./tutorial_bridge) — Export/import PyTorch models
- [Primitives Reference](../primitives) — All 36 TALON IR primitives
