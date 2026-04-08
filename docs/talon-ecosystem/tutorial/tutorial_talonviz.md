---
sidebar_position: 12
---

# Tutorial: talonviz — Visualization

talonviz provides interactive visualization tools for TALON IR graphs and neuromorphic event data.

## What You'll Learn

1. Rendering interactive graph visualizations (HTML)
2. Processing neuromorphic events into frames, rasters, and grids
3. Exporting animated event visualizations
4. Detecting architectural patterns (skip connections, SPP)

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Setup

```python
import os
import numpy as np
from talon import ir, viz

os.makedirs("viz", exist_ok=True)

print(f"talonviz version: {viz.__version__}")
print(f"Tonic available: {viz.TONIC_AVAILABLE}")
print(f"PIL available: {viz.PIL_AVAILABLE}")
```

**Output:**

```
talonviz version: 0.0.1
Tonic available: False
PIL available: True
```

---

## 2. Graph Visualization

talonviz renders TALON IR graphs as interactive HTML with zoomable/pannable views, node details on hover, and architecture pattern highlighting.

### Create a Graph

```python
nodes = {
    "input": ir.Input(np.array([784])),
    "fc1": ir.Affine(
        weight=np.random.randn(256, 784).astype(np.float32) * 0.01,
        bias=np.zeros(256, dtype=np.float32),
    ),
    "lif1": ir.LIF(
        tau=np.ones(256, dtype=np.float32) * 10.0,
        r=np.ones(256, dtype=np.float32),
        v_leak=np.zeros(256, dtype=np.float32),
        v_threshold=np.ones(256, dtype=np.float32),
    ),
    "fc2": ir.Affine(
        weight=np.random.randn(10, 256).astype(np.float32) * 0.01,
        bias=np.zeros(10, dtype=np.float32),
    ),
    "lif2": ir.LIF(
        tau=np.ones(10, dtype=np.float32) * 10.0,
        r=np.ones(10, dtype=np.float32),
        v_leak=np.zeros(10, dtype=np.float32),
        v_threshold=np.ones(10, dtype=np.float32),
    ),
    "output": ir.Output(np.array([10])),
}

edges = [
    ("input", "fc1"),
    ("fc1", "lif1"),
    ("lif1", "fc2"),
    ("fc2", "lif2"),
    ("lif2", "output"),
]

graph = ir.Graph(nodes=nodes, edges=edges)
print(f"Created graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
```

**Output:**

```
Created graph: 6 nodes, 5 edges
```

### Export to Interactive HTML

```python
viz.export_html(graph, "viz/snn_visualization.html", title="Simple SNN")
print("Exported to viz/snn_visualization.html")
```

**Output:**

```
Exported to viz/snn_visualization.html
```

:::tip Localhost Serving
Use `viz.serve(graph)` to launch an interactive visualization on a local HTTP server, similar to Plotly's approach.
:::

### Serialize Graph Data

```python
data = viz.graph_to_dict(graph)

print(f"Serialized data keys: {list(data.keys())}")
print(f"\nSummary:")
print(f"  Node count: {data['summary']['node_count']}")
print(f"  Edge count: {data['summary']['edge_count']}")
print(f"  Total params: {data['summary']['total_params']:,}")
```

**Output:**

```
Serialized data keys: ['version', 'nodes', 'edges', 'metadata', 'summary', 'groups', 'node_to_group']

Summary:
  Node count: 6
  Edge count: 5
  Total params: 204,594
```

---

## 3. Event/Spike Visualization

talonviz processes neuromorphic event data into three representations:

| Representation | Shape | Description |
|----------------|-------|-------------|
| **Frames** | `(n_frames, 2, H, W)` | Accumulated event counts per time bin (ON/OFF channels) |
| **Rasters** | `(time_bins, n_neurons)` | Binary spike matrix per neuron per time bin |
| **Grids** | `(n_bins, H, W, 3)` | RGB spatial grids for larger sensors |

### Generate Synthetic Events

```python
n_events = 5000
sensor_size = (28, 28)

events = np.zeros(n_events, dtype=[
    ('x', np.int32),
    ('y', np.int32),
    ('t', np.float64),
    ('p', np.int32),
])

events['x'] = np.random.randint(0, sensor_size[0], n_events)
events['y'] = np.random.randint(0, sensor_size[1], n_events)
events['t'] = np.sort(np.random.uniform(0, 100000, n_events))
events['p'] = np.random.choice([-1, 1], n_events)

print(f"Generated {n_events} events")
print(f"Time range: {events['t'].min():.0f} - {events['t'].max():.0f} μs")
print(f"Sensor size: {sensor_size}")
```

**Output:**

```
Generated 5000 events
Time range: 44 - 99985 μs
Sensor size: (28, 28)
```

### Convert to Frames

```python
frames = viz.events_to_frames(
    events,
    sensor_size=sensor_size,
    n_frames=25,
)

print(f"Frame shape: {frames.shape}")
print(f"  (n_frames, channels, height, width)")
print(f"  Channels: 2 (ON/OFF polarity)")
```

**Output:**

```
Frame shape: (25, 2, 28, 28)
  (n_frames, channels, height, width)
  Channels: 2 (ON/OFF polarity)
```

### Convert to Raster

```python
n_neurons = sensor_size[0] * sensor_size[1]

raster = viz.events_to_raster(
    events,
    n_neurons=n_neurons,
    time_bins=100,
)

print(f"Raster shape: {raster.shape}")
print(f"  (time_bins, n_neurons)")
print(f"  n_neurons = {sensor_size[0]} * {sensor_size[1]} = {n_neurons}")
```

**Output:**

```
Raster shape: (100, 784)
  (time_bins, n_neurons)
  n_neurons = 28 * 28 = 784
```

### Convert to Grid

```python
grid = viz.events_to_grid(
    events,
    sensor_size=sensor_size,
    n_bins=5,
)

print(f"Grid shape: {grid.shape}")
print(f"  (n_bins, height, width, channels)")
```

**Output:**

```
Grid shape: (5, 28, 28, 3)
  (n_bins, height, width, channels)
```

### Export Event Visualization

```python
viz.export_events_html(
    events,
    "viz/tutorial_events.html",
    sensor_size=sensor_size,
    title="Tutorial: Synthetic Events",
    n_frames=30,
)

print("Exported to viz/tutorial_events.html")
print("Features:")
print("  - Animated frame playback")
print("  - Spike raster plot")
print("  - Event statistics")
```

**Output:**

```
Exported to viz/tutorial_events.html
Features:
  - Animated frame playback
  - Spike raster plot
  - Event statistics
```

---

## 4. Pattern Detection

talonviz can detect common architectural patterns in graphs:

| Pattern | Description |
|---------|-------------|
| **SkipConnection** | ResNet-style residual blocks |
| **SPP** | Spatial Pyramid Pooling |
| **SPPF** | Fast SPP (YOLOv5+) |
| **RepConv** | Reparameterizable convolutions |

### Create a Graph with Skip Connections

```python
skip_nodes = {
    "input": ir.Input(np.array([64, 14, 14])),
    "conv1": ir.Conv2d(
        weight=np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01,
        bias=np.zeros(64, dtype=np.float32),
        stride=(1, 1), padding=(1, 1),
    ),
    "lif1": ir.LIF(
        tau=np.ones(64, dtype=np.float32) * 10.0,
        r=np.ones(64, dtype=np.float32),
        v_leak=np.zeros(64, dtype=np.float32),
        v_threshold=np.ones(64, dtype=np.float32),
    ),
    "conv2": ir.Conv2d(
        weight=np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01,
        bias=np.zeros(64, dtype=np.float32),
        stride=(1, 1), padding=(1, 1),
    ),
    "skip": ir.Skip(
        input_type={"input": np.array([64, 14, 14])},
        skip_type="residual",
    ),
    "lif2": ir.LIF(
        tau=np.ones(64, dtype=np.float32) * 10.0,
        r=np.ones(64, dtype=np.float32),
        v_leak=np.zeros(64, dtype=np.float32),
        v_threshold=np.ones(64, dtype=np.float32),
    ),
    "output": ir.Output(np.array([64, 14, 14])),
}

skip_edges = [
    ("input", "conv1"),
    ("conv1", "lif1"),
    ("lif1", "conv2"),
    ("conv2", "skip"),
    ("input", "skip"),
    ("skip", "lif2"),
    ("lif2", "output"),
]

skip_graph = ir.Graph(nodes=skip_nodes, edges=skip_edges)
print(f"Created ResNet-style graph: {len(skip_graph.nodes)} nodes")
```

**Output:**

```
Created ResNet-style graph: 7 nodes
```

### Detect Patterns

```python
patterns = viz.detect_all_patterns(skip_graph)

print(f"Patterns detected: {len(patterns)}")
for pattern in patterns:
    print(f"  - {pattern.pattern_type}: {pattern.group_id}")
    print(f"    Nodes: {pattern.nodes}")
```

**Output:**

```
Patterns detected: 1
  - SkipConnection: skip_input_skip
    Nodes: ['skip', 'conv1', 'lif1', 'conv2']
```

### Export with Pattern Highlighting

```python
viz.export_html(
    skip_graph,
    "viz/resnet_block.html",
    title="ResNet Block with Skip Connection",
)
print("Exported to viz/resnet_block.html")
print("The skip connection pattern will be highlighted in the visualization.")
```

**Output:**

```
Exported to viz/resnet_block.html
The skip connection pattern will be highlighted in the visualization.
```

---

## 5. Performance Considerations

| Implementation | Speed | Requirements |
|----------------|-------|-------------|
| NumPy (default) | ~23M events/sec | None |
| Tonic/Numba | ~30M events/sec | `pip install tonic` |

```python
print(f"Tonic available: {viz.TONIC_AVAILABLE}")

if viz.TONIC_AVAILABLE:
    print("Using Tonic/Numba implementation (faster)")
else:
    print("Using NumPy implementation")
    print("Install tonic for ~30% speedup: pip install tonic")
```

**Output:**

```
Tonic available: False
Using NumPy implementation
Install tonic for ~30% speedup: pip install tonic
```

---

## Summary

| Function | Purpose |
|----------|---------|
| `viz.export_html()` | Interactive graph visualization |
| `viz.graph_to_dict()` | Serialize graph for visualization |
| `viz.events_to_frames()` | Convert events to frame arrays |
| `viz.events_to_raster()` | Convert events to spike rasters |
| `viz.events_to_grid()` | Convert events to spatial grids |
| `viz.export_events_html()` | Interactive spike visualization |
| `viz.detect_all_patterns()` | Find architectural patterns |
| `viz.serve()` | Localhost HTTP visualization server |

## What's Next

- [Tutorial: TALON SDK](./tutorial_talon) — Analyze, profile, and deploy graphs
- [Tutorial: Bridge](./tutorial_bridge) — Export/import PyTorch models
- [Visualization Reference](../visualization) — Full visualization API
