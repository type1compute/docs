---
sidebar_position: 8
---

# Visualization Guide

Generate interactive visualizations of T1C-IR graphs and event-based data using T1C-Viz (t1c.viz).

## Interactive Tutorial

For a hands-on introduction with step-by-step explanations, see the [T1C-Viz Tutorial Notebook](./example_notebooks/tutorial_t1cviz.ipynb).

## Quick Start

### Graph Visualization

```python
from t1c import ir, viz

# Load graph
graph = ir.read('model.t1c')

# Display interactively
viz.visualize(graph)
```

### Spike/Event Visualization (Tonic Integration)

```python
import tonic
from t1c import viz

# Load event data from Tonic dataset
dataset = tonic.datasets.NMNIST("./data", train=True)
events, label = dataset[0]

# Visualize events interactively
viz.plot_events(events, title=f"NMNIST Sample (label={label})")

# Export to standalone HTML
viz.export_events_html(events, "events.html")
```

## Display Methods

### Browser Display

Outside Jupyter, opens in default browser:

```python
viz.visualize(graph, title="My SNN Model")
# Output: Visualization written to: C:\...\T1C-Viz\My_SNN_Model_xxx.html
```

### Jupyter Display

In Jupyter notebooks, displays as embedded iframe:

```python
viz.visualize(graph, title="My SNN", width=960, height=640)
```

### Export to File

Save as standalone HTML:

```python
path = viz.export_html(graph, "model.html", title="Production Model")
print(f"Saved to: {path}")
```

The HTML is self-contained with embedded D3.js and Dagre - no server needed.

## Visualization Features

### Interactive Controls

- **Pan**: Click and drag
- **Zoom**: Scroll wheel
- **Fit**: Click "Fit" button to fit graph to viewport
- **Reset**: Click "Reset" to restore default zoom

### Node Information

Click any node to see:

- **Shapes**: Input and output tensor shapes
- **Parameters**: Weight matrices with dtype and size
- **Configuration**: Stride, padding, spike_mode, etc.
- **Metadata**: Any custom attributes

### Graph Summary

The side panel shows:

- Total node count
- Total edge count
- Total parameter count
- Input/output shapes

## Node Colors

Each primitive type has a distinct color:

| Primitive | Color | Hex |
|-----------|-------|-----|
| Input | Light green | #c8e6c9 |
| Output | Light red | #ffcdd2 |
| Affine | Light blue | #bbdefb |
| SpikingAffine | Light purple | #e1bee7 |
| Conv2d | Light orange | #ffe0b2 |
| Flatten | Light gray | #e0e0e0 |
| LIF | Light cyan | #b2ebf2 |
| MaxPool2d | Light brown | #d7ccc8 |
| SepConv2d | Light pink | #f8bbd9 |
| Skip | Light yellow | #fff9c4 |

## Low-Level API

For custom rendering:

```python
from t1c import viz
from t1c.viz import graph_to_dict, render_html

# Convert graph to JSON-serializable dict
data = graph_to_dict(graph)
print(data.keys())  # dict_keys(['nodes', 'edges', 'node_count', ...])

# Render to HTML
html = render_html(data, title="Custom Title")

# Write manually
with open('custom.html', 'w') as f:
    f.write(html)
```

### Data Structure

`graph_to_dict()` returns:

```python
{
    'nodes': [
        {
            'id': 'fc1',
            'type': 'Affine',
            'input_shape': [784],
            'output_shape': [128],
            'params': {
                'weight': {'shape': [128, 784], 'dtype': 'float32', 'size': 100352},
                'bias': {'shape': [128], 'dtype': 'float32', 'size': 128}
            },
            'param_count': 100480,
            'config': {}
        },
        # ... more nodes
    ],
    'edges': [
        {'source': 'input', 'target': 'fc1'},
        # ... more edges
    ],
    'node_count': 6,
    'edge_count': 5,
    'total_params': 102530,
    'inputs': [{'id': 'input', 'shape': [784]}],
    'outputs': [{'id': 'output', 'shape': [10]}]
}
```

## Managing Visualizations

### List Exported Files

```python
files = viz.list_graphs()
print(files)  # ['model_a_xxx.html', 'model_b_xxx.html']
```

### Temporary Directory

Visualizations are stored in:

```python
print(viz.TEMP_GRAPHS_DIR)
# Windows: C:\Users\<user>\AppData\Local\Temp\t1cviz
# Linux: /tmp/t1cviz
```

## Multiple Graphs

Visualize multiple architectures:

```python
graphs = [
    (graph1, 'simple_snn', 'Simple SNN'),
    (graph2, 'conv_snn', 'Convolutional SNN'),
    (graph3, 'deep_snn', 'Deep SNN'),
]

for graph, filename, title in graphs:
    path = viz.export_html(graph, f"{filename}.html", title=title)
    print(f"{title}: {path}")
```

## Embedding in Documentation

The exported HTML can be embedded in documentation:

```html
<iframe src="model.html" width="100%" height="600px"></iframe>
```

Or screenshot for static docs:

1. Export HTML
2. Open in browser
3. Take screenshot
4. Include in markdown

## Customization

### CSS Styling

The visualization uses embedded CSS. To customize, modify the rendered HTML:

```python
html = render_html(graph_to_dict(graph), "Custom")

# Replace colors
html = html.replace('#bbdefb', '#your-color')

with open('custom.html', 'w') as f:
    f.write(html)
```

### Layout Direction

Dagre uses top-to-bottom layout by default. This is configured in the JavaScript:

```javascript
.setGraph({ rankdir: 'TB', ranksep: 50, nodesep: 30 })
```

Options: `'TB'` (top-bottom), `'LR'` (left-right), `'BT'`, `'RL'`

## Hierarchical Visualization (Pan In/Out)

T1C-Viz automatically detects architectural patterns and groups them for cleaner visualization.

### Supported Patterns

| Pattern | Description | Collapse Shape |
|---------|-------------|----------------|
| RepConv | Multi-branch reparameterizable conv (3×3, 1×1, identity) | Hexagon |
| SPP | Spatial Pyramid Pooling (parallel MaxPool2d with different kernels) | Octagon |
| SPPF | Spatial Pyramid Pooling Fast (sequential MaxPool2d chain) | Octagon |
| SkipConnection | Residual connection (bypass + processing path) | Diamond |

#### RepConv (SpikeYOLO)

Multi-branch convolution with parallel 3×3, 1×1, and identity paths that merge via residual addition.

```
input ─┬─> Conv2d (3×3) ─┬─> Skip (residual) ─> LIF
       ├─> Conv2d (1×1) ─┤
       └─> Skip (identity)┘
```

#### SPP (Spatial Pyramid Pooling)

Parallel MaxPool2d with different kernel sizes for multi-scale feature extraction.

```
input ─┬─> Skip (passthrough) ─┬─> Skip (concatenate)
       ├─> MaxPool2d (5×5) ────┤
       ├─> MaxPool2d (9×9) ────┤
       └─> MaxPool2d (13×13) ──┘
```

#### SPPF (Spatial Pyramid Pooling Fast)

Sequential MaxPool2d chain with same kernel size (more efficient than SPP).

```
input ─┬─> Skip (passthrough) ─────────────────────────┬─> Skip (concatenate)
       └─> MaxPool2d ─┬─> MaxPool2d ─┬─> MaxPool2d ────┤
                      │              │                  │
                      └──────────────┴──────────────────┘
```

#### SkipConnection (ResNet-style)

Residual/skip connection where one path bypasses processing nodes.

```
lif1 ─┬─> Skip (residual) ────────────────────> lif2 (merge)
      │                                           ↑
      └─> fc2 ────────────────────────────────────┘
```

The bypass path uses `Skip(skip_type='residual')` which performs element-wise addition at the merge point. This is the fundamental building block for ResNet-style architectures.

### Group Controls

- **Groups Button**: Click "Groups" in toolbar to expand/collapse all patterns
- **Click Group**: Click a collapsed group node to expand it
- **Tooltip**: Hover over group to see contained nodes

### Pattern Detection API

```python
from t1c.viz.patterns import (
    detect_all_patterns,
    detect_repconv_patterns,
    detect_spp_patterns,
    detect_sppf_patterns,
    detect_skip_connection_patterns,
)

# Detect all patterns
patterns = detect_all_patterns(graph)

# Or detect specific patterns
repconv_patterns = detect_repconv_patterns(graph)
spp_patterns = detect_spp_patterns(graph)
sppf_patterns = detect_sppf_patterns(graph)
skip_patterns = detect_skip_connection_patterns(graph)
```

---

## Spike/Event Visualization

Visualize event-based neuromorphic data compatible with the Tonic library.

### Performance

Event visualization uses optimized processing:

```python
from t1c import viz

# Check available backends
print(f"Tonic available: {viz.TONIC_AVAILABLE}")  # Numba-accelerated
print(f"Pillow available: {viz.PIL_AVAILABLE}")   # Required for images

# Performance numbers:
# - With Tonic: ~30M events/sec (Numba JIT compilation)
# - Without Tonic: ~23M events/sec (pure NumPy with np.add.at)
# Both exceed the 5M events/sec requirement for real-time processing
```

### Tonic Integration

For best performance, install with Tonic support:

```bash
pip install t1c-viz[tonic]
```

When Tonic is available, `events_to_frames` uses Tonic's Numba-accelerated `ToFrame` transform automatically.

### Event Data Format

Events should be a structured numpy array with fields:
- `x`: X coordinate (or channel for audio)
- `y`: Y coordinate (optional for 1D sensors)
- `t` or `ts`: Timestamp
- `p` or `pol`: Polarity (ON/OFF)

### Functions

#### `plot_events(events, sensor_size=None, title="Spike Visualization")`

Opens interactive visualization in browser:

```python
import tonic
from t1c import viz

dataset = tonic.datasets.DVSGesture("./data", train=True)
events, label = dataset[0]

viz.plot_events(events, title=f"Gesture {label}")
```

#### `export_events_html(events, output_path, **kwargs)`

Export to standalone HTML file:

```python
viz.export_events_html(events, "gesture_sample.html")
```

#### `plot_frames(frames, title="Frame Visualization")`

Visualize pre-computed frames (from Tonic `ToFrame` transform):

```python
from tonic.transforms import ToFrame

transform = ToFrame(sensor_size=dataset.sensor_size, time_window=1000)
frames = transform(events)

viz.plot_frames(frames, title="Frame Sequence")
```

### Visualization Components

The spike visualization includes:

1. **Event Animation**: Playable animation of events over time
2. **Event Grid**: Time-binned event frames (like Tonic's `plot_event_grid`)
3. **Spike Raster**: Neuron × time spike raster plot
4. **Statistics**: Event count, duration, rate

### Data Conversion Functions

For custom processing:

```python
from t1c import viz
# Or: from t1c.viz import events_to_frames, events_to_grid, events_to_raster
events_to_frames = viz.events_to_frames
events_to_grid = viz.events_to_grid
events_to_raster = viz.events_to_raster

# Convert events to frame tensor
frames = events_to_frames(events, sensor_size=(34, 34), n_frames=20)
# Shape: (n_frames, 2, H, W) - 2 channels for ON/OFF

# Convert to RGB grid
grid = events_to_grid(events, sensor_size=(34, 34), n_bins=5)
# Shape: (n_bins, H, W, 3)

# Convert to spike raster
raster = events_to_raster(events, n_neurons=1156, time_bins=100)
# Shape: (time_bins, n_neurons)
```

---

## Troubleshooting

### Blank Visualization

If visualization is blank:

1. Check browser console for JavaScript errors
2. Verify graph has nodes: `len(graph.nodes) > 0`
3. Try export_html instead of visualize

### Large Graphs

For graphs with many nodes (>100):

- Increase browser memory
- Use export_html and open in new tab
- Consider filtering nodes before visualization
- Use hierarchical grouping to collapse patterns

### Jupyter Not Displaying

If iframe doesn't show in Jupyter:

```python
# Check if IPython is available
from t1c import viz
print(f"IPython available: {getattr(viz, '_HAS_IPYTHON', 'N/A')}")

# Force browser display
import webbrowser
path = viz.export_html(graph, "model.html")
webbrowser.open(path)
```

### Slow Event Visualization

For large event datasets (>1M events):

```python
# Reduce number of frames/bins
viz.plot_events(events, n_frames=10, n_grid_bins=3)
```

### Images Not Loading in Spike Visualization

If event animation or grid frames show empty boxes:

1. **Check PIL is installed**: Pillow is required for image encoding
   ```python
   from t1c import viz
   print(viz.PIL_AVAILABLE)  # Should be True
   ```

2. **Install if missing**:
   ```bash
   pip install pillow>=9.0
   ```

3. **Browser compatibility**: Open the HTML file in a modern browser (Chrome, Firefox, Edge)


