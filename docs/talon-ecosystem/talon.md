---
sidebar_position: 10
---

# TALON — Tactical AI at Low-power On-device Nodes

**TALON** is the unified development kit for Type 1 Compute neuromorphic computing. It provides analysis, profiling, conversion, and deployment tools for spiking neural networks.

## Installation

```bash
pip install t1c-talon
```

Or with uv:

```bash
uv add t1c-talon
```

This installs the SDK along with all ecosystem packages:
- **talon.ir** (PyPI: `talon-ir`) - Core IR primitives (36) and HDF5 serialization
- **talon.bridge** (PyPI: `talon-bridge`) - PyTorch export/import bridge with mixed-precision quantization
- **talon.viz** (PyPI: `talon-viz`) - Interactive graph & spike visualization, pattern detection
- **talon.graph** (PyPI: `talon-graph`) - Graph partitioning, hardware placement, spike routing
- **talon.backend** (PyPI: `talon-backend`) - Backend compilation, CPU simulation/profiling, HLS4ML FPGA config
- **talon.io** (PyPI: `talon-io`) - Event streaming, sensor I/O (EVT2/EVT3/AEDAT4), neural encoding

## Tutorials

- [Tutorial: TALON SDK](./tutorial/tutorial_talon) — Step-by-step guide with inline code and output
- [Tutorial: Bridge](./tutorial/tutorial_bridge) — Export/import, stateful LIF, CyclicGraphExecutor
- [Tutorial: Backend](./tutorial/tutorial_backend) — Simulate, profile, energy presets, FPGA
- [Tutorial: snnTorch Integration](./tutorial/tutorial_snntorch_integration) — snnTorch export/import with SDK analysis
- [Tutorial: End-to-End Pipeline](./tutorial/tutorial_end_to_end) — Full workflow from model to hardware

For local notebook execution, see the [example_notebooks](./example_notebooks/) directory.

## Quick Start

```python
from talon import sdk

# Export PyTorch model to TALON IR
graph = sdk.to_ir(model, sample_input)
sdk.write('model.t1c', graph)

# Analyze and profile
stats = sdk.analyze_graph(graph)
profile = sdk.profile_graph(graph)

# Visualize
sdk.visualize(graph)

# Import and execute
executor = sdk.ir_to_torch('model.t1c', return_state=True)
output, state = executor(input_tensor, state)
```

---

## CLI Reference

The SDK provides a comprehensive command-line interface:

| Command | Description |
|---------|-------------|
| `talon info` | Ecosystem version information |
| `talon analyze FILE` | Graph structure and statistics |
| `talon profile FILE` | Hardware profiling and resource estimation |
| `talon compare A B` | Compare two graphs (structural + numerical diff) |
| `talon convert FILE` | Convert to SpikingAffine, quantize weights |
| `talon validate FILE` | Validate graph structure |
| `talon lint FILE` | Lint graph for common issues |
| `talon hash FILE` | Generate deterministic fingerprint |
| `talon stamp FILE` | Add provenance metadata |
| `talon node FILE NODE` | Inspect specific node |
| `talon trace FILE A B` | Trace paths from A to B |
| `talon visualize FILE` | Interactive browser visualization |
| `talon export-html FILE` | Export to standalone HTML |
| `talon primitives` | List available primitives |
| `talon quantize FILE` | Mixed-precision fixed-point quantization |
| `talon partition FILE` | Graph partitioning for hardware cores |
| `talon compile FILE` | Compile to hardware descriptor (JSON/binary) |
| `talon simulate FILE` | CPU simulation for correctness validation |
| `talon energy FILE` | Energy estimation (MAC-based, 45nm process) |
| `talon pipeline FILE` | Full pipeline: load -> lint -> partition -> compile -> simulate |
| `talon run FILE` | Quick simulation (graph.run convenience) |
| `talon profile-hw FILE` | From-scratch CPU profiler (latency/energy) |

> All commands also available as `t1c <command>` (alias).

### CLI Examples

```bash
# Analyze graph structure
$ talon analyze model.t1c -v

TALON IR Graph Summary
==================================================
Nodes: 7  |  Edges: 6  |  Depth: 6  |  Width: 1
Parameters: 235.1K  |  Memory: 918.5 KB  |  FLOPs: 469.5K

Layer Types:
  Affine: 2
  LIF: 2
  Input: 1
  Output: 1

SNN: 2 LIF neuron layer(s)

# Hardware profiling
$ talon profile model.t1c

TALON Hardware Profile
==================================================

Memory Estimates:
  Weight memory:     918.5 KB
  Activation memory: 3.1 KB (peak)
  State memory:      512 B (LIF membrane)
  Total:             922.1 KB
  8-bit quantized:   232.7 KB (estimate)

Compute Estimates:
  MAC operations: 234.8K
  Spike operations: 384

# Compare two graphs
$ talon compare model_v1.t1c model_v2.t1c

╭─────────── Graph Comparison ───────────╮
│ model_v1.t1c vs model_v2.t1c           │
╰────────────────────────────────────────╯
✗ Graphs differ

Modified nodes: 2
  fc1:
    - weight: max_diff=1.23e-04
Max weight difference: 1.23e-04

# Lint for issues
$ talon lint model.t1c

✓ Graph is valid
Warnings: 2
  [LIF_HIGH_TAU] Node 'lif2' has high tau=85.2
  [LARGE_MODEL] Graph has 10,523,456 parameters

# Inspect specific node
$ talon node model.t1c fc1

Node: fc1 (Affine)
├─ Inputs: input
├─ Outputs: lif1
├─ Weight: (256, 784), mean=-0.0012, std=0.0357
└─ Bias: (256,), mean=0.0001, std=0.0089

# Convert to SpikingAffine
$ talon convert model.t1c --spiking --weight-bits 8 -o model_hw.t1c
✓ Converted 2 Affine layers to SpikingAffine
✓ Saved to: model_hw.t1c
```

---

## Python API - Core Methods

This section provides detailed documentation for all core SDK methods. These are the functions you'll call directly from your Python code.

---

### Analysis Module

#### `analyze_graph(graph) → GraphStats`

Analyze a TALON IR graph and return comprehensive statistics.

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Graph object or path to `.t1c` file

**Returns:**
- `GraphStats` dataclass with the following fields:
  - `total_params` (int): Total learnable parameters
  - `total_bytes` (int): Total memory in bytes
  - `total_flops` (int): Total floating-point operations
  - `node_count` (int): Number of nodes
  - `edge_count` (int): Number of edges
  - `depth` (int): Longest path length (graph depth)
  - `width` (int): Maximum nodes at any depth level
  - `type_counts` (dict): Count of each node type
  - `layers` (list): List of `LayerStats` for each node
  - `input_shapes` (list): Input tensor shapes
  - `output_shapes` (list): Output tensor shapes
  - `memory_breakdown` (dict): Memory usage by node type
  - `lif_count` (int): Number of LIF neuron layers
  - `affine_count` (int): Number of FC layers
  - `conv_count` (int): Number of convolution layers

**Example:**
```python
from talon import sdk

stats = sdk.analyze_graph("model.t1c")

print(f"Parameters: {sdk.format_number(stats.total_params)}")  # "235.1K"
print(f"Memory: {sdk.format_bytes(stats.total_bytes)}")        # "918.5 KB"
print(f"Depth: {stats.depth}, Width: {stats.width}")       # "Depth: 6, Width: 1"
print(f"LIF layers: {stats.lif_count}")                    # "LIF layers: 2"

# Per-layer breakdown
for layer in stats.layers:
    if layer.params > 0:
        print(f"  {layer.name}: {layer.params:,} params")
```

---

#### `analyze_node(name, node) → LayerStats`

Analyze a single node and return its statistics.

**Parameters:**
- `name` (str): Node name
- `node` (`talon.ir.Node`): Node object

**Returns:**
- `LayerStats` dataclass with:
  - `name` (str): Node name
  - `node_type` (str): Node type (e.g., "Affine", "LIF")
  - `params` (int): Parameter count
  - `bytes` (int): Memory in bytes
  - `input_shape` (tuple): Input tensor shape
  - `output_shape` (tuple): Output tensor shape
  - `flops` (int): FLOPs for this layer
  - `config` (dict): Layer-specific configuration

**Example:**
```python
from talon import sdk, ir

graph = ir.read("model.t1c")
layer_stats = sdk.analyze_node("fc1", graph.nodes["fc1"])

print(f"Type: {layer_stats.node_type}")       # "Affine"
print(f"Params: {layer_stats.params:,}")      # "200,960"
print(f"Shape: {layer_stats.input_shape} → {layer_stats.output_shape}")
print(f"Config: {layer_stats.config}")        # {"weight_shape": (256, 784)}
```

---

#### `summarize(graph, verbose=False) → str`

Generate a human-readable summary string for a graph.

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Graph or path
- `verbose` (bool): Include per-layer breakdown (default: False)

**Returns:**
- Formatted summary string

**Example:**
```python
from talon import sdk

print(sdk.summarize("model.t1c", verbose=True))
# TALON IR Graph Summary
# ==================================================
# Nodes: 7  |  Edges: 6  |  Depth: 6  |  Width: 1
# Parameters: 235.1K  |  Memory: 918.5 KB  |  FLOPs: 469.5K
# ...
```

---

#### `format_bytes(bytes) → str`

Format byte count as human-readable string.

```python
from talon import sdk

sdk.format_bytes(1024)        # "1.0 KB"
sdk.format_bytes(1048576)     # "1.0 MB"
sdk.format_bytes(500)         # "500 B"
```

---

#### `format_number(n) → str`

Format large numbers with K/M suffix.

```python
from talon import sdk

sdk.format_number(500)        # "500"
sdk.format_number(1500)       # "1,500"
sdk.format_number(150000)     # "150.0K"
sdk.format_number(1500000)    # "1.50M"
```

---

### Comparison Module

#### `compare_graphs(graph_a, graph_b, atol=1e-6, rtol=1e-5) → GraphDiff`

Compare two TALON IR graphs and return detailed differences.

**Parameters:**
- `graph_a` (`talon.ir.Graph | str`): First graph
- `graph_b` (`talon.ir.Graph | str`): Second graph
- `atol` (float): Absolute tolerance for numerical comparison
- `rtol` (float): Relative tolerance for numerical comparison

**Returns:**
- `GraphDiff` dataclass with:
  - `identical` (bool): Exact match (structure + weights)
  - `structural_match` (bool): Same topology
  - `numerical_match` (bool): Same weights within tolerance
  - `nodes_added` (list): Nodes in B but not A
  - `nodes_removed` (list): Nodes in A but not B
  - `nodes_modified` (list): Nodes with differences
  - `nodes_unchanged` (list): Identical nodes
  - `edges_added` (list): New edges in B
  - `edges_removed` (list): Removed edges from A
  - `max_weight_diff` (float): Maximum weight difference
  - `mean_weight_diff` (float): Average weight difference
  - `node_diffs` (list): Detailed per-node diffs
  - `summary` (str): Human-readable summary

**Example:**
```python
from talon import sdk

diff = sdk.compare_graphs("model_v1.t1c", "model_v2.t1c")

if diff.identical:
    print("Graphs are identical!")
else:
    print(f"Structural match: {diff.structural_match}")
    print(f"Numerical match: {diff.numerical_match}")
    print(f"Modified nodes: {diff.nodes_modified}")
    print(f"Max weight diff: {diff.max_weight_diff:.2e}")
    
    # Detailed diff for each modified node
    for node_diff in diff.node_diffs:
        if node_diff.status == "modified":
            print(f"  {node_diff.name}: {node_diff.changes}")
```

---

#### `compare_nodes(name, node_a, node_b, atol=1e-6, rtol=1e-5) → NodeDiff`

Compare two individual nodes.

**Parameters:**
- `name` (str): Node name
- `node_a` (`talon.ir.Node`): First node
- `node_b` (`talon.ir.Node`): Second node
- `atol` (float): Absolute tolerance
- `rtol` (float): Relative tolerance

**Returns:**
- `NodeDiff` with:
  - `name` (str): Node name
  - `status` (str): "added", "removed", "modified", or "unchanged"
  - `changes` (list): List of change descriptions
  - `weight_diff` (float | None): Max weight difference

---

#### `assert_graphs_equal(graph_a, graph_b, atol=1e-6, rtol=1e-5, check_weights=True)`

Assert two graphs are equal, raising `AssertionError` if not.

**Parameters:**
- `graph_a`, `graph_b`: Graphs to compare
- `atol`, `rtol`: Tolerance values
- `check_weights` (bool): If False, only check structure

**Raises:**
- `AssertionError` with detailed message if graphs differ

**Example:**
```python
from talon import sdk, ir

# In a test
def test_roundtrip():
    graph = ir.read("model.t1c")
    ir.write("/tmp/copy.t1c", graph)
    reloaded = ir.read("/tmp/copy.t1c")
    
    # Raises AssertionError if different
    sdk.assert_graphs_equal(graph, reloaded)
```

---

### Profiling Module

#### `profile_graph(graph) → HardwareProfile`

Generate hardware resource estimates for a TALON IR graph.

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Graph or path

**Returns:**
- `HardwareProfile` dataclass with:
  - **Memory estimates (bytes):**
    - `weight_memory`: Weight storage
    - `activation_memory`: Peak activation memory
    - `state_memory`: LIF membrane state memory
    - `total_memory`: Sum of above
    - `estimated_quantized_memory`: 8-bit estimate
  - **Operation counts:**
    - `mac_ops`: Multiply-accumulate operations
    - `spike_ops`: Spike generation operations
    - `total_ops`: Total operations
  - **Layer counts:**
    - `conv_layers`, `fc_layers`, `lif_layers`, `pooling_layers`, `upsample_layers`
  - **Compatibility flags:**
    - `uses_spiking_affine`, `uses_skip`, `uses_sepconv`
  - **Analysis:**
    - `largest_layer`: Name of bottleneck layer
    - `largest_layer_memory`: Memory of bottleneck
    - `warnings`: List of potential issues
    - `recommendations`: List of optimization suggestions

**Example:**
```python
from talon import sdk

profile = sdk.profile_graph("model.t1c")

print(f"Weight memory: {sdk.format_bytes(profile.weight_memory)}")
print(f"Total memory: {sdk.format_bytes(profile.total_memory)}")
print(f"Quantized est: {sdk.format_bytes(profile.estimated_quantized_memory)}")
print(f"MAC ops: {profile.mac_ops:,}")
print(f"Spike ops: {profile.spike_ops:,}")

# Check for hardware compatibility
if profile.uses_spiking_affine:
    print("✓ Uses hardware-optimized SpikingAffine")
    
# Review recommendations
for rec in profile.recommendations:
    print(f"💡 {rec}")
```

---

#### `format_profile(profile) → str`

Format a HardwareProfile as a human-readable string.

**Example:**
```python
from talon import sdk

profile = sdk.profile_graph("model.t1c")
print(sdk.format_profile(profile))
```

---

### Conversion Module

#### `convert_to_spiking(graph, weight_bits=8, accumulator_bits=16, spike_mode="binary") → Graph`

Convert Affine layers to SpikingAffine for hardware optimization.

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Input graph
- `weight_bits` (int): Bit width for quantized weights (default: 8)
- `accumulator_bits` (int): Bit width for accumulator (default: 16)
- `spike_mode` (str): Spike encoding - "binary", "rate", or "temporal"

**Returns:**
- New graph with SpikingAffine layers

**Example:**
```python
from talon import sdk, ir

# Load model
graph = ir.read("model.t1c")

# Convert for hardware
hw_graph = sdk.convert_to_spiking(
    graph,
    weight_bits=8,
    accumulator_bits=16,
    spike_mode="binary"
)

# Save converted model
ir.write("model_hw.t1c", hw_graph)

# Metadata shows conversion info
print(hw_graph.metadata)
# {'converted_to_spiking': True, 'spiking_config': {...}, 'converted_layers': 3}
```

---

#### `quantize_weights(graph, bits=8, per_channel=True) → Graph`

Quantize graph weights to fixed-point representation (simulated).

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Input graph
- `bits` (int): Target bit width (default: 8)
- `per_channel` (bool): Use per-channel scaling vs per-tensor

**Returns:**
- New graph with quantized weights (float32 with quantized values)

**Note:** This performs simulated quantization for analysis. Actual hardware quantization is done by the compiler.

**Example:**
```python
from talon import sdk, ir

graph = ir.read("model.t1c")
quantized = sdk.quantize_weights(graph, bits=8)

# Check quantization impact
diff = sdk.compare_graphs(graph, quantized)
print(f"Max weight change: {diff.max_weight_diff:.6f}")
```

---

#### `batch_convert(sources, dest_dir, processor=None, overwrite=False) → list[ConversionResult]`

Batch convert multiple TALON IR graphs.

**Parameters:**
- `sources` (list): List of source .t1c file paths
- `dest_dir` (str | Path): Destination directory
- `processor` (callable | None): Optional function to transform each graph
- `overwrite` (bool): Whether to overwrite existing files

**Returns:**
- List of `ConversionResult` objects with:
  - `success` (bool): Whether conversion succeeded
  - `source` (str): Source file path
  - `destination` (str): Destination file path
  - `error` (str): Error message if failed
  - `stats` (dict): Node/edge counts

**Example:**
```python
from talon import sdk
from pathlib import Path

# Convert all models in a directory to spiking
sources = list(Path("models/").glob("*.t1c"))

results = sdk.batch_convert(
    sources,
    dest_dir="models_hw/",
    processor=lambda g: sdk.convert_to_spiking(g, weight_bits=8),
    overwrite=True
)

for r in results:
    status = "✓" if r.success else "✗"
    print(f"{status} {r.source} → {r.destination}")
```

---

#### `merge_graphs(*graphs, prefix=True) → Graph`

Merge multiple TALON IR graphs into a single graph.

**Parameters:**
- `*graphs`: Variable number of graphs to merge
- `prefix` (bool): Prefix node names with graph index to avoid collisions

**Returns:**
- Merged graph (no connections between sub-graphs)

---

#### `prune_disconnected(graph) → Graph`

Remove disconnected nodes from a graph.

**Parameters:**
- `graph` (`talon.ir.Graph | str`): Input graph

**Returns:**
- New graph with only connected nodes

---

### Linting Module

#### `lint_graph(graph, strict=False) → LintResult`

Lint a TALON IR graph for common issues and best practices.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to lint
- `strict` (bool): If True, treat warnings as errors

**Returns:**
- `LintResult` with:
  - `issues` (list): All detected issues
  - `errors` (list): Critical issues that will cause failures
  - `warnings` (list): Suspicious patterns
  - `infos` (list): Informational messages
  - `is_valid` (bool): True if no errors

**Checks performed:**
- Basic structure (dangling edges, self-loops, missing I/O)
- Unreachable nodes (disconnected from Input/Output)
- Multiple outputs without clear semantics
- Missing shapes on computational nodes
- Suspicious LIF parameters (tau, threshold)
- Skip node issues (wrong number of inputs)
- Naming conventions
- Large parameter counts

**Example:**
```python
from talon import sdk

result = sdk.lint_graph(graph)

if not result.is_valid:
    print("❌ Graph has critical errors!")
    for error in result.errors:
        print(f"  [{error.code}] {error.message}")
        if error.suggestion:
            print(f"    → {error.suggestion}")
else:
    print("✓ Graph is valid")

# Review warnings
for warning in result.warnings:
    print(f"⚠ [{warning.code}] {warning.message}")

# Convert to dict for JSON serialization
lint_dict = result.to_dict()
```

---

### Fingerprinting Module

#### `fingerprint_graph(graph, include_weights=True, include_metadata=False) → str`

Generate a deterministic fingerprint (SHA256 hash) of a graph.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to fingerprint
- `include_weights` (bool): Include parameter values in hash
- `include_metadata` (bool): Include metadata dict in hash

**Returns:**
- Hexadecimal hash string (64 characters)

**Use cases:**
- Verify graph identity before deployment
- Track which exact model ran on hardware
- Cache validation (invalidate if hash changes)
- Diff detection in CI/CD pipelines

**Example:**
```python
from talon import sdk

# Full fingerprint (structure + weights)
hash_full = sdk.fingerprint_graph(graph, include_weights=True)
print(f"Full hash: {hash_full[:16]}...")

# Structure-only fingerprint (ignores weight values)
hash_struct = sdk.fingerprint_graph(graph, include_weights=False)
print(f"Structure hash: {hash_struct[:16]}...")

# Structure hash changes only when topology changes
# Weight values can vary without affecting structure hash
```

---

#### `stamp_graph(graph, notes=None, git_commit=None, training_run_id=None, calibration_config=None, quantization_config=None) → Graph`

Add provenance metadata to a graph.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to stamp (not modified in-place)
- `notes` (str | None): Human-readable notes
- `git_commit` (str | None): Git commit hash
- `training_run_id` (str | None): Training run identifier (e.g., MLflow run ID)
- `calibration_config` (dict | None): Calibration configuration
- `quantization_config` (dict | None): Quantization configuration

**Returns:**
- New graph with stamped metadata including:
  - SDK versions (talon, talon.ir, talon.bridge, talon.viz)
  - UTC timestamp
  - Structure and full fingerprints
  - User-provided metadata

**Example:**
```python
from talon import sdk, ir

graph = ir.read("model.t1c")

stamped = sdk.stamp_graph(
    graph,
    notes="Production model v2.1, 94.5% accuracy on test set",
    git_commit="a1b2c3d4e5f6",
    training_run_id="mlflow-run-12345",
    quantization_config={
        "weight_bits": 8,
        "accumulator_bits": 16,
        "calibration_dataset": "train_subset"
    }
)

ir.write("model_production.t1c", stamped)
```

---

#### `get_stamp(graph) → dict | None`

Extract provenance stamp from a graph's metadata.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to read stamp from

**Returns:**
- Provenance dict if present, None otherwise

**Example:**
```python
from talon import sdk

stamp = sdk.get_stamp(graph)
if stamp:
    print(f"Created: {stamp['timestamp']}")
    print(f"SDK: {stamp['sdk_versions']['talon']}")
    print(f"Hash: {stamp['fingerprint_full'][:24]}...")
```

---

#### `verify_fingerprint(graph, expected_hash, include_weights=True) → bool`

Verify a graph matches an expected fingerprint.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to verify
- `expected_hash` (str): Expected hash
- `include_weights` (bool): Whether to include weights in verification

**Returns:**
- True if hashes match

**Example:**
```python
from talon import sdk, ir

# Save expected hash
expected = sdk.fingerprint_graph(original_graph)

# Later, verify loaded graph
loaded = ir.read("model.t1c")
if not sdk.verify_fingerprint(loaded, expected):
    raise ValueError("Graph was modified!")
```

---

### Query Module

#### `inspect_node(graph, node_name) → dict`

Get detailed information about a specific node.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph containing the node
- `node_name` (str): Name of node to inspect

**Returns:**
- Dict with:
  - `name` (str): Node name
  - `type` (str): Node type
  - `inputs` (list): Incoming edge sources
  - `outputs` (list): Outgoing edge destinations
  - `parameters` (dict): Parameter info with statistics
  - `shapes` (dict): Input/output shapes
  - `attributes` (dict): Node-specific attributes

**Raises:**
- `KeyError` if node doesn't exist

**Example:**
```python
from talon import sdk

info = sdk.inspect_node(graph, "fc1")

print(f"Type: {info['type']}")                    # "Affine"
print(f"Inputs: {info['inputs']}")                # ["input"]
print(f"Outputs: {info['outputs']}")              # ["lif1"]

# Weight statistics
w = info['parameters']['weight']
print(f"Weight shape: {w['shape']}")              # (256, 784)
print(f"Weight mean: {w['mean']:.4f}")
print(f"Weight std: {w['std']:.4f}")
print(f"Weight range: [{w['min']:.4f}, {w['max']:.4f}]")
```

---

#### `trace_path(graph, src, dst) → list[list[str]]`

Find all paths from source to destination node.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to search
- `src` (str): Source node name
- `dst` (str): Destination node name

**Returns:**
- List of paths, where each path is a list of node names

**Raises:**
- `KeyError` if source or destination doesn't exist

**Example:**
```python
from talon import sdk

paths = sdk.trace_path(graph, "input", "output")

print(f"Found {len(paths)} path(s)")
for path in paths:
    print(" → ".join(path))
    # input → fc1 → lif1 → fc2 → lif2 → fc3 → lif3 → output
```

---

#### `extract_subgraph(graph, node_names) → Graph`

Extract a subgraph containing only specified nodes.

**Parameters:**
- `graph` (`talon.ir.Graph`): Source graph
- `node_names` (list): Nodes to include

**Returns:**
- New graph with only specified nodes and edges between them

**Example:**
```python
from talon import sdk

# Extract first layer for analysis
subgraph = sdk.extract_subgraph(graph, ["input", "fc1", "lif1"])
```

---

#### `find_pattern(graph, pattern) → list[tuple]`

Find all occurrences of a node type pattern in the graph.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to search
- `pattern` (str): Pattern string, e.g., "Conv2d->LIF" or "Affine->LIF->Affine"

**Returns:**
- List of tuples, where each tuple contains node names matching the pattern

**Example:**
```python
from talon import sdk

# Find all FC→LIF sequences
matches = sdk.find_pattern(graph, "Affine->LIF")
print(f"Found {len(matches)} Affine→LIF patterns")
for affine, lif in matches:
    print(f"  {affine} → {lif}")

# Find Conv→LIF→Pool sequences
conv_patterns = find_pattern(graph, "Conv2d->LIF->MaxPool2d")
```

---

#### `get_node_statistics(graph) → dict`

Get statistics about node types and parameters.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to analyze

**Returns:**
- Dict with stats grouped by node type

**Example:**
```python
from talon import sdk

stats = sdk.get_node_statistics(graph)

for node_type, info in stats.items():
    print(f"{node_type}:")
    print(f"  Count: {info['count']}")
    print(f"  Nodes: {info['nodes']}")
    print(f"  Total params: {info['total_params']:,}")
```

---

### Visualization Module (re-exported from talon.viz)

#### `visualize(graph, **kwargs)`

Open interactive graph visualization in browser.

```python
from talon import sdk

sdk.visualize(graph, title="My SNN Model")
```

---

#### `export_html(graph, path, **kwargs)`

Export graph to standalone HTML file.

```python
from talon import sdk

sdk.export_html(graph, "model.html", title="Production Model")
```

---

### Export/Import Module (re-exported from talon.bridge)

#### `to_ir(module, sample_data, **kwargs) → Graph`

Export a PyTorch module to TALON IR graph.

```python
from talon import sdk

graph = sdk.to_ir(model, sample_input)
```

---

#### `ir_to_torch(graph_or_path, return_state=False, **kwargs) → GraphExecutor`

Import a TALON IR graph to a PyTorch executor.

```python
from talon import sdk

executor = sdk.ir_to_torch("model.t1c", return_state=True)
output, state = executor(input_tensor, state)
```

---

### Serialization (re-exported from talon.ir)

#### `read(path) → Graph`

Read a TALON IR graph from HDF5 file.

```python
from talon import sdk

graph = sdk.read("model.t1c")
```

---

#### `write(path, graph)`

Write a TALON IR graph to HDF5 file.

```python
from talon import sdk

sdk.write("model.t1c", graph)
```

---

### Energy Module

#### `estimate_energy(graph, spike_rate=0.1, timesteps=100) -> EnergyEstimate`

Estimate energy consumption using a MAC-based cost model (45nm process).

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to estimate
- `spike_rate` (float): Average LIF spike rate (0.0-1.0)
- `timesteps` (int): Number of timesteps per inference

**Returns:**
- `EnergyEstimate` with `total_uj`, `mac_energy_uj`, `spike_energy_uj`, `memory_energy_uj`, `per_layer`

**Example:**
```python
from talon import sdk

est = sdk.estimate_energy(graph, spike_rate=0.1, timesteps=100)
print(f"Total: {est.total_uj:.4f} uJ")
print(f"MACs: {est.mac_count:,}, Spikes: {est.spike_count:,}")
```

#### Device-Specific Energy Presets (via talon.backend)

The backend profiler supports device-specific energy coefficients:

```python
from talon.backend import get_backend, ENERGY_PRESETS, get_energy_preset

# Available presets
print(list(ENERGY_PRESETS.keys()))
# ['45nm_cmos', 'zynq_7020', 'zynq_us_plus', 'neuromorphic_int8', 't1c_asic_target']

cpu = get_backend("cpu")
profile = cpu.profile(graph, n_steps=10, energy_preset="zynq_us_plus")
print(profile.summary())  # Formatted report with MAC/spike/SRAM breakdown
```

Each preset provides `mac_pj`, `spike_pj`, `sram_read_pj`, `sram_write_pj` coefficients. All energy values are **estimates** based on analytical models, not live hardware measurement.

---

### Pipeline Module

#### `run_pipeline(path, config=None) -> PipelineResult`

Run the full deployment pipeline: load -> lint -> analyze -> partition -> compile -> simulate -> report.

**Parameters:**
- `path` (str): Path to .t1c file
- `config` (`PipelineConfig`): Pipeline configuration

**Returns:**
- `PipelineResult` with `success`, `errors`, `analysis`, `simulation`, `energy`

**Example:**
```python
from talon import sdk

result = sdk.run_pipeline("model.t1c", config=sdk.PipelineConfig(
    target="cpu", timesteps=100, partition=True, num_cores=4
))

if result.success:
    print(f"Nodes: {result.analysis.node_count}")
    print(f"Energy: {result.energy['total_uj']:.4f} uJ")
```

---

### Graph Partitioning & Hardware Mapping (from talon.graph)

#### `partition(graph, hw_spec, algorithm="greedy") -> Graph`

Partition a graph across hardware cores.

**Parameters:**
- `graph` (`talon.ir.Graph`): Graph to partition
- `hw_spec` (`HardwareSpec`): Hardware constraints
- `algorithm` (str): "greedy", "edgemap", or "spectral"

**Returns:**
- Graph with `partition_metadata` populated

**Example:**
```python
from talon import sdk

hw = sdk.HardwareSpec(max_neurons_per_core=256, num_cores=64, sram_bytes_per_core=65536)
partitioned = sdk.partition(graph, hw, algorithm="greedy")
placed = sdk.place(partitioned, hw)
routed = sdk.route(partitioned, hw)
resources = sdk.allocate(partitioned, hw)

print(f"Cores used: {partitioned.partition_metadata['num_cores_used']}")
print(f"Fits hardware: {resources.fits_hardware}")
```

---

### Backend Compilation & Simulation (from talon.backend)

#### `get_backend(name) -> BackendBase`

Get a registered backend by name (e.g., "cpu", "fpga").

**Example:**
```python
from talon import sdk

be = sdk.get_backend("cpu")
validation = be.validate(graph)
descriptor = be.compile(graph)
result = be.simulate(graph, n_steps=100)
profile = be.profile(graph, n_steps=100)

print(f"Valid: {validation.is_valid}")
print(f"Latency: {profile.total_latency_us:.1f} us")
print(f"Energy: {profile.energy_estimate_uj:.4f} uJ")
```

---

### Event I/O & Streaming (from talon.io)

The SDK re-exports all I/O modules for event camera data processing:

```python
from talon import sdk

# Event streaming with buffered reader
reader = sdk.BufferedEventReader(capacity=100_000)

# Neural encoding
from talon.io.encoding import rate_encode, latency_encode, delta_encode

# Format conversion (vectorized, >5M events/sec)
from talon.io.formats import decode_evt2, decode_evt3

# Sensor file I/O
from talon.io.dvs import read_prophesee, read_iniVation

# AEDAT4 frame unpacking
from talon.io.aedat4 import unpack_frames

# Multi-sensor synchronization
from talon.io.sync import align_sensors, merge_events

# UDP event streaming (10G ethernet)
from talon.io.ethernet import pack_events_udp, udp_event_receiver
```

---

## Data Classes

### GraphStats

```python
@dataclass
class GraphStats:
    total_params: int       # Total learnable parameters
    total_bytes: int        # Total memory (bytes)
    total_flops: int        # Total FLOPs
    node_count: int         # Number of nodes
    edge_count: int         # Number of edges
    depth: int              # Longest path length
    width: int              # Max nodes at same depth
    type_counts: dict       # {node_type: count}
    layers: list            # List of LayerStats
    input_shapes: list      # Input tensor shapes
    output_shapes: list     # Output tensor shapes
    memory_breakdown: dict  # {node_type: bytes}
    lif_count: int          # LIF neuron layers
    affine_count: int       # FC layers
    conv_count: int         # Conv layers
```

### GraphDiff

```python
@dataclass
class GraphDiff:
    identical: bool           # Exact match
    structural_match: bool    # Same nodes and edges
    numerical_match: bool     # Same weights (within tolerance)
    nodes_added: list         # Nodes in B but not A
    nodes_removed: list       # Nodes in A but not B
    nodes_modified: list      # Nodes with differences
    nodes_unchanged: list     # Identical nodes
    edges_added: list         # New edges
    edges_removed: list       # Removed edges
    max_weight_diff: float    # Maximum weight difference
    mean_weight_diff: float   # Mean weight difference
    node_diffs: list          # Per-node NodeDiff objects
    summary: str              # Human-readable summary
```

### HardwareProfile

```python
@dataclass
class HardwareProfile:
    # Memory (bytes)
    weight_memory: int
    activation_memory: int
    state_memory: int
    total_memory: int
    estimated_quantized_memory: int
    
    # Operations
    mac_ops: int
    spike_ops: int
    total_ops: int
    
    # Layer counts
    conv_layers: int
    fc_layers: int
    lif_layers: int
    pooling_layers: int
    upsample_layers: int
    
    # Features
    uses_spiking_affine: bool
    uses_skip: bool
    uses_sepconv: bool
    
    # Analysis
    largest_layer: str
    largest_layer_memory: int
    warnings: list
    recommendations: list
```

### LintResult

```python
class LintResult:
    issues: List[LintIssue]   # All detected issues
    errors: List[LintIssue]   # Critical issues
    warnings: List[LintIssue] # Suspicious patterns
    infos: List[LintIssue]    # Informational
    is_valid: bool            # True if no errors
    
    def to_dict(self) -> dict  # JSON-serializable
```

### LintIssue

```python
@dataclass
class LintIssue:
    severity: Severity        # ERROR, WARNING, INFO
    code: str                 # Issue code (e.g., "LIF_HIGH_TAU")
    message: str              # Human-readable message
    node: str | None          # Affected node (if applicable)
    suggestion: str | None    # How to fix
```

---

## Full API Reference

```python
from talon import sdk

# Access via sdk.*, e.g. sdk.analyze_graph, sdk.read, sdk.to_ir
# Full API (conceptual; use sdk.xxx in code):
from talon.sdk import (
    # Version
    __version__, get_versions, info,
    
    # Analysis
    analyze_graph, analyze_node, summarize,
    GraphStats, LayerStats,
    format_bytes, format_number,
    
    # Comparison
    compare_graphs, compare_nodes, assert_graphs_equal,
    GraphDiff, NodeDiff,
    
    # Profiling
    profile_graph, format_profile,
    HardwareProfile,
    
    # Conversion
    convert_to_spiking, quantize_weights,
    batch_convert, merge_graphs, prune_disconnected,
    ConversionResult,
    
    # Linting
    lint_graph, LintResult, LintIssue, Severity,
    
    # Fingerprinting
    fingerprint_graph, stamp_graph, get_stamp, verify_fingerprint,
    
    # Query
    inspect_node, trace_path, extract_subgraph,
    find_pattern, get_node_statistics,
    
    # talon.ir: Core Primitives
    Graph, Input, Output,
    Affine, SpikingAffine,
    Conv1d, Conv2d, SepConv2d,
    MaxPool2d, AvgPool2d, Upsample,
    Flatten, LIF, Skip,
    
    # talon.ir: ANN Activations (for hybrid architectures)
    ReLU, Sigmoid, Tanh, Softmax, GELU, ELU, PReLU,
    
    # talon.ir: Normalization
    BatchNorm1d, BatchNorm2d, LayerNorm,
    
    # talon.ir: Regularization
    Dropout,
    
    # talon.ir: Hybrid Architecture
    HybridRegion, NeuronMode,
    
    # talon.ir: Spiking Convolutions
    SConv, SDConv, SGhostConv, IF,
    
    # talon.ir: Ghost / Detect
    ChannelSplit, Concat,
    SGhostEncoderLite, GhostBasicBlock1, GhostBasicBlock2,
    SDDetect, DFLDecode, Dist2BBox, NMS,
    
    # talon.ir: Types and Enums
    Edges, Nodes, Shape,
    SkipType, SpikeMode, UpsampleMode,
    
    # talon.ir: Registry
    str_to_node, register_node, list_primitives,
    
    # talon.ir: Serialization
    read, write, read_version, T1CFormatError,
    
    # talon.bridge: Export/Import
    to_ir, torch_to_ir, TALONExporter,
    ir_to_torch, from_ir, load,
    
    # talon.bridge: Executor
    GraphExecutor, LIFModule, SkipModule,
    ChannelSplitModule, ConcatModule,
    SGhostConvModule, SGhostEncoderLiteModule,
    GhostBasicBlock1Module, GhostBasicBlock2Module,
    SDDetectModule, DFLDecodeModule, Dist2BBoxModule, NMSModule,
    
    # talon.bridge: Graph utilities
    validate_graph, has_cycles, get_disconnected_nodes,
    get_topological_order, get_input_nodes, get_output_nodes,
    
    # talon.viz: Graph visualization
    visualize, export_html, graph_to_dict, render_html,
    
    # talon.viz: Spike visualization
    plot_events, export_events_html, plot_frames,
    events_to_frames, events_to_grid, events_to_raster,
    TONIC_AVAILABLE, PIL_AVAILABLE,
    
    # talon.graph: Partitioning, Routing, Placement
    HardwareSpec, partition, partition_edgemap, partition_spectral,
    partition_greedy, route, RoutingTable,
    allocate, ResourceMap, CoreBudget,
    place, PlacementResult,
    
    # talon.backend: Compilation and Simulation
    BackendBase, get_backend, register_backend, list_backends,
    query_backends, CompileConfig, HardwareDescriptor,
    BackendCapabilities, ValidationResult,
    SimulationResult, ProfileResult,
    run_graph, RunResult,
    
    # talon.io: Event Streaming, Encoding, Format Conversion
    streaming, formats, encoding, sync, ethernet, dvs, aedat4,
    BufferedEventReader, EventBuffer,
    generate_random_events, EVENT_DTYPE,
    
    # talon.backend: Energy Presets
    ENERGY_PRESETS, get_energy_preset,
    
    # talon.io: Throughput Benchmarking
    throughput,
    
    # SDK Energy and Pipeline
    estimate_energy, EnergyEstimate, ENERGY_TABLE,
    run_pipeline, PipelineConfig, PipelineResult,
)
```
