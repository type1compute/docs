---
sidebar_position: 2
---

# Architecture

## System Overview

TALON is a neuromorphic computing SDK built around a layered package architecture. Each package has a single responsibility and depends only on packages below it in the stack.

```mermaid
graph TD
    SDK["🦅 t1c-talon<br/>talon.sdk<br/>SDK meta-package + CLI"]

    SDK --> IR["talon-ir<br/>talon.ir<br/>Core IR primitives"]
    SDK --> BRIDGE["talon-bridge<br/>talon.bridge<br/>PyTorch export/import"]
    SDK --> VIZ["talon-viz<br/>talon.viz<br/>Graph visualization"]
    SDK --> GRAPH["talon-graph<br/>talon.graph<br/>Partitioning + placement"]
    SDK --> BACKEND["talon-backend<br/>talon.backend<br/>Compilation + simulation"]
    SDK --> IO["talon-io<br/>talon.io<br/>Event streaming + encoding"]

    BRIDGE --> IR
    VIZ --> IR
    GRAPH --> IR
    BACKEND --> IR
    BACKEND --> GRAPH

    style SDK fill:#1a1a2e,color:#e0e0ff,stroke:#4a4aff
    style IR fill:#16213e,color:#e0e0ff,stroke:#4a4aff
    style BRIDGE fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
    style VIZ fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
    style GRAPH fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
    style BACKEND fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
    style IO fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

**Dependency rules:**
- `talon.ir` has no dependency on PyTorch, visualization, or hardware
- `talon.bridge` imports `talon.ir`; the reverse is never allowed
- `talon.viz` imports `talon.ir`; does not import `talon.bridge`
- `talon.graph` imports `talon.ir` for node type inspection
- `talon.backend` imports `talon.ir` and `talon.graph`
- `talon.io` is fully independent; handles event data without any IR dependency
- `talon.sdk` imports all packages and re-exports a unified API

---

## Package Dependency Map

```
                          ┌──────────────────────────────┐
                          │         talon.ir              │
                          │  (numpy, rustworkx, h5py)    │
                          └──────────────┬───────────────┘
                                         │
        ┌────────────┬───────────────────┼────────────┬────────────┐
        │            │                   │            │            │
        ▼            ▼                   ▼            ▼            │
 ┌────────────┐ ┌─────────┐ ┌──────────────┐ ┌──────────┐        │
 │talon.bridge│ │talon.viz│ │  talon.graph │ │ talon.io │        │
 │ (torch)    │ │(pillow) │ │(numpy,scipy) │ │ (numpy)  │        │
 └────────────┘ └─────────┘ └──────┬───────┘ └──────────┘        │
                                    │                              │
                                    ▼                              │
                            ┌──────────────┐                      │
                            │talon.backend │◄─────────────────────┘
                            │(numpy)       │
                            └──────────────┘

              ┌────────────────────────────────────────────┐
              │                 talon.sdk                   │
              │          (all packages + click, rich)       │
              └────────────────────────────────────────────┘
```

---

## talon.ir: Core IR

The foundation layer. All other packages depend on it; it depends on nothing from the TALON stack.

```mermaid
graph LR
    subgraph talon.ir
        Node["node.py<br/>BaseNode dataclass"]
        Types["types.py<br/>Shape, Nodes, Edges"]
        Graph["graph.py<br/>Graph, Input, Output"]
        Linear["linear.py<br/>Affine, SpikingAffine"]
        Conv["conv.py<br/>Conv1d/2d, SConv, SDConv, SGhostConv"]
        Pool["pooling.py<br/>MaxPool2d, AvgPool2d"]
        Neurons["neurons.py<br/>LIF, IF, ReLU, ...BatchNorm"]
        Ghost["ghost.py<br/>ChannelSplit, Concat, GhostBlocks"]
        Detect["detect.py<br/>SDDetect, DFLDecode, Dist2BBox, NMS"]
        Skip["skip.py<br/>Skip, SkipType"]
        Upsample["upsample.py<br/>Upsample, UpsampleMode"]
        Flatten["flatten.py<br/>Flatten"]
        Serial["serialization.py<br/>read/write HDF5"]
        Utils["utils.py<br/>Shape utilities"]
    end

    Node --> Graph
    Types --> Graph
    Graph --> Serial
    Linear --> Node
    Conv --> Node
    Pool --> Node
    Neurons --> Node
    Ghost --> Node
    Detect --> Node
    Skip --> Node
    Upsample --> Node
    Flatten --> Node

    style talon.ir fill:#16213e,color:#e0e0ff,stroke:#4a4aff
```

### Primitive Categories

TALON IR provides 36 primitives across 8 categories:

| Category | Primitives |
|----------|-----------|
| Linear | Affine, SpikingAffine |
| Convolution | Conv1d, Conv2d, SepConv2d, SConv, SDConv, SGhostConv |
| Spatial | MaxPool2d, AvgPool2d, Upsample, Flatten |
| SNN Neurons | LIF, IF |
| ANN Activations | ReLU, Sigmoid, Tanh, Softmax, GELU, ELU, PReLU |
| Normalization | BatchNorm1d, BatchNorm2d, LayerNorm, Dropout |
| Routing | ChannelSplit, Concat, Skip, HybridRegion |
| Ghost/Detection | SGhostEncoderLite, GhostBasicBlock1/2, SDDetect, DFLDecode, Dist2BBox, NMS |

### Graph Lifecycle

```
Define nodes dict + edges list
        ↓
ir.Graph(nodes, edges)      ← rustworkx DAG, validated on construction
        ↓
ir.write('model.t1c', graph)  ← HDF5, all params as datasets
        ↓
ir.read('model.t1c')          ← deserializes back to Graph
```

---

## talon.bridge: PyTorch Bridge

Bidirectional conversion between PyTorch `nn.Module` objects and `talon.ir.Graph`.

```mermaid
graph LR
    subgraph talon.bridge
        ToIR["to_ir.py<br/>PyTorch → IR<br/>TALONExporter, torch_to_ir"]
        FromIR["from_ir.py<br/>IR → PyTorch<br/>node_to_module, LIFModule, ..."]
        Executor["executor.py<br/>GraphExecutor<br/>CyclicGraphExecutor"]
        Tracer["tracer.py<br/>TALONTracer<br/>symbolic tracing"]
        Runtime["runtime.py<br/>load() helper"]
        GUtils["graph_utils.py<br/>topological ordering"]
        Quantize["quantize.py<br/>weight quantization"]
        Utils["utils.py<br/>sanitize_module_name, etc."]
    end

    Tracer --> ToIR
    ToIR --> GUtils
    FromIR --> Executor
    Runtime --> FromIR
    Quantize --> ToIR

    style talon.bridge fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

### Export Flow (PyTorch → IR)

```
nn.Module
    │
    ├── TALONTracer (symbolic trace)
    │       └── captures module graph, shapes, dtypes
    │
    ├── node_converter (module → IR node)
    │       └── Linear → Affine, Conv → Conv2d, snn.Leaky → LIF, ...
    │
    └── ir.Graph(nodes, edges)
```

### Import Flow (IR → PyTorch)

```
ir.Graph
    │
    ├── node_to_module (IR node → nn.Module)
    │       └── Affine → Linear, LIF → LIFModule, SDConv → DepthwiseConv2d, ...
    │
    ├── GraphExecutor
    │       ├── topological sort of nodes
    │       ├── routes tensors along edges during forward()
    │       └── handles multi-output nodes (ChannelSplit → tuple)
    │
    └── CyclicGraphExecutor (for recurrent/SNN graphs)
            └── carries hidden state across timesteps
```

### Multi-Output Routing

`ChannelSplit` nodes are marked with `_is_multi_output = True`. The `GraphExecutor` uses a set of such node names to correctly route tuple outputs without treating them as hidden state:

```python
from talon import ir, bridge
import numpy as np

split = ir.ChannelSplit(split_sections=[8, 8], dim=1)
concat = ir.Concat(num_inputs=2, dim=1)

graph = ir.Graph(
    nodes={'x': ir.Input(...), 'split': split, 'concat': concat, 'y': ir.Output(...)},
    edges=[('x', 'split'), ('split', 'concat'), ('split', 'concat'), ('concat', 'y')]
)
executor = bridge.ir_to_torch(graph)
```

---

## talon.viz: Visualization

Generates self-contained HTML visualizations using D3.js and Dagre. No server required.

```mermaid
graph LR
    subgraph talon.viz
        Init["__init__.py<br/>visualize(), export_html()"]
        Serialize["serialize.py<br/>graph_to_dict, serialize_node"]
        Render["render.py<br/>render_html, load_template"]
        Display["display.py<br/>export_html, list_graphs, serve"]
        Patterns["patterns.py<br/>PatternDetector, RepConv, SPP, SPPF"]
        Compiled["compiled.py<br/>visualize_partitioned<br/>visualize_execution_schedule"]
        Spikes["spikes.py<br/>export_events_html, spike raster"]
        Encode["encode.py<br/>encode_array, format_shape"]
        Constants["constants.py<br/>NODE_COLORS, NODE_SHAPES, NODE_ICONS"]
    end

    Init --> Serialize
    Init --> Display
    Serialize --> Encode
    Serialize --> Constants
    Render --> Serialize
    Display --> Render
    Compiled --> Serialize

    style talon.viz fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

### Visualization Pipeline

```
ir.Graph
    │
    ├── graph_to_dict()       ← JSON-serializable representation
    │       ├── serialize_node() per node (type, shape, params, color, icon)
    │       └── detect_all_patterns() (RepConv, SPP, skip connections, fan-out)
    │
    ├── render_html()         ← inject graph_dict into Jinja2 template
    │       └── viewer.html template with D3.js + Dagre
    │
    └── export_html() / serve()
```

### Pattern Detection

```
PatternDetector.detect_all_patterns(graph)
    ├── detect_repconv_patterns()      → RepConv blocks
    ├── detect_spp_patterns()          → Spatial Pyramid Pooling
    ├── detect_sppf_patterns()         → Fast SPP (YOLOv8 style)
    ├── detect_skip_connection_patterns() → residual / concatenate skips
    ├── find_fan_out_points()          → nodes with multiple output branches
    └── find_merge_points()            → nodes with multiple inputs
```

---

## talon.graph: Partitioning and Placement

Maps IR graphs onto neuromorphic hardware meshes.

```mermaid
graph LR
    subgraph talon.graph
        Init["__init__.py<br/>HardwareSpec, partition, route, allocate, place"]
        HW["HardwareSpec<br/>max_neurons, sram, num_cores, mesh, feedback_delay"]
        Partition["partition.py<br/>partition_greedy, partition_edgemap, partition_spectral"]
        Route["routing.py<br/>route() → routing_table per core"]
        Alloc["resource.py<br/>allocate() → ResourceBudget per core"]
        Place["placement.py<br/>place() → PlacementResult with hop distance"]
        Constraints["constraints.py<br/>validate_partition, check_sram, check_neurons"]
    end

    HW --> Partition
    Partition --> Route
    Route --> Alloc
    Alloc --> Place
    Constraints --> Partition
    Constraints --> Route

    style talon.graph fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

### Partitioning Algorithms

| Algorithm | Strategy | Best For |
|-----------|----------|---------|
| `partition_greedy` | BFS-order, fill cores sequentially | Dense feedforward networks |
| `partition_edgemap` | Minimize cross-core edges greedily | Networks with many skip connections |
| `partition_spectral` | Graph Laplacian spectral bisection | Balanced partitions on complex topologies |

### Core Metrics

```
Core utilization  = layer_sram_bytes / sram_bytes_per_core
Hop distance      = Σ manhattan_dist(src_core, dst_core) over cross-core edges
PlacementResult.improvement = (baseline_hops - placed_hops) / baseline_hops
```

---

## talon.backend: Compilation and Simulation

Compiles IR graphs to hardware descriptors and simulates execution on CPU and FPGA targets.

```mermaid
graph LR
    subgraph talon.backend
        Init["__init__.py<br/>get_backend, list_backends, register_backend"]
        Registry["registry.py<br/>BackendRegistry"]
        Base["base.py<br/>BackendBase (validate, compile, simulate, profile)"]
        Config["config.py<br/>CompileConfig, NeuronConfig"]
        Run["run.py<br/>run() helper (compile + simulate)"]

        subgraph cpu
            CPUSim["cpu/simulator.py<br/>CPUBackend.simulate()"]
            CPUProf["cpu/profiler.py<br/>CPUProfiler, ENERGY_PRESETS"]
            CPUUtils["cpu/graph_utils.py<br/>topological execution order"]
        end

        subgraph fpga
            HLS["fpga/hls4ml_backend.py<br/>HLS4MLBackend, BitstreamConfig"]
            ConfigGen["fpga/config_generator.py<br/>ir_to_hls4ml_config, parameters.h"]
            Zynq["fpga/zynq.py<br/>Zynq7020Backend, ZynqUSPlusBackend"]
            Templates["fpga/templates.py<br/>HLS template strings"]
        end
    end

    Registry --> Base
    Base --> CPUSim
    Base --> HLS
    Config --> CPUSim
    CPUSim --> CPUProf
    CPUSim --> CPUUtils
    HLS --> ConfigGen
    HLS --> Zynq
    ConfigGen --> Templates

    style talon.backend fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

### Simulation Pipeline

```
ir.Graph + CompileConfig
    │
    ├── validate()     ← check all node types are supported
    ├── compile()      ← produce HardwareDescriptor (JSON/dict)
    ├── simulate()     ← CPU forward pass for n_steps timesteps
    │       └── returns SimulationResult (outputs, spikes, membrane per core)
    └── profile()      ← SimulationResult + energy estimation
            └── returns ProfileResult with .summary() formatted string
```

### Energy Presets

```
ENERGY_PRESETS = {
    "45nm_cmos":           {mac_pj: 1.0,  spike_pj: 0.1,  sram_pj: 2.0}
    "zynq_7020":           {mac_pj: 4.5,  spike_pj: 0.5,  sram_pj: 8.0}
    "zynq_us_plus":        {mac_pj: 3.2,  spike_pj: 0.35, sram_pj: 6.0}
    "neuromorphic_int8":   {mac_pj: 0.08, spike_pj: 0.03, sram_pj: 0.5}
    "t1c_asic_target":     {mac_pj: 0.05, spike_pj: 0.02, sram_pj: 0.3}
}
```

---

## talon.io: Event Streaming and Encoding

Handles neuromorphic event data I/O, neural spike encoding, and sensor integration. Fully independent of `talon.ir`.

```mermaid
graph LR
    subgraph talon.io
        Init["__init__.py<br/>streaming, formats, encoding, sync, ethernet, dvs, aedat4, h5"]
        Stream["streaming.py<br/>EventBuffer, BufferedEventReader<br/>generate_random_events"]
        Formats["formats.py<br/>events_to_frames, events_to_raster<br/>events_to_grid, events_to_voxel"]
        Encode["encoding.py<br/>rate_encode, latency_encode<br/>delta_encode, temporal_encode"]
        Sync["sync.py<br/>SensorSync, align_timestamps"]
        Ethernet["ethernet.py<br/>UDPEventStreamer"]
        DVS["dvs.py<br/>DVSReader (iniVation/dv-processing)"]
        AEDAT["aedat4.py<br/>AEDAT4Reader (aedat library)"]
        H5["h5.py<br/>H5EventReader, H5EventWriter<br/>read_h5, write_h5"]
        Throughput["throughput.py<br/>benchmark_throughput<br/>ThroughputResult"]
    end

    Init --> Stream
    Init --> Formats
    Init --> Encode
    Init --> Sync
    Init --> Ethernet
    Init --> DVS
    Init --> AEDAT
    Init --> H5
    Init --> Throughput

    style talon.io fill:#0f3460,color:#e0e0ff,stroke:#4a4aff
```

### Event Data Model

```python
EVENT_DTYPE = np.dtype([
    ('t', np.int64),    # timestamp in microseconds
    ('x', np.int16),    # pixel x coordinate
    ('y', np.int16),    # pixel y coordinate
    ('p', np.int8),     # polarity: 0 (OFF) or 1 (ON)
])
```

### Encoding Schemes

| Scheme | Description | Output Shape |
|--------|-------------|-------------|
| `rate_encode` | Poisson spike train from firing rate | `(T, N)` |
| `latency_encode` | Single spike at time proportional to intensity | `(T, N)` |
| `delta_encode` | Spike on pixel value change | `(T, H, W, 2)` |
| `temporal_encode` | Threshold-crossing temporal codes | `(T, N)` |

---

## talon.sdk: SDK Meta-Package

`talon.sdk` is the unified entry point. It re-exports the full API of all six sub-packages plus SDK-specific tools.

```mermaid
graph TD
    subgraph "talon.sdk (t1c-talon)"
        Init["__init__.py<br/>re-exports all sub-packages"]
        CLI["cli.py<br/>talon / t1c CLI<br/>analyze, profile, lint, compare,<br/>inspect, validate, convert,<br/>pipeline, energy, run"]
        Analyze["analyze.py<br/>analyze_graph() → GraphStats"]
        Profile["profile.py<br/>profile_graph() → HardwareProfile"]
        Lint["lint.py<br/>lint_graph() → LintResult"]
        Compare["compare.py<br/>compare_graphs(), assert_graphs_equal()"]
        Convert["convert.py<br/>convert_to_spiking(), quantize_weights()"]
        Query["query.py<br/>inspect_node(), find_pattern()"]
        Fingerprint["fingerprint.py<br/>fingerprint_graph(), stamp_graph()"]
        Energy["energy.py<br/>estimate_energy() → EnergyEstimate"]
        Pipeline["pipeline.py<br/>TALONPipeline (partition+compile+run)"]
        Visualize["visualize.py<br/>TALONVisualizer (rich terminal + HTML)"]
        Version["version.py<br/>get_versions() → {talon.* : ver}"]
        Main["__main__.py<br/>python -m talon.sdk"]
    end

    Init --> Analyze
    Init --> Profile
    Init --> Lint
    Init --> Compare
    Init --> Convert
    Init --> Query
    Init --> Fingerprint
    Init --> Energy
    Init --> Pipeline
    Init --> Visualize
    Init --> Version
    CLI --> Analyze
    CLI --> Profile
    CLI --> Lint
    CLI --> Pipeline
    Main --> CLI

    style "talon.sdk (t1c-talon)" fill:#1a1a2e,color:#e0e0ff,stroke:#4a4aff
```

### CLI Commands

```
talon --help          # Show all commands
talon info            # Ecosystem version table
talon primitives      # List all 36 IR primitives

talon analyze  <model.t1c>          # Layer stats, parameter counts
talon profile  <model.t1c>          # Latency/memory/energy estimation
talon lint     <model.t1c>          # Validate IR constraints
talon compare  <a.t1c> <b.t1c>      # Diff two graphs
talon inspect  <model.t1c> --json   # JSON node/edge dump
talon validate <model.t1c>          # Quick validity check

talon convert  <model.t1c> --to spiking  # ANN → SNN conversion
talon energy   <model.t1c>               # Energy breakdown
talon run      <model.t1c> --steps 10    # CPU simulation
talon pipeline <model.t1c>               # Full partition+compile+run
```

---

## Full System Data Flow

End-to-end workflow from PyTorch model to hardware deployment:

```mermaid
flowchart TD
    PT["PyTorch / snnTorch\nnn.Module"] -->|"bridge.to_ir()"| IR["talon.ir.Graph\n(.t1c file)"]
    IR -->|"viz.visualize()"| HTML["Interactive HTML\nvisualization"]
    IR -->|"graph.partition()"| PART["Partitioned Graph\n(core assignments)"]
    PART -->|"graph.route()"| ROUTED["Routed Graph\n(routing tables)"]
    ROUTED -->|"graph.allocate()"| ALLOC["Resource Budget\n(SRAM, neurons per core)"]
    ALLOC -->|"graph.place()"| PLACED["Placed Graph\n(hop-optimized layout)"]
    PLACED -->|"backend.compile()"| DESC["HardwareDescriptor\n(JSON / HLS config)"]
    DESC -->|"backend.simulate()"| SIM["SimulationResult\n(spikes, membrane)"]
    SIM -->|"backend.profile()"| PROF["ProfileResult\n(energy, latency)"]
    DESC -->|"fpga.build_bitstream()"| FPGA["FPGA Bitstream\n(Xilinx Zynq HLS)"]
    IR -->|"bridge.ir_to_torch()"| EXEC["GraphExecutor\n(runnable nn.Module)"]

    EVT["Event Camera\nDVS / Prophesee"] -->|"io.H5EventWriter"| H5["H5 Event File"]
    H5 -->|"io.encoding.*_encode()"| SPIKES["Spike Trains\nnp.ndarray (T, N)"]
    SPIKES --> EXEC

    style PT fill:#0f3460,color:#e0e0ff
    style IR fill:#16213e,color:#e0e0ff,stroke:#4a4aff,stroke-width:3px
    style FPGA fill:#1a1a2e,color:#e0e0ff
    style EXEC fill:#0f3460,color:#e0e0ff
```

---

## Design Decisions

### Why HDF5?

- Efficient storage of large weight matrices (float16/32/64, int8/16/32)
- Self-describing format with embedded metadata and version info
- Wide language support (Python, C++, MATLAB, Julia)
- Compression support for deployment on embedded targets
- Compatible with myhdf5.hdfgroup.org web viewer for inspection

### Why Separate Packages?

| Concern | Benefit |
|---------|---------|
| Dependency isolation | `talon.ir` works without PyTorch; `talon.io` works without `talon.ir` |
| Independent versioning | IR can evolve separately from framework bindings |
| Clear ownership | Different teams own different packages |
| Focused testing | Each package has its own test suite (327 + 171 + 162 + 61 + 96 + 63 + 142 = 1022 tests) |
| Deployment flexibility | Hardware teams only need `talon.ir` + `talon.backend`, not PyTorch |

### Why Namespace Packages?

All packages share the `talon` namespace:

```python
from talon import ir, bridge, viz, sdk
from talon.ir import Graph, LIF, Conv2d
from talon.backend import get_backend
```

This allows the full ecosystem to feel like a single library (`talon`) while still being independently installable packages (`talon-ir`, `talon-bridge`, etc.). The SDK meta-package (`t1c-talon`) pulls in everything.

### Talon Package Manifest

| PyPI Package | Python Namespace | Repository |
|-------------|-----------------|------------|
| `talon-ir` | `talon.ir` | t1cir |
| `talon-bridge` | `talon.bridge` | t1ctorch |
| `talon-viz` | `talon.viz` | t1cviz |
| `talon-graph` | `talon.graph` | t1cgraph |
| `talon-backend` | `talon.backend` | t1cbackend |
| `talon-io` | `talon.io` | t1cio |
| `t1c-talon` | `talon.sdk` | t1c-sdk |
