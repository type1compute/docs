---
sidebar_position: 20
---

# Tutorial: End-to-End Pipeline

This tutorial demonstrates the complete TALON workflow — from defining a spiking neural network in PyTorch/snnTorch, through TALON IR export, hardware-aware analysis, partitioning, CPU simulation, energy profiling, and visualization. Every step in the pipeline is covered.

## What You'll Learn

1. Define and export a convolutional SNN
2. Graph analysis (parameters, FLOPs, memory)
3. Hardware profiling (memory breakdown, MAC/spike ops)
4. Linting and fingerprinting
5. Weight quantization (INT8)
6. Hardware partitioning and resource allocation
7. CPU backend simulation (multi-timestep)
8. Energy estimation with device presets
9. Visualization (graph, partitions, schedule)
10. Event encoding and I/O

## Prerequisites

```bash
pip install t1c-talon snntorch
```

---

## 1. Define the Model

A convolutional SNN for MNIST: `Conv2d` → `ReLU` → `MaxPool2d` → `Flatten` → `Affine` → `LIF` → `Affine` → `LIF`.

```python
import torch
import torch.nn as nn
import snntorch as snn

class ConvSNN(nn.Module):
    """Conv feature extractor with spiking classifier."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x, mem1=None, mem2=None):
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        spk1, mem1 = self.lif1(x, mem1)
        x = self.fc2(spk1)
        spk2, mem2 = self.lif2(x, mem2)
        return spk2, mem1, mem2

model = ConvSNN()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
```

**Output:**

```
Model: 402,986 parameters
```

---

## 2. Export to TALON IR

The snnTorch bridge traces the model and converts each layer to its TALON IR primitive.

```python
from snntorch.export_talonir import export_to_ir
from talon import ir

graph = export_to_ir(model, torch.randn(1, 1, 28, 28))

print(f"Exported: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
for name, node in graph.nodes.items():
    print(f"  {name:10s} → {type(node).__name__}")
```

**Output:**

```
Exported: 10 nodes, 9 edges
  input      → Input
  conv1      → Conv2d
  relu       → ReLU
  pool       → MaxPool2d
  flatten    → Flatten
  fc1        → Affine
  lif1       → LIF
  fc2        → Affine
  lif2       → LIF
  output     → Output
```

### Serialize

```python
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "conv_snn.t1c")
ir.write(path, graph)
print(f"Saved: {os.path.getsize(path):,} bytes")
```

**Output:**

```
Saved: 1,647,760 bytes
```

---

## 3. Graph Analysis

`analyze_graph()` computes parameter counts, memory usage, FLOPs, graph depth/width, and per-type breakdowns.

```python
from talon import sdk

stats = sdk.analyze_graph(graph)

print(f"Parameters:  {stats.total_params:,}")
print(f"Memory:      {stats.total_bytes:,} bytes")
print(f"FLOPs:       {stats.total_flops:,}")
print(f"Depth:       {stats.depth}")
print(f"Width:       {stats.width}")
print(f"LIF neurons: {stats.lif_count}")
print(f"Conv layers: {stats.conv_count}")
print(f"Types:       {stats.type_counts}")
```

**Output:**

```
Parameters:  403,538
Memory:      1,614,152 bytes
FLOPs:       1,031,168
Depth:       9
Width:       1
LIF neurons: 2
Conv layers: 1
Types:       {'Input': 1, 'Conv2d': 1, 'ReLU': 1, 'MaxPool2d': 1, 'Flatten': 1, 'Affine': 2, 'LIF': 2, 'Output': 1}
```

---

## 4. Hardware Profile

`profile_graph()` estimates memory layout, compute operations, and provides optimization recommendations.

```python
prof = sdk.profile_graph(graph)

print(f"Weight memory:     {prof.weight_memory:,} bytes")
print(f"Activation memory: {prof.activation_memory:,} bytes")
print(f"State memory:      {prof.state_memory:,} bytes")
print(f"Total memory:      {prof.total_memory:,} bytes")
print(f"MAC ops:           {prof.mac_ops:,}")
print(f"Spike ops:         {prof.spike_ops:,}")
print(f"Largest layer:     {prof.largest_layer}")
print(f"Recommendations:   {prof.recommendations}")
```

**Output:**

```
Weight memory:     1,614,152 bytes
Activation memory: 50,176 bytes
State memory:      552 bytes
Total memory:      1,664,880 bytes
MAC ops:           515,584
Spike ops:         138
Largest layer:     fc1
Recommendations:   ['Consider using SpikingAffine for FC layers to enable hardware-optimized quantization', '8-bit quantization could reduce memory by ~72%']
```

---

## 5. Linting & Fingerprinting

Lint catches potential issues before deployment. Fingerprinting generates a reproducibility hash.

```python
lint_result = sdk.lint_graph(graph)
print(f"Lint issues: {len(lint_result.issues)}")
for issue in lint_result.issues:
    print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")

fp = sdk.fingerprint_graph(graph)
print(f"\nFingerprint: {fp}")
```

**Output:**

```
Lint issues: 2
  [warning] PYTHON_KEYWORD: Node name 'input' is a Python keyword
  [warning] PYTHON_KEYWORD: Node name 'output' is a Python keyword

Fingerprint: 40c35f62bd3cd98c26df74a6cbf9df221737ee5cc8c95552ec4a5514a3eb6ae7
```

---

## 6. Weight Quantization

`quantize_weights()` applies per-channel INT8 quantization to reduce weight memory for hardware deployment.

```python
qgraph = sdk.quantize_weights(graph, bits=8, per_channel=True)

prof_q = sdk.profile_graph(qgraph)
print(f"FP32 weight memory: {prof.weight_memory:,} bytes")
print(f"INT8 weight memory: {prof_q.weight_memory:,} bytes")
```

**Output:**

```
FP32 weight memory: 1,614,152 bytes
INT8 weight memory: 1,614,152 bytes
```

:::note
`quantize_weights` stores the quantized values into the same weight arrays with reduced precision, but the serialized memory depends on the storage format. The estimated quantized memory (`prof.estimated_quantized_memory`) reflects the reduction for hardware targets.
:::

---

## 7. Hardware Partitioning

Map the graph onto multi-core neuromorphic hardware using `talon.graph`.

### Hardware Specification

```python
from talon.graph import partition, HardwareSpec, allocate, place

hw = HardwareSpec.zynq_us_plus()
hw.validate()

print(f"Cores:          {hw.num_cores}")
print(f"Neurons/core:   {hw.max_neurons_per_core}")
print(f"SRAM/core:      {hw.sram_bytes_per_core:,} bytes")
print(f"Feedback delay: {hw.max_feedback_delay}")
```

**Output:**

```
Cores:          32
Neurons/core:   512
SRAM/core:      98,304 bytes
Feedback delay: 4
```

### Partition

```python
partitioned = partition(graph, hw)
pm = partitioned.partition_metadata

print(f"Algorithm:  {pm['algorithm']}")
print(f"Cores used: {pm['num_cores_used']}/{hw.num_cores}")
```

**Output:**

```
Algorithm:  greedy
Cores used: 1/32
```

### Resource Allocation

```python
resources = allocate(partitioned, hw)

print(f"Total weight bytes:    {resources.total_weight_bytes:,}")
print(f"Total state bytes:     {resources.total_state_bytes:,}")
print(f"Peak core utilization: {resources.peak_core_utilization:.2f}")
print(f"Fits hardware:         {resources.fits_hardware}")
```

**Output:**

```
Total weight bytes:    1,611,944
Total state bytes:     2,208
Peak core utilization: 16.42
Fits hardware:         False
```

The model exceeds the SRAM budget for a single core (16.42x over budget). For deployment, the model would need quantization or partitioning across more cores.

### Placement

```python
placement = place(partitioned, hw)

print(f"Mapping:          {placement.logical_to_physical}")
print(f"Total hop distance: {placement.total_hop_distance}")
print(f"Improvement:       {placement.improvement:.1f}%")
```

**Output:**

```
Mapping:          {0: (4, 0)}
Total hop distance: 0
Improvement:       0.0%
```

---

## 8. CPU Simulation

Run the graph through the TALON CPU backend to simulate spiking dynamics across multiple timesteps.

```python
from talon.backend import get_backend

cpu = get_backend("cpu")

sim = cpu.simulate(graph, n_steps=10)
print(f"Timesteps:    {sim.timesteps_run}")
print(f"Spike counts: {sim.spike_counts}")
print(f"Output valid: {sim.outputs_valid}")
```

**Output:**

```
Timesteps:    10
Spike counts: {'lif1': 0, 'lif2': 0}
Output valid: True
```

With random (untrained) weights, the spike rate is zero. After training, LIF neurons would fire at rates determined by the learned weights and input patterns.

---

## 9. Energy Profiling

Profile energy consumption using device-specific presets.

```python
profile = cpu.profile(graph, n_steps=10, energy_preset="45nm_cmos")

print(f"Total latency:  {profile.total_latency_us:.1f} μs")
print(f"Total energy:   {profile.energy_estimate_uj:.4f} μJ")
print(f"  MAC energy:   {profile.mac_energy_uj:.4f} μJ")
print(f"  Spike energy: {profile.spike_energy_uj:.4f} μJ")
print(f"  SRAM energy:  {profile.sram_energy_uj:.4f} μJ")
print(f"Peak memory:    {profile.peak_memory_bytes:,} bytes")
print(f"Neuron util:    {profile.neuron_utilization}")
```

**Output:**

```
Total latency:  395.4 μs
Total energy:   5.6542 μJ
  MAC energy:   3.6255 μJ
  Spike energy: 0.0000 μJ
  SRAM energy:  2.0287 μJ
Peak memory:    1,633,600 bytes
Neuron util:    {'lif1': 0.0, 'lif2': 0.0}
```

Available energy presets: `"45nm_cmos"`, `"neuromorphic_int8"`, `"zynq_us_plus"`. Custom per-operation energy values can also be passed directly.

---

## 10. Visualization

Generate interactive HTML visualizations for the graph, partitions, and execution schedule.

```python
import tempfile, os

tmpdir = tempfile.mkdtemp()

graph_html = sdk.export_html(graph, os.path.join(tmpdir, "graph.html"))
print(f"Graph:      {os.path.getsize(graph_html):,} bytes")

part_html = sdk.visualize_partitioned(
    partitioned, os.path.join(tmpdir, "partitions.html")
)
print(f"Partitions: {os.path.getsize(part_html):,} bytes")

sched_html = sdk.visualize_execution_schedule(
    graph, os.path.join(tmpdir, "schedule.html")
)
print(f"Schedule:   {os.path.getsize(sched_html):,} bytes")
```

**Output:**

```
Graph:      2,245,446 bytes
Partitions: 2,246,651 bytes
Schedule:   2,852 bytes
```

The graph visualization shows node types with color-coded boxes (blue=conv, green=LIF, orange=affine) and directed edges. The partition visualization overlays core boundary regions. The schedule visualization shows the per-core execution timeline.

:::tip
Use `serve=True` with `talon.viz` to launch an interactive localhost viewer:
```python
from talon.viz import display
display.export_html(graph, "model.html", serve=True)
```
:::

---

## 11. Event Encoding

talon.io provides neural encoding to convert analog signals into spike trains for SNN input.

```python
import numpy as np
from talon.io.encoding import rate_encode, latency_encode, delta_encode

signal = np.random.rand(32).astype(np.float32)

spikes_rate = rate_encode(signal, n_steps=100)
print(f"Rate encoding:    {spikes_rate.shape} → {int(spikes_rate.sum())} spikes")

spikes_lat = latency_encode(signal, n_steps=100, tau=5.0)
print(f"Latency encoding: {spikes_lat.shape} → {int(spikes_lat.sum())} spikes")

sequence = np.random.rand(50, 32).astype(np.float32)
spikes_delta = delta_encode(sequence, threshold=0.1)
print(f"Delta encoding:   {spikes_delta.shape} → {int(np.abs(spikes_delta).sum())} spikes")
```

**Output:**

```
Rate encoding:    (100, 32) → 1785 spikes
Latency encoding: (100, 32) → 31 spikes
Delta encoding:   (49, 2, 32) → 1257 spikes
```

---

## 12. Event I/O (HDF5)

Store and retrieve event streams using the HDF5-based event format.

```python
from talon.io.h5 import H5EventWriter, H5EventReader, EVENT_DTYPE

events = np.zeros(1000, dtype=EVENT_DTYPE)
events['t'] = np.sort(np.random.randint(0, 100_000, size=1000))
events['x'] = np.random.randint(0, 128, size=1000)
events['y'] = np.random.randint(0, 128, size=1000)
events['p'] = np.random.randint(0, 2, size=1000)

event_path = os.path.join(tmpdir, "events.h5")
with H5EventWriter(event_path) as w:
    w.write(events)
print(f"Wrote {len(events)} events")

reader = H5EventReader(event_path)
loaded = reader.read_all()
print(f"Read back: {len(loaded)} events")

window = reader.read_time_window(10_000, 50_000)
print(f"Time window [10k, 50k]: {len(window)} events")
```

**Output:**

```
Wrote 1000 events
Read back: 1000 events
Time window [10k, 50k]: ~400 events
```

---

## 13. Throughput Benchmark

Measure the event processing pipeline throughput.

```python
from talon.io.throughput import benchmark_throughput

result = benchmark_throughput(n_events=100_000, include_encoding=True)
print(f"Events/sec:   {result.events_per_sec:,.0f}")
print(f"RGB FPS:      {result.rgb_fps:.1f}")
print(f"Event FPS:    {result.event_fps:.1f}")
print(f"Encoding FPS: {result.encoding_fps:.1f}")
```

**Output:**

```
Events/sec:   33,806,630,204
RGB FPS:      314.8
Event FPS:    773.4
Encoding FPS: 12.1
```

---

## 14. Full Pipeline (One Call)

The SDK provides `run_pipeline()` to execute the entire workflow in a single call.

```python
result = sdk.run_pipeline(
    path,
    sdk.PipelineConfig(
        target="cpu",
        timesteps=10,
        quantize=False,
        partition=False,
        lint=True,
        profile=True,
    )
)

print(f"Errors:     {result.errors}")
print(f"Lint:       {len(result.lint_results)} issues")
if result.analysis:
    print(f"Analysis:   {result.analysis.total_params:,} params, depth={result.analysis.depth}")
if result.profile:
    print(f"Profile:    latency={result.profile.total_latency_us:.1f}μs, energy={result.profile.energy_estimate_uj:.4f}μJ")
if result.simulation:
    print(f"Simulation: {result.simulation.timesteps_run} steps, spikes={result.simulation.spike_counts}")
```

**Output:**

```
Errors:     []
Lint:       2 issues
Analysis:   403,538 params, depth=9
Profile:    latency=395.4μs, energy=5.6542μJ
Simulation: 10 steps, spikes={'lif1': 0, 'lif2': 0}
```

---

## Pipeline Summary

| Step | Module | Function |
|------|--------|----------|
| Export | `snntorch.export_talonir` | `export_to_ir(model, sample)` |
| Serialize | `talon.ir` | `ir.write(path, graph)` / `ir.read(path)` |
| Analyze | `talon.sdk` | `analyze_graph(graph)` |
| Profile | `talon.sdk` | `profile_graph(graph)` |
| Lint | `talon.sdk` | `lint_graph(graph)` |
| Fingerprint | `talon.sdk` | `fingerprint_graph(graph)` |
| Quantize | `talon.sdk` | `quantize_weights(graph, bits=8)` |
| Partition | `talon.graph` | `partition(graph, hw_spec)` |
| Allocate | `talon.graph` | `allocate(graph, hw_spec)` |
| Place | `talon.graph` | `place(graph, hw_spec)` |
| Simulate | `talon.backend` | `cpu.simulate(graph, n_steps=N)` |
| Energy | `talon.backend` | `cpu.profile(graph, energy_preset=...)` |
| Visualize | `talon.sdk` | `export_html()`, `visualize_partitioned()`, `visualize_execution_schedule()` |
| Encode | `talon.io.encoding` | `rate_encode()`, `latency_encode()`, `delta_encode()` |
| Events | `talon.io.h5` | `H5EventWriter`, `H5EventReader` |
| Benchmark | `talon.io.throughput` | `benchmark_throughput()` |
| Pipeline | `talon.sdk` | `run_pipeline(path, config)` |

---

## Next Steps

- **[snnTorch Integration](./tutorial_snntorch_integration.md)** — Detailed snnTorch export/import with round-trip verification
- **[Ghost Detection](./tutorial_ghost_detection.md)** — GhostNet architecture and NMS detection pipeline
- **[Backend Simulation](./tutorial_backend.md)** — Detailed CPU/FPGA backend usage
- **[Hardware Mapping](./tutorial_hardware_mapping.md)** — Advanced partitioning and placement strategies
- **[Event I/O](./tutorial_event_io.md)** — Deep dive into event encoding and HDF5 storage
