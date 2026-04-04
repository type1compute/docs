---
sidebar_position: 17
---

# Tutorial: Backend — Simulate, Profile & FPGA

The talon.backend (`talon.backend`) provides CPU simulation, energy/latency profiling with device-specific presets, and FPGA compilation via hls4ml.

## What You'll Learn

1. Backend registry and capabilities
2. Validate a graph against a backend
3. Compile for hardware
4. Simulate on CPU with spike tracking
5. Profile energy, latency, and memory with device presets
6. FPGA compilation and bitstream generation

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Backend Registry

TALON provides pluggable backends. The built-in backends are `cpu` and `fpga`.

```python
from talon import backend

print(f"Available backends: {backend.list_backends()}")

cpu = backend.get_backend("cpu")
print(f"\nCPU backend: {cpu.name}")

caps = cpu.get_capabilities()
print(f"  Supported primitives: {len(caps.supported_primitives)}")
print(f"  Supported precisions: {caps.supported_precisions}")
print(f"  Supports cyclic: {caps.supports_cyclic}")
print(f"  Max graph size: {caps.max_graph_size:,}")
```

**Output:**

```
Available backends: ['cpu', 'fpga']

CPU backend: cpu
  Supported primitives: 22
  Supported precisions: ['fp32', 'fp16', 'int16', 'int8', 'int4']
  Supports cyclic: True
  Max graph size: 1,000,000
```

---

## 2. Energy Presets

TALON includes energy presets for different hardware targets. All energy values are in picojoules (pJ) per operation.

```python
print("Energy presets:")
for name, preset in backend.ENERGY_PRESETS.items():
    print(f"  {name}:")
    print(f"    MAC:  {preset['mac_pj']} pJ")
    print(f"    Spike: {preset['spike_pj']} pJ")
    print(f"    {preset['description']}")
```

**Output:**

```
Energy presets:
  45nm_cmos:
    MAC:  0.9 pJ
    Spike: 0.1 pJ
    Standard 45 nm CMOS process (Horowitz 2014)
  28nm_cmos:
    MAC:  0.5 pJ
    Spike: 0.06 pJ
    28 nm CMOS process
  neuromorphic_int8:
    MAC:  4.6 pJ
    Spike: 0.9 pJ
    Representative INT8 neuromorphic accelerator
  zynq_us_plus:
    MAC:  3.2 pJ
    Spike: 0.7 pJ
    Zynq UltraScale+ XCZU9EG on ZCU102 (16 nm FinFET, 274K LUTs, 2520 DSP48, 32.1 Mb BRAM)
  zcu102:
    MAC:  3.2 pJ
    Spike: 0.7 pJ
    Alias for zynq_us_plus (XCZU9EG-2FFVB1156 on ZCU102)
  zynq_7020:
    MAC:  6.5 pJ
    Spike: 1.2 pJ
    Xilinx Zynq-7020 FPGA (28 nm)
```

---

## 3. Build a Test Graph

```python
import numpy as np
from talon import ir

nodes = {
    "input": ir.Input(np.array([784])),
    "fc1": ir.Affine(
        weight=np.random.randn(128, 784).astype(np.float32) * 0.01,
        bias=np.zeros(128, dtype=np.float32),
    ),
    "lif1": ir.LIF(
        tau=np.ones(128, dtype=np.float32) * 10.0,
        r=np.ones(128, dtype=np.float32),
        v_leak=np.zeros(128, dtype=np.float32),
        v_threshold=np.ones(128, dtype=np.float32),
    ),
    "fc2": ir.Affine(
        weight=np.random.randn(10, 128).astype(np.float32) * 0.01,
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
    ("input", "fc1"), ("fc1", "lif1"), ("lif1", "fc2"),
    ("fc2", "lif2"), ("lif2", "output"),
]

graph = ir.Graph(nodes=nodes, edges=edges)
print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
```

**Output:**

```
Graph: 6 nodes, 5 edges
```

---

## 4. Validate

Check whether all nodes in the graph are supported by the backend.

```python
val = cpu.validate(graph)

print(f"Validation:")
print(f"  Valid: {val.is_valid}")
print(f"  Unsupported nodes: {val.unsupported_nodes}")
print(f"  Warnings: {val.warnings}")
```

**Output:**

```
Validation:
  Valid: True
  Unsupported nodes: []
  Warnings: []
```

---

## 5. Compile

Compile produces a `HardwareDescriptor` containing the execution plan.

```python
config = backend.CompileConfig()
compiled = cpu.compile(graph, config)
print(f"Compiled: {type(compiled).__name__}")
```

**Output:**

```
Compiled: HardwareDescriptor
```

---

## 6. Simulate

Run the graph on CPU with spike state tracking across timesteps.

```python
x = np.random.randn(1, 784).astype(np.float32)
sim = cpu.simulate(graph, {"input": x}, n_steps=3)

print(f"Simulation:")
print(f"  Steps run: {sim.timesteps_run}")
print(f"  Outputs valid: {sim.outputs_valid}")
print(f"  Spike counts: {sim.spike_counts}")
```

**Output:**

```
Simulation:
  Steps run: 3
  Outputs valid: True
  Spike counts: {'lif1': 15, 'lif2': 0}
```

:::tip Step-by-Step Mode
Use `step_by_step=True` to get per-timestep intermediate states in `sim.node_states`.
:::

---

## 7. Profile

Profile computes estimated latency, energy, and memory usage. All energy values are labeled as **estimated** — they use analytical models, not measured hardware power.

### Default Preset (neuromorphic_int8)

```python
prof = cpu.profile(graph)

print(f"Profile (neuromorphic_int8):")
print(f"  Total latency: {prof.total_latency_us:.2f} μs")
print(f"  MAC operations: {prof.mac_ops:,}")
print(f"  Energy estimate: {prof.energy_estimate_uj:.4f} μJ")
print(f"    MAC energy: {prof.mac_energy_uj:.4f} μJ")
print(f"    Spike energy: {prof.spike_energy_uj:.4f} μJ")
print(f"    SRAM energy: {prof.sram_energy_uj:.4f} μJ")
print(f"  Peak memory: {prof.peak_memory_bytes:,} bytes")
print(f"  Weight memory: {prof.weight_memory_bytes:,} bytes")
print(f"  Energy preset: {prof.energy_preset}")

print(f"\n  Per-layer energy:")
for layer, energy in prof.per_layer_energy_uj.items():
    print(f"    {layer}: {energy:.4f} μJ")
```

**Output:**

```
Profile (neuromorphic_int8):
  Total latency: 54.58 μs
  MAC operations: 101,632
  Energy estimate: 0.9777 μJ
    MAC energy: 0.4675 μJ
    Spike energy: 0.0000 μJ
    SRAM energy: 0.5102 μJ
  Peak memory: 413,568 bytes
  Weight memory: 407,080 bytes
  Energy preset: neuromorphic_int8

  Per-layer energy:
    fc1: 0.9640 μJ
    lif1: 0.0013 μJ
    fc2: 0.0123 μJ
    lif2: 0.0001 μJ
```

### ZCU102 Preset

```python
prof_zcu = cpu.profile(graph, energy_preset="zcu102")

print(f"Profile (ZCU102):")
print(f"  Energy preset: {prof_zcu.energy_preset}")
print(f"  Energy estimate: {prof_zcu.energy_estimate_uj:.4f} μJ")
```

**Output:**

```
Profile (ZCU102):
  Energy preset: zcu102
  Energy estimate: 0.7334 μJ
```

### Custom Energy Parameters

```python
prof_custom = cpu.profile(
    graph,
    energy_per_mac_pj=2.0,
    energy_per_spike_pj=0.5,
    energy_per_sram_read_pj=3.0,
)
print(f"Custom preset energy: {prof_custom.energy_estimate_uj:.4f} μJ")
```

---

## 8. FPGA Backend

The FPGA backend generates hls4ml-compatible configurations for Xilinx FPGAs.

### Validate for FPGA

```python
fpga = backend.get_backend("fpga")
val = fpga.validate(graph)
print(f"FPGA validation: {val.is_valid}")
print(f"  Supported precisions: {fpga.get_capabilities().supported_precisions}")
```

**Output:**

```
FPGA validation: True
  Supported precisions: ['fp32', 'int16', 'int8', 'int4', 'ap_fixed<16,6>']
```

### Compile for FPGA

```python
descriptor = fpga.compile(
    graph,
    output_dir="hls4ml_project",
    project_name="snn_demo",
    part="xczu9eg-ffvb1156-2-i",
    clock_period=5,
    default_precision="ap_fixed<16,6>",
    default_reuse_factor=1,
    strategy="Latency",
)
print(f"FPGA compiled: {type(descriptor).__name__}")
```

**Output:**

```
FPGA compiled: HardwareDescriptor
```

### Build Bitstream Project

```python
result = fpga.build_bitstream(
    graph,
    output_dir="./hls4ml_project",
    project_name="snn_demo",
    part="xczu9eg-ffvb1156-2-i",
    clock_period=5,
    default_precision="ap_fixed<16,6>",
    default_reuse_factor=1,
    strategy="Latency",
    bitstream_config=backend.BitstreamConfig(
        csim=True,
        synth=False,
        cosim=False,
        export=False,
        vsynth=False,
        bitfile=False,
    ),
    write_to_disk=True,
)

print(f"Generated files: {list(result.keys())}")
```

:::tip Two-Phase FPGA Flow
1. **SDK generates** configs, templates (LIF/IF headers, `parameters.h`, `build.py`, `build_prj.tcl`)
2. **hls4ml creates** the actual HLS project from those configs

`hls4ml` is an optional dependency — install it separately when targeting FPGA synthesis.
:::

---

## API Reference

| Function / Class | Purpose |
|------------------|---------|
| `backend.list_backends()` | List available backends |
| `backend.get_backend(name)` | Get a backend instance |
| `backend.ENERGY_PRESETS` | Energy preset configurations |
| `cpu.validate(graph)` | Check graph compatibility |
| `cpu.compile(graph, config)` | Compile to hardware descriptor |
| `cpu.simulate(graph, inputs, n_steps)` | Run CPU simulation |
| `cpu.profile(graph, energy_preset)` | Profile energy, latency, memory |
| `fpga.validate(graph)` | Check FPGA compatibility |
| `fpga.compile(graph, ...)` | Compile for FPGA |
| `fpga.build_bitstream(graph, ...)` | Generate hls4ml project files |
| `BitstreamConfig(...)` | Configure bitstream generation steps |

## What's Next

- [Tutorial: Event I/O](./tutorial_event_io) — Neuromorphic event data handling
- [Tutorial: Hardware Mapping](./tutorial_hardware_mapping) — Partition, route, and place
- [SDK Reference](../talon) — Full SDK API documentation
