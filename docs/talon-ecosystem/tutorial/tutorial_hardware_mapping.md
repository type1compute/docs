---
sidebar_position: 16
---

# Tutorial: Hardware Mapping

The talon.graph library (`talon.graph`) maps TALON IR graphs onto multi-core neuromorphic hardware. This tutorial covers defining hardware constraints, partitioning graphs across cores, allocating resources, routing inter-core communication, and placing cores on a physical mesh.

## What You'll Learn

1. Define hardware constraints with `HardwareSpec`
2. Partition a graph across cores
3. Allocate resources (SRAM, neurons, synapses)
4. Route inter-core communication
5. Place cores on a physical mesh
6. Use device presets (Zynq-7020, ZCU102)

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. HardwareSpec

`HardwareSpec` defines the constraints of the target hardware — core capacity, memory, fan-in/fan-out limits, and mesh topology.

```python
import numpy as np
from talon import ir, graph as t1cgraph

spec = t1cgraph.HardwareSpec(
    max_neurons_per_core=1024,
    max_synapses_per_core=200000,
    sram_bytes_per_core=524288,
    num_cores=4,
    max_feedback_delay=3,
)

print(f"HardwareSpec:")
print(f"  max_neurons_per_core: {spec.max_neurons_per_core}")
print(f"  max_synapses_per_core: {spec.max_synapses_per_core:,}")
print(f"  sram_bytes_per_core: {spec.sram_bytes_per_core:,}")
print(f"  num_cores: {spec.num_cores}")
print(f"  max_feedback_delay: {spec.max_feedback_delay}")
```

**Output:**

```
HardwareSpec:
  max_neurons_per_core: 1024
  max_synapses_per_core: 200,000
  sram_bytes_per_core: 524,288
  num_cores: 4
  max_feedback_delay: 3
```

### Device Presets

TALON includes presets for common FPGA targets:

```python
zynq = t1cgraph.HardwareSpec.zynq_7020()
print(f"Zynq-7020: cores={zynq.num_cores}, sram={zynq.sram_bytes_per_core:,}")

zcu = t1cgraph.HardwareSpec.zcu102()
print(f"ZCU102:    cores={zcu.num_cores}, sram={zcu.sram_bytes_per_core:,}")
```

**Output:**

```
Zynq-7020: cores=4, sram=131,072
ZCU102:    cores=8, sram=524,288
```

### Validation

```python
spec.validate()
print("HardwareSpec validated successfully")
```

**Output:**

```
HardwareSpec validated successfully
```

---

## 2. Build a Test Graph

```python
nodes = {
    "input": ir.Input(np.array([784])),
    "fc1": ir.Affine(
        weight=np.random.randn(64, 784).astype(np.float32) * 0.01,
        bias=np.zeros(64, dtype=np.float32),
    ),
    "lif1": ir.LIF(
        tau=np.ones(64, dtype=np.float32) * 10.0,
        r=np.ones(64, dtype=np.float32),
        v_leak=np.zeros(64, dtype=np.float32),
        v_threshold=np.ones(64, dtype=np.float32),
    ),
    "fc2": ir.Affine(
        weight=np.random.randn(10, 64).astype(np.float32) * 0.01,
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

## 3. Partition

Partition assigns each node to a core while respecting SRAM, neuron, and synapse constraints. The function returns a new `Graph` with `partition_metadata` attached.

```python
partitioned = t1cgraph.partition(graph, spec)
pm = partitioned.partition_metadata

print(f"Algorithm: {pm['algorithm']}")
print(f"Cores used: {pm['num_cores_used']}")
print(f"\nAssignments:")
for name, core in pm['assignments'].items():
    print(f"  {name} -> core {core}")
print(f"\nCross-core edges: {pm['cross_core_edges']}")
print(f"Metrics:")
print(f"  Cut ratio: {pm['metrics']['cut_ratio']:.2f}")
print(f"  Balance stddev: {pm['metrics']['balance_stddev']:.2f}")
```

**Output:**

```
Algorithm: greedy
Cores used: 1

Assignments:
  input -> core 0
  fc1 -> core 0
  lif1 -> core 0
  fc2 -> core 0
  lif2 -> core 0
  output -> core 0

Cross-core edges: []
Metrics:
  Cut ratio: 0.00
  Balance stddev: 0.00
```

:::tip Partition Algorithms
Use `algorithm="greedy"` (default) for fast partitioning, or `algorithm="spectral"` for graph-cut optimization on larger models.
:::

---

## 4. Resource Allocation

After partitioning, allocate checks whether each core's assigned nodes fit within the hardware budget.

```python
res_map = t1cgraph.allocate(partitioned, spec)

print(f"Resource Allocation:")
print(f"  Total weight bytes: {res_map.total_weight_bytes:,}")
print(f"  Total state bytes: {res_map.total_state_bytes:,}")
print(f"  Peak utilization: {res_map.peak_core_utilization:.2%}")
print(f"  Fits hardware: {res_map.fits_hardware}")

if res_map.violations:
    print(f"  Violations (cores): {res_map.violations}")

for core_id, budget in res_map.budgets.items():
    print(f"\n  Core {core_id}:")
    print(f"    Neurons: {budget.neuron_count}")
    print(f"    Weight bytes: {budget.weight_bytes:,}")
    print(f"    Utilization: {budget.utilization:.2%}")
```

**Output:**

```
Resource Allocation:
  Total weight bytes: 203,560
  Total state bytes: 1,184
  Peak utilization: 39.05%
  Fits hardware: True

  Core 0:
    Neurons: 4
    Weight bytes: 203,560
    Utilization: 39.05%
```

---

## 5. Routing

Route computes inter-core communication paths for cross-core edges. For single-core graphs, no routing is needed.

```python
routed = t1cgraph.route(partitioned, spec)
print(f"Routed graph: {len(routed.nodes)} nodes, {len(routed.edges)} edges")
```

**Output:**

```
Routed graph: 6 nodes, 5 edges
```

---

## 6. Placement

Place maps logical cores onto the physical mesh to minimize total hop distance (Manhattan distance weighted by edge count).

```python
placement = t1cgraph.place(partitioned, spec)

print(f"Placement:")
print(f"  Total hop distance: {placement.total_hop_distance}")
print(f"  Improvement over identity: {placement.improvement:.1f}%")

for core_id, pos in placement.logical_to_physical.items():
    print(f"  Core {core_id} -> row={pos[0]}, col={pos[1]}")
```

**Output:**

```
Placement:
  Total hop distance: 0
  Improvement over identity: 0.0%
  Core 0 -> row=0, col=0
  Core 1 -> row=1, col=0
  Core 2 -> row=2, col=0
  Core 3 -> row=3, col=0
```

---

## 7. End-to-End Pipeline

The full hardware mapping pipeline:

```python
spec = t1cgraph.HardwareSpec.zcu102()

# 1. Partition
partitioned = t1cgraph.partition(graph, spec)
pm = partitioned.partition_metadata
print(f"1. Partitioned: {pm['num_cores_used']} cores, algo={pm['algorithm']}")

# 2. Allocate
res = t1cgraph.allocate(partitioned, spec)
print(f"2. Allocated: fits={res.fits_hardware}, peak_util={res.peak_core_utilization:.1%}")

# 3. Route
routed = t1cgraph.route(partitioned, spec)
print(f"3. Routed: {len(routed.edges)} edges")

# 4. Place
placed = t1cgraph.place(partitioned, spec)
print(f"4. Placed: total_hops={placed.total_hop_distance}, improvement={placed.improvement:.1f}%")
```

**Output:**

```
1. Partitioned: 1 cores, algo=greedy
2. Allocated: fits=True, peak_util=39.0%
3. Routed: 5 edges
4. Placed: total_hops=0, improvement=0.0%
```

---

## API Reference

| Function / Class | Purpose |
|------------------|---------|
| `HardwareSpec(...)` | Define hardware constraints |
| `HardwareSpec.zynq_7020()` | Zynq-7020 preset |
| `HardwareSpec.zcu102()` | ZCU102 (Zynq UltraScale+) preset |
| `HardwareSpec.t1c_asic_target()` | TALON ASIC target preset |
| `partition(graph, spec)` | Assign nodes to cores |
| `allocate(graph, spec)` | Check resource budgets per core |
| `route(graph, spec)` | Compute inter-core routing paths |
| `place(graph, spec)` | Map cores to physical mesh positions |

## What's Next

- [Tutorial: Backend](./tutorial_backend) — Simulate, profile, and compile for FPGA
- [Tutorial: Event I/O](./tutorial_event_io) — Neuromorphic event data handling
- [Architecture](../architecture) — Ecosystem architecture overview
