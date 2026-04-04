---
sidebar_position: 18
---

# Tutorial: Event I/O — Encoding, Formats & Throughput

The talon.io library (`talon.io`) provides neural spike encoding, event-stream file I/O (HDF5), and throughput benchmarking for neuromorphic event data.

## What You'll Learn

1. Neural encoding: rate, latency, and delta
2. Event data format and structured arrays
3. HDF5 event I/O with `H5EventWriter` and `H5EventReader`
4. Time-windowed event queries
5. Throughput benchmarking

## Prerequisites

```bash
pip install t1c-talon
```

---

## 1. Neural Encoding

talon.io provides three encoding schemes to convert continuous data into spike trains.

### Rate Encoding

Each value is treated as a firing probability. Higher values produce more spikes.

```python
import numpy as np
from talon.io import encoding

data = np.random.rand(1, 28, 28).astype(np.float32)
spikes = encoding.rate_encode(data, n_steps=10)

print(f"Rate encoding:")
print(f"  Input: {data.shape}")
print(f"  Output: {spikes.shape}")
print(f"  Spike rate: {spikes.mean():.3f}")
```

**Output:**

```
Rate encoding:
  Input: (1, 28, 28)
  Output: (10, 1, 28, 28)
  Spike rate: 0.488
```

### Latency Encoding

Stronger inputs fire earlier. The `tau` parameter controls the time constant.

```python
spikes_lat = encoding.latency_encode(data, n_steps=10, tau=5.0)

print(f"Latency encoding:")
print(f"  Output: {spikes_lat.shape}")
print(f"  Spike rate: {spikes_lat.mean():.3f}")
```

**Output:**

```
Latency encoding:
  Output: (10, 1, 28, 28)
  Spike rate: 0.099
```

### Delta Encoding

Converts a sequence of frames into ON/OFF events based on temporal differences exceeding a threshold.

```python
frames = np.random.rand(10, 1, 28, 28).astype(np.float32)
events = encoding.delta_encode(frames, threshold=0.1)

print(f"Delta encoding:")
print(f"  Input: {frames.shape}")
print(f"  Output: {events.shape}")
print(f"  Channels: 2 (ON / OFF)")
```

**Output:**

```
Delta encoding:
  Input: (10, 1, 28, 28)
  Output: (9, 2, 1, 28, 28)
  Channels: 2 (ON / OFF)
```

---

## 2. Event Data Format

TALON uses NumPy structured arrays with the standard event camera format:

| Field | Type | Description |
|-------|------|-------------|
| `t` | `float64` | Timestamp in microseconds |
| `x` | `int32` | Horizontal pixel coordinate |
| `y` | `int32` | Vertical pixel coordinate |
| `p` | `int32` | Polarity (-1 for OFF, +1 for ON) |

```python
from talon.io.h5 import EVENT_DTYPE

print(f"Event dtype: {EVENT_DTYPE}")
```

**Output:**

```
Event dtype: [('t', '<f8'), ('x', '<i4'), ('y', '<i4'), ('p', '<i4')]
```

### Generate Synthetic Events

```python
n_events = 5000
sensor_size = (28, 28)

events = np.zeros(n_events, dtype=EVENT_DTYPE)
events['x'] = np.random.randint(0, sensor_size[0], n_events)
events['y'] = np.random.randint(0, sensor_size[1], n_events)
events['t'] = np.sort(np.random.uniform(0, 1e6, n_events))
events['p'] = np.random.choice([-1, 1], n_events)

print(f"Generated {n_events} events")
print(f"Time range: {events['t'].min():.0f} - {events['t'].max():.0f} μs")
print(f"Fields: {events.dtype.names}")
```

**Output:**

```
Generated 5000 events
Time range: 180 - 999800 μs
Fields: ('t', 'x', 'y', 'p')
```

---

## 3. HDF5 Event I/O

### Writing Events

```python
from talon.io.h5 import H5EventWriter, H5EventReader

with H5EventWriter("events.h5", sensor_size=(28, 28)) as writer:
    writer.write(events)

print(f"Wrote {n_events} events to events.h5")
```

**Output:**

```
Wrote 5000 events to events.h5
```

### Reading Events

```python
reader = H5EventReader("events.h5")
loaded = reader.read_all()

print(f"Read back: {len(loaded)} events")
print(f"Fields: {loaded.dtype.names}")
```

**Output:**

```
Read back: 5000 events
Fields: ('t', 'x', 'y', 'p')
```

---

## 4. Time-Windowed Queries

Read only events within a specific time range without loading the entire file into memory.

```python
window = reader.read_time_window(100000, 500000)
print(f"Events in [100000, 500000] μs: {len(window)}")
```

**Output:**

```
Events in [100000, 500000] μs: 1979
```

---

## 5. Throughput Benchmarking

Measure ingestion and conversion throughput for the event pipeline.

```python
from talon.io.throughput import benchmark_throughput

result = benchmark_throughput(n_events=100000, sensor_size=(28, 28))

print(f"Throughput Benchmark:")
print(f"  Events/sec: {result.events_per_sec:,.0f}")
print(f"  RGB FPS: {result.rgb_fps:,.1f}")
print(f"  Event FPS: {result.event_fps:,.1f}")
print(f"  Total events: {result.total_events:,}")
print(f"  Duration: {result.duration_sec:.4f} sec")
print(f"  Sensor size: {result.sensor_size}")
```

**Output:**

```
Throughput Benchmark:
  Events/sec: 25,806,451,962
  RGB FPS: 518.6
  Event FPS: 864.6
  Total events: 100,000
  Duration: 0.0100 sec
  Sensor size: (28, 28)
```

---

## 6. Encoding Signatures

All encoding functions accept and return NumPy arrays:

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `rate_encode(data, n_steps)` | `(*, H, W)` | `(n_steps, *, H, W)` | Probabilistic spike generation |
| `latency_encode(data, n_steps, tau)` | `(*, H, W)` | `(n_steps, *, H, W)` | Time-to-first-spike encoding |
| `delta_encode(frames, threshold)` | `(T, *, H, W)` | `(T-1, 2, *, H, W)` | Temporal difference ON/OFF events |

---

## API Reference

| Function / Class | Purpose |
|------------------|---------|
| `encoding.rate_encode()` | Rate-coded spike generation |
| `encoding.latency_encode()` | Latency-coded spike generation |
| `encoding.delta_encode()` | Delta/change detection encoding |
| `H5EventWriter(path, sensor_size)` | Write events to HDF5 |
| `H5EventReader(path)` | Read events from HDF5 |
| `reader.read_all()` | Load all events |
| `reader.read_time_window(t0, t1)` | Load events in time range |
| `benchmark_throughput(n_events, ...)` | Measure pipeline throughput |
| `EVENT_DTYPE` | Standard event structured array dtype |

## What's Next

- [Tutorial: T1CViz](./tutorial_t1cviz) — Visualize events as frames, rasters, and animated HTML
- [Tutorial: Backend](./tutorial_backend) — Simulate and profile with the CPU/FPGA backend
- [Architecture](../architecture) — Ecosystem architecture overview
