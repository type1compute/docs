---
sidebar_position: 2
---

# Quick Start

This page walks through a minimal forward pass, a training command, and instance-level inference.

## Minimal Forward Pass

```python
import torch
from spikeseg.models import SpikeSEG

# Instantiate from the IGARSS 2023 paper configuration
model = SpikeSEG.from_paper("igarss2023", n_classes=1)

# Simulate an event frame: (batch, channels, height, width)
input_events = torch.randn(1, 1, 128, 128)

model.reset_state()
for t in range(10):
    saliency, encoder_output = model(input_events)
    if encoder_output.has_spikes:
        print(f"t={t}: {encoder_output.n_classification_spikes} classification spikes")
```

`saliency` is a pixel-level map; `encoder_output` holds per-layer spikes, pooling indices, and membrane potentials.

## Train with One Command

```bash
python scripts/train.py --config configs/config.yaml
```

Training is layer-wise: Conv1 uses fixed DoG filters, Conv2 and Conv3 are trained via STDP with WTA competition. The script saves checkpoints, TensorBoard logs, and a final metrics JSON. See [Training Guide](../guides/training) for details.

## Instance Segmentation with HULK-SMASH

```python
import torch
from spikeseg.models import SpikeSEG
from spikeseg.algorithms import HULKDecoder, group_instances_to_objects

model = SpikeSEG.from_paper("igarss2023", n_classes=1)
model.load_state_dict(torch.load("checkpoint.pth"))

hulk = HULKDecoder.from_encoder(model.encoder)

model.reset_state()
encoder_output = model.encode(input_events)

instances = []
for spike_loc in encoder_output.get_spike_locations():
    inst = hulk.unravel_spike(
        spike_location=spike_loc,
        pool1_indices=encoder_output.pooling_indices.pool1_indices,
        pool2_indices=encoder_output.pooling_indices.pool2_indices,
        pool1_output_size=encoder_output.pooling_indices.pool1_output_size,
        pool2_output_size=encoder_output.pooling_indices.pool2_output_size,
    )
    instances.append(inst)

objects = group_instances_to_objects(instances, smash_threshold=0.1)
print(f"Detected {len(objects)} objects from {len(instances)} instances")
```

## Evaluate a Checkpoint

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt \
  --data-root /path/to/EBSSA \
  --volume-based \
  --output metrics.json
```

This runs volume-based evaluation (IGARSS 2023 methodology) and writes all metrics to `metrics.json`. See [Evaluation Guide](../guides/evaluation).

## What Next

| Goal | Page |
|------|------|
| Understand the theory | [Spiking Neurons](../theory/spiking-neurons) |
| See the full architecture | [Architecture Overview](../architecture/overview) |
| Configure training | [Configuration](../guides/configuration) |
| Browse the API | [API Reference](../api/overview) |
