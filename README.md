# NeuroForge Multimodal Prototype

An experimental PyTorch architecture for combining 3D radiology volumes with pathology image patches. The repository currently provides model components and a single training-step helper; it does not yet include a dataset loader, complete training entry point, evaluation pipeline, or trained weights.

## Architecture

| Component | Expected input | Output |
| --- | --- | --- |
| `RadiologyEncoder` | MRI tensor shaped `(batch, 4, depth, height, width)` | 128-dimensional feature vector per sample |
| `PathologyEncoder` | Patch tensor shaped `(batch, patches, 3, 224, 224)` | Attention-weighted 128-dimensional feature vector per sample |
| `NeuroFusionModel` | One radiology tensor and one pathology tensor | Configurable classification logits; default is three classes |
| `train_step` | Prepared batches, labels, optimizer, and loss | One optimization step and scalar loss |

The pathology branch uses an ImageNet-pretrained ResNet-18 feature extractor and gated attention. The radiology branch uses 3D convolutions and adaptive pooling before the two feature vectors are fused.

## Repository structure

```text
.
├── app/models/
│   ├── Radiology.py
│   ├── Pathology.py
│   ├── NeuroFusion.py
│   └── Train.py
├── requirements.txt
└── README.md
```

## Environment

Create an isolated Python environment and install the declared dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

The requirements use minimum versions rather than a reproducible lock. Instantiating the pathology encoder may download pretrained ResNet weights.

## Current limitations

- No dataset-loading or preprocessing entry point is included.
- No end-to-end training command, checkpoints, metrics, or benchmark results are provided.
- Medical dataset access, licensing, consent, label definitions, and evaluation methodology are not documented.
- The code is a research prototype and must not be treated as a clinical or diagnostic system.

Static source checks were performed during the repository refresh. Full model execution and medical-data validation were not performed.
