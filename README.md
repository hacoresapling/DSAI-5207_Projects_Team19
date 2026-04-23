# YOLO Attention Distillation

Knowledge distillation project that transfers attention patterns from DINOv2 teacher to YOLOv8 student for object detection improvement.

## Project Structure

```
.
├── configs/              # Training configurations
│   ├── baseline.yaml     # E1: Standard YOLOv8m baseline
│   ├── relation_distill.yaml  # E2: Attention relation distillation
│   └── feature_distill_d2.yaml  # E3: Feature distillation (D2)
├── models/
│   ├── teacher.py         # DINOv2 teacher wrapper
│   ├── relation_constructor.py  # Student relation constructor
│   └── d2_head.py         # D2 projection head
├── losses/
│   ├── relation_loss.py   # Relation distillation loss (KL divergence)
│   └── feature_loss.py    # Feature distillation loss (SmoothL1)
├── trainers/
│   ├── base_distill_trainer.py  # Base trainer for distillation
│   ├── relation_trainer.py     # E2 trainer
│   └── d2_trainer.py          # E3 trainer
├── evaluation/
│   ├── compare.py         # mAP comparison across experiments
│   ├── gradcam.py        # GradCAM attention visualization
│   └── relation_vis.py   # Relation matrix visualization
├── scripts/
│   ├── download_weights.py  # Download model weights
│   └── setup_dinov2.sh   # Setup DINOv2 source
├── train.py              # Unified training entry point
└── evaluate.py           # Unified evaluation entry point
```

## Experiments

| Experiment | Description |
|------------|-------------|
| **E1 Baseline** | Standard YOLOv8m training (baseline) |
| **E2 Ours** | Attention relation distillation from DINOv2 |
| **E3 d2** | Feature distillation (DINOv2 to YOLO) |

## Setup

```bash
pip install -r requirements.txt

# Download model weights
python scripts/download_weights.py --output weights/

# Or setup DINOv2 from source
bash scripts/setup_dinov2.sh
```

## Training

```bash
# E1: Baseline
python train.py --config configs/baseline.yaml

# E2: Relation Distillation
python train.py --config configs/relation_distill.yaml

# E3: Feature Distillation (D2)
python train.py --config configs/feature_distill_d2.yaml

# Resume training
python train.py --config configs/relation_distill.yaml --resume
```

## Evaluation

```bash
# Compare mAP across experiments
python evaluate.py --mode compare

# GradCAM visualization
python evaluate.py --mode gradcam --images img1.jpg img2.jpg

# Relation matrix visualization
python evaluate.py --mode relation --images img1.jpg --teacher weights/dinov2_vitl14_reg4_pretrain.pth
```

## Dataset

COCO 2017 (80 classes), configured in `configs/data.yaml`:
- Train: `images/train2017`
- Val: `images/val2017`

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- ultralytics >= 8.3.0
- CUDA-capable GPU recommended