# Emergent Global Receptive Fields in CNNs via Knowledge Distillation from Vision Transformers

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Transferring DINOv2's Global Perception to YOLOv8 via Knowledge Distillation**

 [🔗 Code](https://github.com/hacoresapling/DSAI-5207_Projects_Team19) •
 [📊 presentation](https://docs.google.com/videos/d/1k20cDaTBJY5HpcqQXgU9dSA7u-XXQLzsrh3AJfP_ps0/edit?usp=drive_link)

</div>

---

## 🎯 Overview

Vision Transformers like DINOv2 achieve superior **global receptive fields** through Multi-Head Self-Attention, capturing long-range spatial relationships across entire images. However, their computational demands (307M parameters, quadratic complexity) make them unsuitable for real-time deployment. Meanwhile, YOLO-family detectors excel at inference speed but are fundamentally limited by the **local receptive fields** of convolutional architectures.

**Can we transfer DINOv2's global perceptual ability to YOLOv8 without sacrificing speed?**

This project investigates whether YOLOv8m can acquire DINOv2-Large's global context understanding through knowledge distillation, without any modification to its inference-time architecture. We explore two complementary distillation strategies:

- **E2 Relation Distillation** : Aligns cosine-similarity spatial relation matrices between teacher and student
- **E3 Feature Distillation** : Directly aligns DINOv2's final-layer patch features with YOLO's neck outputs

### Key Motivation

<div align="center">

| Architecture      | Global Context | Speed (FPS) | Deployment       |
| ----------------- | -------------- | ----------- | ---------------- |
| **DINOv2-Large**  | ✅ Excellent    | ❌ ~15       | ❌ Cloud only     |
| **YOLOv8m**       | ❌ Limited      | ✅ ~267      | ✅ Edge-ready     |
| **Our E3 (Ours)** | ⚡ **Improved** | ✅ **~265**  | ✅ **Edge-ready** |

</div>

---

## 📁 Project Structure

```
.
├── configs/                      # Training configurations
│   ├── baseline.yaml             # E1: Standard YOLOv8m baseline
│   ├── relation_distill.yaml     # E2: Attention relation distillation
│   └── feature_distill_d2.yaml   # E3: Feature distillation (D2)
├── models/
│   ├── teacher.py                # DINOv2 teacher wrapper
│   ├── relation_constructor.py   # Student relation constructor
│   └── d2_head.py                # D2 projection head
├── losses/
│   ├── relation_loss.py          # Relation distillation loss (KL divergence)
│   └── feature_loss.py           # Feature distillation loss (SmoothL1)
├── trainers/
│   ├── base_distill_trainer.py   # Base trainer for distillation
│   ├── relation_trainer.py       # E2 trainer
│   └── d2_trainer.py             # E3 trainer
├── evaluation/
│   ├── compare.py                # mAP comparison across experiments
│   ├── gradcam.py                # GradCAM attention visualization
│   └── relation_vis.py           # Relation matrix visualization
├── scripts/
│   ├── download_weights.py       # Download model weights
│   └── setup_dinov2.sh           # Setup DINOv2 source
├── train.py                      # Unified training entry point
└── evaluate.py                   # Unified evaluation entry point
```

---

## 📊 Experimental Results

### Standard Detection Performance

<div align="center">


| Experiment      | Description           | mAP@50     | mAP@50-95  | Precision  | Recall | FPS   |
| --------------- | --------------------- | ---------- | ---------- | ---------- | ------ | ----- |
| **E1 Baseline** | Standard YOLOv8m      | 0.5927     | 0.4301     | 0.6579     | 0.5441 | 267.5 |
| **E2 Relation** | Relation distillation | 0.5744     | 0.4133     | 0.6584     | 0.5372 | 264.1 |
| **E3 Feature**  | Feature distillation  | **0.5808** | **0.4200** | **0.6669** | 0.5333 | 265.6 |

</div>


---

## 🧪 Experiment Pipeline


### 🔧 Setup

**1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**Requirements:**

- Python >= 3.8
- PyTorch >= 2.0.0
- ultralytics >= 8.3.0
- CUDA-capable GPU recommended

**2. Download Model Weights**

```bash
# Option 1: Download pre-trained weights
python scripts/download_weights.py --output weights/

# Option 2: Setup DINOv2 from source
bash scripts/setup_dinov2.sh
```

**3. Dataset Configuration**

COCO 2017 (80 classes), configured in `configs/data.yaml`:

- **Train**: `images/train2017` (118K images)
- **Val**: `images/val2017` (5K images)

---

### 🚀 Training

**E1: Baseline YOLOv8m**

```bash
python train.py --config configs/baseline.yaml
```

**E2: Relation Distillation**

```bash
python train.py --config configs/relation_distill.yaml
```

**E3: Feature Distillation (Recommended)** ⭐

```bash
python train.py --config configs/feature_distill_d2.yaml
```

**Resume Training**

```bash
python train.py --config configs/relation_distill.yaml --resume
```

**Training Hyperparameters:**

- Optimizer: SGD
- Learning rate: 0.01 (cosine schedule)
- Momentum: 0.937
- Weight decay: 0.0005
- Batch size: 32
- Epochs: 100
- Warmup: 3 epochs
- Input size: 640×640 (YOLO), 630×630 (DINOv2)

---

### 📈 Evaluation

**Compare mAP across all experiments**

```bash
python evaluate.py --mode compare
```

**Generate GradCAM visualizations**

```bash
python evaluate.py --mode gradcam --images img1.jpg img2.jpg
```

**Visualize relation matrices**

```bash
python evaluate.py --mode relation \
  --images img1.jpg \
  --teacher weights/dinov2_vitl14_reg4_pretrain.pth
```

**Occlusion Robustness Testing**

```bash
python evaluate.py --mode occlusion \
  --images img1.jpg img2.jpg \
  --types grid_25 grid_50 center_30 random_patches
```

</div>

---

## 🎓 Methodology Comparison

<div align="center">


| Dimension               | E2 Relation Distillation           | E3 Feature Distillation           |
| ----------------------- | ---------------------------------- | --------------------------------- |
| **Teacher Signal**      | Cosine-similarity matrix [B, N, N] | Last-layer patch tokens [B, N, D] |
| **Knowledge Type**      | Procedural (attention patterns)    | Representational (feature space)  |
| **Loss Function**       | KL Divergence                      | SmoothL1                          |
| **Student Component**   | EfficientRelationConstructor       | D2ProjectionHead (1×1 conv)       |
| **Teacher Depth**       | Shallow / Middle / Deep avg.       | Last block only                   |
| **Distillation Weight** | λ = 0.00045                        | λ = 3000                          |
| **Inference Overhead**  | **None** (removed at test time)    | **None** (removed at test time)   |

</div>

---

## 💡 Key Findings

### 🎯 Main Discoveries

1. **Feature Distillation Outperforms Relation Distillation**: E3 (feature-based) transfers meaningful global context to YOLOv8, while E2 (relation-based) shows limited transfer due to fundamental architectural mismatch between ViT attention patterns and CNN feature maps.

2. **Dense Scene Improvements**: E3 is the **only model to exceed baseline in dense scenes** (16+ objects, +0.0029 AP), confirming successful transfer of global context awareness.

3. **Enhanced Occlusion Robustness**: E3 shows consistent improvements across all occlusion types, with the largest gains under heavy occlusion (grid 50%: +3.0%) and center occlusion (+3.4%).

4. **Small Object Retention**: E3's relative loss on small objects (-0.0015 AP) is approximately half that of E2 (-0.0032 AP), suggesting DINOv2's high-resolution patch-level semantics provide benefits for fine-grained detection.

5. **Per-Class Accuracy Gains**: E3 improves AP for 23/80 COCO classes (vs. 15 for E2), with the largest gains concentrated in visually fine-grained or small-object categories (toothbrush: +11.6 pp, orange: +9.6 pp, hot dog: +7.1 pp).

### 📌 Practical Takeaway

**When performing cross-architecture knowledge distillation from ViTs to CNNs, representational alignment (feature space) is more effective than procedural alignment (attention patterns).** The distillation framework requires no modification to the inference-time model, making it compatible with any deployment scenario that demands YOLOv8's speed.

---

## 🚀 Future Directions

- **Dynamic Distillation Weighting**: Implement an annealing strategy for λ—start high to leverage DINOv2's global features, decay over time to converge to detection-optimal solution
- **Selective-Layer Distillation**: Constrain distillation to high-level stages (P4/P5) where DINOv2's global semantics are most relevant
- **Architectural Bridging Modules**: Introduce lightweight attention mechanisms (e.g., CBAM) or single transformer layers into YOLO neck to facilitate seamless feature transfer
- **Cross-Domain Generalization**: Extend evaluation to cross-domain or few-shot settings to rigorously test robustness
- **Precision-Recall Trade-offs**: Analyze precision-recall curves under varying occlusion degrees for more granular robustness assessment

---

## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{he2026emergent,
  title={Emergent Global Receptive Fields in CNNs via Knowledge Distillation from Vision Transformers},
  author={Yang QI，Mai Bonan， Hong He ，Li Jiabo},
  journal={DSAI 5207 - Modern Deep Learning},
  year={2026}
}
```



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **DINOv2**: [Meta AI Research](https://github.com/facebookresearch/dinov2)
- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **COCO Dataset**: [Microsoft COCO](https://cocodataset.org/)

---

<div align="center">


**⭐ Star this repository if you find it helpful!**

**Made with by DSAI 5207 Team 19**

</div>
