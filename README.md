# MultiViewDentalRecon3D

**3D Reconstruction of Dental Images Using Single-View or Multi-View Feed-Forward Deep Learning**

MultiViewDentalRecon3D reconstructs a colored 3D point cloud of teeth and oral structures from single-view or multi-view dental images. The system leverages a transformer-based feed-forward deep learning model (MapAnything) to directly estimate dense 3D geometry from images without traditional multi-view stereo optimization or iterative bundle adjustment.

This project is designed for research in digital dentistry, oral scanning, AI-driven dental reconstruction, and low-cost 3D reconstruction systems.

<p align="center">
  <img src="images/201935201908021.JPG" width="30%" />
  <img src="images/201935201908022.JPG" width="30%" />
  <img src="images/201935201908023.JPG" width="30%" />
</p>

<p align="center">
  <img src="images/result.gif" width="80%" />
</p>


---

## 🦷 Overview

The framework supports:

- Single-view 3D reconstruction
- Multi-view reconstruction and geometric fusion
- Optional camera pose integration
- Optional binary mask filtering
- Second-pass refinement for improved consistency
- Export of colored 3D point clouds (`.obj`)

Unlike classical geometry pipelines, this method relies on feed-forward transformer inference for scalable and efficient reconstruction.

---

## 🔬 Method Pipeline

1. Load single-view or multi-view dental images  
2. Optionally apply binary segmentation masks  
3. Perform first-pass 3D inference using MapAnything  
4. Optionally perform second-pass refinement using predicted camera parameters  
5. Filter valid regions using model confidence mask and optional input mask  
6. Export colored 3D point cloud  

---

## 📁 Project Structure

```
MultiViewDentalRecon3D/
│
├── data/
│   ├── images/              # Input dental images
│   └── masks/               # Optional binary masks
│
├── main.py
├── utilities.py
├── README.md
└── requirements.txt
```

---

## 📦 Installation

This project depends on **MapAnything**.

Please install MapAnything first using the official repository:

```bash
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything

# Create and activate conda environment
conda create -n mapanything python=3.12 -y
conda activate mapanything

# Install MapAnything
pip install -e .

# Install full optional dependencies
pip install -e ".[all]"

pre-commit install
```

After installing MapAnything, clone this repository:

```bash
git clone https://github.com/yourusername/MultiViewDentalRecon3D.git
cd MultiViewDentalRecon3D
```

Make sure the `mapanything` environment is activated before running.

---

## 🚀 Usage

### Basic Reconstruction

```bash
python main.py \
    --data_root data \
    --output output/reconstruction
```

---

### With Binary Mask Filtering

```bash
python main.py \
    --data_root data \
    --output output/reconstruction \
    --with_mask
```

Mask requirements:

- Masks must be stored in: `data/masks/`
- Mask filenames must match image filenames
- White (255) → valid region
- Black (0) → ignored region
- RGB masks are automatically converted to grayscale and binarized

---

### With Camera Pose Integration

```bash
python main.py \
    --data_root data \
    --output output/reconstruction \
    --with_pose
```

---

### With Second-Pass Refinement

```bash
python main.py \
    --data_root data \
    --output output/reconstruction \
    --second_pass
```

---

### Full Pipeline (Pose + Mask + Second Pass)

```bash
python main.py \
    --data_root data \
    --output output/reconstruction \
    --with_pose \
    --with_mask \
    --second_pass
```

---

## 📤 Output

The pipeline generates:

```
reconstruction_first.obj
reconstruction_second.obj
```

Each vertex in the exported OBJ file contains:

```
v x y z r g b
```

Only geometrically valid and mask-filtered points are exported.

You can visualize results using:

- MeshLab  
- CloudCompare  
- Blender  

---

## 🧠 Model Backbone

This project uses:

**facebook/map-anything**

for dense depth and 3D structure estimation via transformer-based feed-forward inference.

---

## 🎯 Research Applications

- Digital dentistry
- Smartphone-based oral 3D reconstruction
- Dental alignment analysis
- Vision-based prosthetic modeling
- Low-cost intraoral 3D scanning systems

---

## 🧪 Future Work

- Surface mesh reconstruction from dense point clouds
- Confidence-weighted point filtering
- Multi-view geometric consistency refinement
- Texture blending across views
- Dental-specific model fine-tuning

---

## 📜 License

MIT License
