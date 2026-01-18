# Camera + Skeleton Projection Tools (Python)

A small Python library for working with COLMAP-style camera files and per-frame data (images, skeletons, meshes). It includes utilities to load intrinsics/extrinsics, project 3D points to image space, draw overlays, and generate videos from rendered frames.

This repo is meant to stay simple and editable: most scripts use a few path variables at the top so you can plug in your own dataset layout fast.

## What’s included

- **Skeleton overlay renderer**
  - Loads camera intrinsics (`cameras.txt`) and extrinsics (`images.txt`)
  - Loads per-frame skeleton files (`.skl`)
  - Projects joints into image space and draws a colored skeleton overlay on top of frames
  - Optionally exports an MP4 from the generated frames

- **Camera/frame loader utilities**
  - Minimal loader + iteration logic for intrinsics/extrinsics/images
  - Useful as a base for other projection, debug, or dataset processing scripts

## Requirements

- Python 3.x
- `numpy`
- `opencv-python`

Optional (only if you use the parts that need them):
- `matplotlib`

Install:
```bash
pip install numpy opencv-python matplotlib
