# Vehicular Collaborative Perception for Illegal Parking Detection in the CARLA Simulation Environment

This repository provides Python code for vehicular collaborative perception-based illegal parking detection in the CARLA simulation environment.

The pipeline processes CARLA-collected vehicle images and metadata, estimates grid-level occupancy and visibility, computes detection reliability, and performs collaborative inference for illegal parking detection.

---

## Files

- **config.json**  
  Defines dataset paths, model paths, output paths, parking-grid settings, and parameters for each module.

- **main.py**  
  Runs the complete illegal parking detection pipeline from data loading to final collaborative inference.

- **detection.py**  
  Detects vehicles from CARLA RGB images using YOLOv8.

- **projection.py**  
  Projects detected vehicles from image coordinates to the CARLA world coordinate system.

- **occupancy.py**  
  Estimates whether each predefined parking grid is occupied by a detected vehicle.

- **visibility.py**  
  Computes the visibility of each parking grid under camera field-of-view and occlusion conditions.

- **reliability.py**  
  Calculates grid-level extrinsic reliability based on occupancy and visibility cues.

- **intrinsic_em.py**  
  Performs EM-based collaborative inference to jointly estimate intrinsic reliability and final illegal parking results.

- **evaluation.py**  
  Saves output matrices and evaluates the final detection results.

---

## Requirements

- Python 3.8+
- NumPy
- OpenCV
- Ultralytics YOLOv8
- CARLA-collected RGB images and metadata

Install dependencies:

```bash
pip install numpy opencv-python ultralytics
