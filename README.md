# Photoelastic-Centroid-Aware-YOLOv8L
YOLOv8-L based centroid detection in photoelastic granular flows on inclined surfaces.

This project implements a centroid detection system for photoelastic granular flows on inclined surfaces using the YOLOv8-L model with transfer learning. The goal is to accurately detect and track particle centroids and radii from experimental images to support further granular flow analysis.


## Features

- Transfer learning using YOLOv8-L for object detection on custom photoelastic datasets  
- Precise centroid and radius estimation from bounding boxes  
- Data preprocessing and augmentation pipelines  
- Inference scripts to visualize and export detection results  
- Evaluation metrics including precision, recall, and F1-score


## Dataset

- Experimental photoelastic images of granular flows on inclined planes  
- Annotations in YOLO format with bounding boxes approximating particle centroids and radii  
- Data folder structure example:
  ```dataset/
    ├── images/
    │ ├── train/
    │ ├── val/
    ├── labels/
    │ ├── train/
    │ ├── val/```
