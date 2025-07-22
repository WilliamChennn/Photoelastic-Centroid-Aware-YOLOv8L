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
    │ ├── test/
    ├── labels/
    │ ├── train/
    │ ├── test/

## Usage

- Training
  ```from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv8l model (large version)
    model = YOLO('yolov8l.pt')  

    # Train the model
    model.train(
        data='C:/Users/lab533/Desktop/Best_now0321/data.yaml',
        epochs=80,        
        imgsz=1280,       
        batch=4,          
        lr0=0.0003,       
        lrf=0.1,         
        freeze=0,         

        # **Reduce the impact of data augmentation**
        # **Color augmentation**      
        # **Adjust IoU and loss weights**
        # **Mixed precision training**
    )

    # Validate the model
    metrics = model.val()

- Inference
  ```yolo task=detect mode=predict model=runs/train/weights/best.pt source=dataset/images/val save=True
  Run prediction and save output images.

## Project Structure

  ```
  photoelastic-centroid-yolov8/
  │
  ├── dataset/                # Training and validation images and labels
  ├── runs/                   # YOLOv8 training output models and results
  ├── scripts/                # Custom training and prediction scripts
  ├── results/                # Prediction and evaluation results
  ├── README.md
  ├── requirements.txt        # Python package dependencies
  └── your_data.yaml          # YOLO format dataset config file
```

## Evaluation
- Evaluate detection performance using Precision, Recall, and F1-score.
- Custom scripts compare predicted centroids with ground truth, calculate errors, and visualize differences.

## References
- YOLOv8 official repository: https://github.com/ultralytics/ultralytics
- Relevant literature on photoelastic granular flow analysis

## Contact
CHEN BO WEI
Email: william20020602@gmail.com
GitHub: https://github.com/WilliamChennn
