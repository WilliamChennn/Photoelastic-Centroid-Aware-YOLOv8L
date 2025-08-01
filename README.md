# Photoelastic-Centroid-Aware-YOLOv8L
YOLOv8-L based centroid detection in photoelastic granular flows on inclined surfaces.

This project implements a centroid detection system for photoelastic granular flows on inclined surfaces using the YOLOv8-L model with transfer learning. The goal is to accurately detect and track particle centroids and radii from experimental images to support further granular flow analysis.


## Features

- Transfer learning using YOLOv8-L for object detection on custom photoelastic datasets  
- Precise centroid and radius estimation from bounding boxes  
- Data preprocessing and augmentation pipelines  
- Inference scripts to visualize and export detection results  
- Evaluation metrics including precision, recall, and F1-score


# YOLO Data Preprocessing Script
```Preprocessing.py``` is the script that converts circle annotation data stored in ```.mat``` files into YOLO-compatible label files. The script normalizes the coordinates and dimensions of annotated circles to match YOLO's labeling format.
## Label Annotation
To avoid manual annotation, we used image processing algorithms written in MATLAB to automatically extract the centroids and radii of photoelastic particles. These results are converted into YOLOv8-compatible labels. The following outlines the script structure and folder organization:
```
    Automated Labeling Pipeline via Image Processing
    │
    ├── Rawimg.tif
    |   ├── Centroid.m            % Main function: image processing and particle extraction
    |   ├── frameName.m           % Auxiliary function: generate file name
    |   ├── dig2str.m             % Auxiliary function: number formatting
    |   ├── PTV2.m                % Main process script (processing parameters and range can be modified)
    |   │
    |   ├── results/   
    |   │   ├── label.tif   
    |   │   └── label.mat
```
## Label Visualization
<table>
  <tr>
    <td>
      <img src="Rawimg.jpg" width="250"><br>
      <p align="center"><b>Raw image</b><br><code>Rawimg.tif</code></p>
    </td>
    <td>
      <img src="labeltif.png" width="250"><br>
      <p align="center"><b>Label image</b><br><code>label.tif</code></p>
    </td>
    <td>
      <p align="center"><b>Photoelastic Centroids</b><br><code>label.mat</code></p>
      <table>
        <tr><th>x (px)</th><th>y (px)</th><th>r (px)</th></tr>
        <tr><td>17.7100</td><td>255.3656</td><td>18</td></tr>
        <tr><td>24.1099</td><td>330.5085</td><td>24</td></tr>
        <tr><td>12.1825</td><td>399.0238</td><td>18</td></tr>
        <tr><td>12.1902</td><td>531.2927</td><td>18</td></tr>
        <tr><td>15.4267</td><td>612.7933</td><td>18</td></tr>
      </table>
    </td>
  </tr>
</table>


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
  ```
  from ultralytics import YOLO
    if __name__ == '__main__':
        model = YOLO('yolov8l.pt')  
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
