# Photoelastic-Centroid-Aware-YOLOv8L
YOLOv8-L based centroid detection in photoelastic granular flows on inclined surfaces.

This project implements a centroid detection system for photoelastic granular flows on inclined surfaces using the YOLOv8-L model with transfer learning. The goal is to accurately detect and track particle centroids and radii from experimental images to support further granular flow analysis.
### photoelastic granular flows on inclined surfaces
<td style="text-align: center; vertical-align: middle;">
  <div style="display: inline-block; text-align: center;">
    <img src="referimages/slope.png" width="700"><br>
  </div>
</td>

## Features

### Industrial Application Value
The photoelastic particle flow image analysis and YOLO + CNN pipeline developed in this project is not only suitable for laboratory environments but also has the potential to be extended to industrial scenarios, such as:

- **Particle Detection**: Can be used in chemical, pharmaceutical, or food production lines to automatically monitor particle size and position.
- **Semiconductor Defect Detection**: High-precision location of tiny defects or particle anomalies on wafer surfaces.
- **High-Speed Image Monitoring**: Suitable for analyzing high-speed image data during production, detecting object position and size in real time.

This pipeline demonstrates the complete process from data processing, object detection, to accuracy assessment. Through self-supervised learning and GAN-based augmentation, it can be extended to industrial applications where data is scarce or annotation costs are high.

## Project Pipeline

- Transfer learning using YOLOv8-L for object detection on custom photoelastic datasets  
- Precise centroid and radius estimation from bounding boxes  
- Data preprocessing and augmentation pipelines  
- Inference scripts to visualize and export detection results  
- Evaluation metrics including precision, recall, and F1-score
<td style="text-align: center; vertical-align: middle;">
  <div style="display: inline-block; text-align: center;">
    <img src="referimages/pipeline.png" width="700"><br>
  </div>
</td>

# YOLO Data Preprocessing Script
## 1. Image Preprocessing
To improve the model's ability to perceive key features in images during training, this study performed a series of image preprocessing operations on the raw photoelastic images.

Mainly these operations include:

- Gaussian blurring: This reduces high-frequency noise and smoothes background and non-target areas.

- Contrast enhancement: This enhances photoelastic streaks and particle edges, making force chain features more distinct.

This helps the model more easily focus on key areas such as particle structure and stress distribution during training, thereby improving prediction accuracy.
### Image-Preprocess Visualization
<table>
  <tr>
    <td>
      <img src="referimages/baselineimage.png" width="350"><br>
      <p align="center"><b>Raw image</b><br><code>basename.tif</code></p>
    </td>
    <td>
      <img src="referimages/blurredimage.png" width="350"><br>
      <p align="center"><b>GaussianBlur</b><br><code>basename_(1).tif</code></p>
    </td>
    <td>
      <img src="referimages/contrastedimage.png" width="350"><br>
      <p align="center"><b>Contrast</b><br><code>basename_(2).tif</code></p>
    </td>
  </tr>
</table>

## 2. Label Annotation
### Matlab Labeling
To avoid manual annotation, we used image processing algorithms written in MATLAB to automatically extract the centroids and radii of photoelastic particles. These results are converted into YOLOv8-compatible labels. The following outlines the script structure and folder organization:
```
    Automated Labeling Scripts via Image Processing
    │
    ├── Rawimg.tif
    |   ├── Centroid.m            % Main function: image processing and particle extraction
    |   ├── frameName.m           % Auxiliary function: generate file name
    |   ├── dig2str.m             % Auxiliary function: number formatting
    |   ├── PTV2.m                % Main process script: modify processing parameters and range 
    |   │
    |   ├── results/   
    |   │   ├── label.tif   
    |   │   └── label.mat
```
### Label Visualization
<table>
  <tr>
    <td>
      <img src="referimages/Rawimg.jpg" width="350"><br>
      <p align="center"><b>Raw image</b><br><code>Rawimg.tif</code></p>
    </td>
    <td>
      <img src="referimages/labeltif.png" width="350"><br>
      <p align="center"><b>Label image</b><br><code>label.tif</code></p>
    </td>
    <td>
      <table>
        <tr><th>x (px)</th><th>y (px)</th><th>r (px)</th></tr>
        <tr><td>17.7100</td><td>255.3656</td><td>18</td></tr>
        <tr><td>24.1099</td><td>330.5085</td><td>24</td></tr>
        <tr><td>12.1825</td><td>399.0238</td><td>18</td></tr>
        <tr><td>12.1902</td><td>531.2927</td><td>18</td></tr>
        <tr><td>15.4267</td><td>612.7933</td><td>18</td></tr>
      </table>
      <p align="center"><b>Photoelastic Centroids</b><br><code>label.mat</code></p>
    </td>
  </tr>
</table>

## 3. Label Transformation

```label preprocessing.py``` is the script that converts circle annotation data stored in ```.mat``` files into YOLO-compatible label files. The script normalizes the coordinates and dimensions of annotated circles to match YOLO's labeling format.
- Extracts circle coordinates and dimensions (```x```, ```y```, ```r```) in ```basename.mat``` files.
- Extracts Raw image dimensions (```W```, ```H```).
- Generates YOLO label as ```basename.txt``` files with the format:
```
<Class_id> <Normalized_x> <Normalized_y> <Normalized_w> <Normalized_h>
```
     <Class_id> : 0 (single class for circles)
     ( <Normalized_x> , <Normalized_y> ) : ( x / W , y / H ), Normalized circle center coordinates.
     ( <Normalized_w> , <Normalized_h> ) : ( 2r / W , 2r / H ), Normalized circle dimensions.

- Add two additional labels with the same content and distinguished by ```basename_(1).txt``` and ```basename_(2).txt``` to correspond to the Gaussian blurred and contrast enhanced images.

## 4. Dataset Organization
### Original Dataset
```
dataset/
├── baseline/
│   ├── basename.tif/
│   │   ├── train/
│   │   ├── test/
│   ├── basename.txt/
│   │   ├── train/
│   │   ├── test/
├── blur/
│   ├── basename_(1).tif/
│   │   ├── train/
│   │   ├── test/
│   ├── basename_(1).txt/
│   │   ├── train/
│   │   ├── test/
├── contrast/
│   ├── basename_(2).tif/
│   │   ├── train/
│   │   ├── test/
│   ├── basename_(2).txt/
│   │   ├── train/
│   │   ├── test/
```
### Step1. Transform the Dataset Structure
To centrally manage training and testing data for different image processing strategies (such as blurring and contrast adjustment), we have reorganized the folder structure, which was originally categorized by processing type (e.g., ```baseline/```, ```blur/```, ```contrast/```), into a structure centered around training purposes (```train/```, ```test/```).
```
dataset/
├── train/
│   ├─ images/
│   │   ├── basename.tif 
│   │   ├── basename_(1).tif 
│   │   ├── basename_(2).tif
│   ├─ labels/
│   │   ├── basename.txt
│   │   ├── basename_(1).txt
│   │   ├── basename_(2).txt 
├── test/ 
│   ├─ images/
│   │   ├── basename.tif 
│   │   ├── basename_(1).tif 
│   │   ├── basename_(2).tif
│   ├─ labels/
│   │   ├── basename.txt
│   │   ├── basename_(1).txt
│   │   ├── basename_(2).txt 
```
### Step2. Create ```data.yaml```
Define the dataset configuration in ```data.yaml```. This file tells YOLO where to find the training and validation data, the number of classes, and their names.
```yaml
train: dataset/train       # Path to training images and labels
val: dataset/test          # Path to testing images and labels

nc: 1                # Number of classes (only circles)
names: [circle]      # Class name
```
## 5. YOLOv8l Training

```train.py``` Training Script with Hyperparameter Tuning
  
```python
from ultralytics import YOLO
if __name__ == '__main__':
# Load YOLOv8l model (large version)
model = YOLO('yolov8l.pt')  

# Train the model
model.train(
data='C:/Users/lab533/Desktop/Best_now0321/data.yaml',
  epochs=80,        # Increase training epochs for better convergence
  imgsz=1280,       # Increase image resolution for better center point accuracy
  batch=4,          # Reduce batch size to accommodate larger model
  lr0=0.0003,       # Lower initial learning rate to prevent overfitting
  lrf=0.1,          # Reduce final learning rate for smoother convergence
  freeze=0,         # Freeze the first 5 layers to stabilize training
)
    # Validate the model
    metrics = model.val()
```
## 6. YOLOv8 circular object detection
```Inference.py``` shows that the particle analysis in photoelastic images by ```best.pt```, which is trained YOLOv8l. Assuming each detection frame represents a roughly circular particle, it converts it into (x, y, r) format for subsequent analysis.
  - Load the trained YOLOv8 model
```python
    model = YOLO('runs/detect/train/weights/best.pt')
```
- Convert each box to a circle center and radius, and normalize by size (18 or 24)
```python
def adjust_radius(r):
    """Adjust radius to the closest predefined value (18 or 24)."""
    return 18 if abs(r - 18) < abs(r - 24) else 24
```
- Perform inference to obtain all bounding boxes
```python
    # inference
    results = model.predict(source=thresholded_path, save=False, conf=0.05, iou=0.2, max_det=1000, agnostic_nms=True)
    
    detections = []
    for result in results:
        if result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy  # bounding box
            
            for box in boxes:
                x1, y1, x2, y2 = box.cpu().numpy()
                x_i = (x1 + x2) / 2
                y_i = (y1 + y2) / 2
                r_i = (x2 - x1) / 2
                
                adjusted_r = adjust_radius(r_i)
                if r_i >= 17:
                    detections.append((x_i, y_i, adjusted_r))
    
    sorted_detections = sorted(detections, key=lambda d: (d[0], d[1]))
```
### Output visualization 
1. Visualizations can clearly show the discrepancies between model predictions and actual conditions.
2. This helps quickly identify model errors and potential problems.
3. It also enhances the persuasiveness of your results, making them easier to explain to non-experts.
<table>
  <tr>
    <td>
      <img src="referimages/Inferenceimage.jpg" width="350"><br>
      <p align="center"><b>Inference image</b></p>
    </td>
  </tr>
</table>


## Project Structure
```css
Photoelastic-Centroid-Aware-YOLOv8L/
│
├── referimages/                     # reference images in the README
|   ├──Inferenceimage.jpg
│   ├── Rawimg.jpg
│   ├── baelineimage.png
│   ├── blurredimage.png
│   ├── contrastedimage.png
│   └── labeltif.png
|
├── matlab_labeling/              # MATLAB automatic labeling source code and results
│   ├── Centroid.m
│   ├── frameName.m
│   ├── dig2str.m
│   ├── PTV2.m
│   └── results/
│   |   ├── label.tif
│   |   └── label.mat
|
├── Inference.py                     # Inference + circle visualization scripts
├── README.md                        # Project description and instructions (written content is recommended here)
├── data.yaml                        # Dataset configuration file required for YOLOv8 training
├── train.py
├── label_preprocessing.py           # Convert .mat files to YOLO-formatted labels
├── image_preprocessing.py           # Image preprocessing such as Gaussian blur and contrast enhancement
|
├── runs/                            # YOLOv8 training output (model weights, logs)
│   ├── detect/
│   ├── train/
│   ├── weights/
│   └── best.pt
```

## Evaluation
- Evaluate detection performance using Precision, Recall, and F1-score.
<table>
  <tr>
    <td>
      <img src="referimages/P_curve.png" width="500"><br>
    </td>
    <td>
      <img src="referimages/F1_curve.png" width="500"><br>
    </td>
    <td>
      <img src="referimages/R_curve.png" width="500"><br>
    </td>
  </tr>
</table>
- Custom scripts compare predicted centroids with ground truth, calculate errors, and visualize differences.

## References
- YOLOv8 official repository: https://github.com/ultralytics/ultralytics
- Relevant literature on photoelastic granular flow analysis

## Contact
CHEN BO WEI
Email: william20020602@gmail.com
GitHub: https://github.com/WilliamChennn
