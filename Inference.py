from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def adjust_radius(r):
    """Adjust radius to the closest predefined value (18 or 24)."""
    return 18 if abs(r - 18) < abs(r - 24) else 24

if __name__ == '__main__':
    model = YOLO('C:/Users/lab533/Desktop/retrain/runs/detect/train4/weights/best.pt')
    
    image_path = 'C:/Users/lab533/Desktop/objection/YOLO28D12cm/28度12cm(4)_img/2021.11.23_002927.tif'
    output_folder = 'C:/Users/lab533/Desktop/'
    os.makedirs(output_folder, exist_ok=True)
    output_image_path = os.path.join(output_folder, 'output_image.jpg')
    

    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    thresholded = np.where(gray_image >= 170, 90, gray_image)
    thresholded_path = os.path.join(output_folder, 'thresholded_sample.tif')
    cv2.imwrite(thresholded_path, thresholded)
    

    results = model.predict(source=thresholded_path, save=False, conf=0.05, iou=0.2, max_det=1000, agnostic_nms=True)
    
    detections = []
    for result in results:
        if result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy  
            
            for box in boxes:
                x1, y1, x2, y2 = box.cpu().numpy()
                x_i = (x1 + x2) / 2
                y_i = (y1 + y2) / 2
                r_i = (x2 - x1) / 2
                
                adjusted_r = adjust_radius(r_i)
                if r_i >= 17:
                    detections.append((x_i, y_i, adjusted_r))
    
    sorted_detections = sorted(detections, key=lambda d: (d[0], d[1]))
    

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  
    
    for x_i, y_i, r_i in sorted_detections:
        circle = plt.Circle((x_i, y_i), r_i, color='r', fill=False, linewidth=1.6)
        ax.add_patch(circle)
    
    ax.set_axis_off()
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    for i, (x_i, y_i, r_i) in enumerate(sorted_detections):
        print(f"Object {i + 1}: Center (x_i, y_i) = ({x_i:.2f}, {y_i:.2f}), Adjusted Radius r_i = {r_i:.2f}")