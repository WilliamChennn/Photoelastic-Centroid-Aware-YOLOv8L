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

        # **Reduce the impact of data augmentation**
        flipud=0.5,       # Reduce vertical flipping
        mosaic=0.3,       # Lower mosaic probability to avoid excessive distortion
        mixup=False,      # Disable mixup to prevent circle deformation
        scale=0.8,        # Limit scaling factor to avoid shifting the center point

        # **Color augmentation**
        hsv_h=0.015,      
        hsv_s=0.3,        
        hsv_v=0.9,        

        # **Adjust IoU and loss weights**
        iou=0.8,          # Lower IoU threshold to allow more detections
        box=4.0,          # Reduce box weight for balanced bounding box regression
        cls=1.0,          # Keep classification weight unchanged
        dfl=1.5,          # Increase distribution focal loss weight for better accuracy

        # **Mixed precision training**
        half=True,        # Enable mixed precision for faster training
        plots=True        # Enable loss curve visualization
    )

    # Validate the model
    metrics = model.val()