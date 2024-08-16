from ultralytics import YOLO

# Initialize the YOLO model with YOLOv8n (YOLOv9 is not yet available in most implementations)
model = YOLO("yolov8n.yaml")  # Using YOLOv8n, as YOLOv9s is not a standard model name

# Train the model
results = model.train(data="config.yaml", epochs=1, imgsz=640)
