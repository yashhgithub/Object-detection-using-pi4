from ultralytics import YOLO
import cv2

# Load a YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # You can also try 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', etc.

# Load an image or video
image_path = "/home/ras4/project/testimg/img3.jpg"
image = cv2.imread(image_path)

# Perform object detection
results = model.predict(image)

# Draw bounding boxes and labels on the image
for box in results[0].boxes:
    # Extract bounding box coordinates and label
    x1, y1, x2, y2 = box.xyxy[0]  # Bounding box (x1, y1) -> (x2, y2)
    conf = box.conf[0]  # Confidence score
    class_id = int(box.cls[0])  # Class ID

    # Draw the bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Draw the label and confidence score
    label = f"{model.names[class_id]}: {conf:.2f}"
    cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Display the result
cv2.imshow("YOLOv8 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()