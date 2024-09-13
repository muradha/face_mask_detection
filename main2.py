from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("mask_detection.pt")

# Capture the video
cap = cv2.VideoCapture("192.168.1.64_01_20240905185931898.mp4")

# Get the original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the desired frame size (e.g., 640x480)
new_width, new_height = 640, 480

# Calculate scaling factors for width and height
scale_x = new_width / original_width
scale_y = new_height / original_height

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to desired dimensions
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Get YOLO model predictions on the original frame
    results = model(frame)  # Pass the original frame

    # Loop through the detected objects and draw bounding boxes with labels
    for result in results:
        boxes = result.boxes  # Get bounding boxes

        for box in boxes:
            # Extract coordinates, confidence, and class
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Scale bounding box coordinates to match resized frame
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            confidence = box.conf[0].item()  # Confidence score
            class_idx = int(box.cls[0])  # Class index
            label = model.names[class_idx]  # Get label name from class index

            # if label == "person":
            # Draw the bounding box on the resized frame
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label with confidence score
            label_text = f"{label} {confidence:.2f}"

            # Draw the label above the bounding box
            (label_width, label_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(
                resized_frame,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                resized_frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Text color (black)
                2,
            )


    # Display the frame with bounding boxes and labels
    cv2.imshow("YOLO Detection", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
