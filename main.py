from ultralytics import YOLO
import cv2

# Load a YOLOv8 PyTorch model
model = YOLO("best.pt")

# Open the video file
video_path = "192.168.1.64_01_20240905185931898.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Check if results are returned and process them
        if results:
            # Extract results from the list or directly
            if isinstance(results, list):
                results = results[0]  # Assuming results is a list and using the first element

            if results.boxes is not None:
                # Extract boxes, confidences, and class IDs
                boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
                confidences = results.boxes.conf.cpu().numpy()  # Confidence scores
                class_ids = results.boxes.cls.cpu().numpy()  # Class IDs

                # Draw bounding boxes on the frame
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                    label = f"Class {int(class_id)}: {conf:.2f}"  # Create label with class ID and confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Put label text
            else:
                print("No boxes detected")

        else:
            print("No results returned")

        # Resize the annotated frame if needed
        resized = cv2.resize(frame, (800, 800), interpolation=cv2.INTER_AREA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
