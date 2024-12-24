import random
import cv2
from ultralytics import YOLO
import time

famous_colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 192, 203),  # Pink
    (128, 128, 128),  # Gray
    (0, 0, 0),  # Black
    (255, 255, 255),  # White
    (128, 128, 0),  # Olive
    (0, 128, 128),  # Teal
    (128, 0, 0),  # Maroon
    (0, 128, 0)  # Dark Green
]

random.shuffle(famous_colors)


def generate_class_colors(class_names):
    # List of 15 famous colors in RGB format
    colors = {}
    for class_name in class_names:
        colors[class_name] = random.choice(famous_colors)

    return colors


def run_webcam_detection():
    # Load the trained model
    model_path = r'runs\detect\gesture_detection\weights\last.pt'
    model = YOLO(model_path)

    # Load class names
    with open('YOLO_labels/classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    class_colors = generate_class_colors(class_names)
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 for default webcam

    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize FPS counter
    fps = 0
    frame_count = 0
    start_time = time.time()

    print("Starting webcam detection... Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break

        # Perform detection
        results = model.predict(frame, conf=0.5)  # Adjust confidence threshold as needed

        # Process results
        for result in results:
            boxes = result.boxes

            # Draw detections
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get confidence and class
                conf = float(box.conf)
                class_id = int(box.cls)
                class_name = class_names[class_id]

                # Draw bounding box
                color = class_colors[class_name]
                # Draw bounding box with the class-specific color
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Add label with class name and confidence
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        # Display FPS on frame
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Object Detection', frame)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run_webcam_detection()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
