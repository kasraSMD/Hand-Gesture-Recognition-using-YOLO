import time
import cv2
import os
import keyboard

class_name = "TimeOut_VFR"
num_saved_frames = 200

path = fr"Dataset\{class_name}"


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    print("Warming up camera...")
    time.sleep(2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for i in range(30):
        _, _ = cap.read()
    print("Camera ready!")

    frame_count = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Display', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):

            frame_name = f'{class_name}_{frame_count:0{len(str(num_saved_frames))}d}.jpg'
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)
            print(frame_count)
            frame_count += 1
        if frame_count == num_saved_frames + 1:
            break

        if keyboard.is_pressed('q'):
            break
    cap.release()
    print(f"Extracted {frame_count - 1} frames")


if __name__ == "__main__":
    extract_frames(video_path=0, output_folder=path)
