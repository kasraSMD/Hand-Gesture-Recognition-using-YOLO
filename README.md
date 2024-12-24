# Hand-Gesture-Recognition-using-YOLO
This project focuses on **Hand Gesture Recognition** using the YOLO object detection model. The system is designed to detect and classify 15 different hand gestures in real-time, making it ideal for applications like gesture-based control, sign language interpretation, and more.

## Dataset

- **Classes:** The dataset includes 15 distinct hand gestures.  
- **Images:** 200 images were captured for each gesture, resulting in a total of **3000 labeled images**.
- **Annotation:** Images were labeled using the [LabelImg](https://github.com/tzutalin/labelImg) tool, generating `.xml` files for each image.  
- **Dataset Split:**  
  - **Training Set:** 2700 images.  
  - **Validation Set:** 300 images.  

### Gesture Classes
![Guide](https://github.com/user-attachments/assets/79cd6f9e-2151-4b67-8d23-14d71c9ae4bd)


## Workflow

### 1. Dataset Preparation
- **Step 1:** Hand gesture images were captured using a webcam and saved into folders named after the gesture classes using the Python script `F1_GenerateDataset.py`.  
- **Step 2:** Images were labeled using the LabelImg tool, and `.xml` files were generated.  
- **Step 3:** The `.xml` files were converted to YOLO-compatible `.txt` annotation files using the Python script `F2_CreateYOLO_Dataset.py`.  
- **Output:** The YOLO annotations are stored in the `YOLO_labels` folder.

### 2. Model
- The YOLO 11 Large model (`yolo11l.pt`) was used as the base model for training and fine-tuning.

### 3. Training
- The model was trained for **100 epochs** with **EarlyStopping** (patience=20) implemented to prevent overfitting.  
- **Optimizer:** Standard YOLO optimizer settings were used.  
- **Training/Validation Split:**  
  - Training: 2700 images.  
  - Validation: 300 images.  

### 4. Evaluation
- The model was evaluated on the validation dataset, producing the following results:  
![image](https://github.com/user-attachments/assets/fcc8ebaa-1f9a-4303-95b0-ee702eadb507)
![image](https://github.com/user-attachments/assets/b6c8238c-7163-4c43-9095-0a23bdcb724a)


## Usage

### Real-Time Gesture Detection
To use the trained model for real-time hand gesture recognition, run the `HandGestureRecognition.py` script.

```bash
python HandGestureRecognition.py
```

### Model Weights
- The best model (`best.pt`) and the last model (`last.pt`) are stored in the `runs` folder.  
- You can use these weights for further testing or deployment.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kasraSMD/Hand-Gesture-Recognition-using-YOLO.git
   cd Hand-Gesture-Recognition
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Project:
   ```bash
   python HandGestureRecognition.py
   ```
## File Structure

```
Hand-Gesture-Recognition/
├── F1_GenerateDataset.py        # Script to capture gesture images
├── F2_CreateYOLO_Dataset.py     # Script to convert XML annotations to YOLO format
├── F3_Train_YOLO.ipynb          # notebooks that uses for train YOLO model
├── HandGestureRecognition.py    # Script for real-time gesture detection
├── YOLO_dataset/                 # YOLO-compatible annotations
├── YOLO_labels/                 # YOLO-compatible annotations
└── runs/                        # Folder containing trained models
```

## Contributing

Contributions are welcome! If you have ideas to improve the system or add new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [YOLO](https://github.com/ultralytics/yolov5) for the object detection framework.
- [LabelImg](https://github.com/tzutalin/labelImg) for annotation tools.
- Everyone who contributed to the dataset creation and testing.

---
Enjoy using the Hand Gesture Recognition system! If you have any questions or feedback, feel free to reach out. 😊
