# Face Detection using Python and OpenCV with webcam

Face detection is a important application of computer vision that involves identifying human faces in images or videos. In this Article, we will see how to build a simple real-time face detection application using Python and OpenCV where webcam will be used as the input source.

## Step 1: Installing OpenCV Library


We have to install OpenCV library in our system, Use the following command to install it:

```python
pip install opencv-python
```


## Step 2: Importing OpenCV and Haar Cascade Classifier

We will use Haar Cascade Classifier which is a machine learning-based method for detecting objects in images. OpenCV provides this classifier as a pre-trained model for detecting faces. This classifier will help us to detect faces in the webcam feed. Download the Haar Cascade file from this link.

`face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')`: Loads pre-trained Haar Cascade face detection model to identify faces in images.

```python
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

## Step 3: Open Webcam and Check for Webcam Access

We use the cv2.VideoCapture() function to open the webcam. Here passing 0 refers to the default webcam. If the webcam is not accessible, we display an error message and stop the program.

```python
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
```