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

## Step 4: Capture Frames, Convert to Grayscale and Detect Faces

Now it will continuously captures frames from the webcam. Each frame is converted to grayscale as face detection algorithms perform better on grayscale images. Then the Haar Cascade Classifier is applied to detect faces.

* gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY): Converts captured color frame (BGR) into grayscale for easier face detection.
* faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)): Detects faces in the grayscale image by adjusting size, filtering false positives and setting the minimum face size.

```python 
while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
```
## Step 5: Draw Rectangles Around Detected Faces and Display the Frame

For each face detected a green rectangle is drawn around it and frame with rectangles is displayed in a window titled "Face Detection"

```python
for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    cv2.imshow('Face Detection', frame)
```
## Step 6: Exit the Program

* if cv2.waitKey(1) & 0xFF == ord('q'): Checks if the 'q' key is pressed to exit the loop and stop the face detection process.

```python
if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Step 7: Release the Webcam and Close All Windows

When 'q' is pressed then webcam is released and any OpenCV windows are closed.

```python
cap.release()
cv2.destroyAllWindows()
```
* Using OpenCV's Haar Cascade Classifier this method provides a solid starting point for exploring real-time face detection and other computer vision projects.

Note : Above code will not run on online IDE. It will work on local system only.
