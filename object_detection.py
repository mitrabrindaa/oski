import cv2
import numpy as np
import pyttsx3

# Load the YOLO object detection model
net = cv2.dnn.readNet(
    "C:\\Users\\asus\\Desktop\\IP\\yolov3.weights",
    "C:\\Users\\asus\\Desktop\\IP\\yolov3.cfg",
)

# Load the classes file
with open("C:\\Users\\asus\\Desktop\\OSKI\\od\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the layer names for the output layers of the YOLO model
layer_names = ["yolo_82", "yolo_94", "yolo_106"]

# Specify the minimum confidence required to filter out weak predictions
min_confidence = 0.5

# Load the video capture object
video_capture = cv2.VideoCapture(0)
desired_fps = 30
video_capture.set(cv2.CAP_PROP_FPS, desired_fps)
# Initialize the text-to-speech engine
engine = pyttsx3.init()

while True:
    # Grab a frame from the video capture object
    ret, frame = video_capture.read()

    # Only proceed if the frame was successfully grabbed
    if not ret:
        break

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Construct a blob from the frame and perform a forward pass with the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layer_names)

    # Initialize lists to store the bounding box coordinates, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    for output in layer_outputs:
        # Loop over each of the detections in the layer output
        for detection in output:
            # Extract the class ID and confidence from the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Only consider detections with a confidence greater than the minimum confidence
            if confidence > min_confidence:
                # Scale the bounding box coordinates back relative to the size of the frame
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append the bounding box, confidence, and class ID to their respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Convert the tuple to a list
    indices = list(indices)

    # Loop over the remaining indices after non-maximum suppression
    for i in indices:
        box = boxes[i]
        x, y, w, h = box

        # Draw the bounding box and label on the frame
        color = (0, 255, 0)  # BGR color format
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

        # Speak the name of the detected object
        engine.say(f"it is a {classes[class_ids[i]]}")
        engine.runAndWait()

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the window
video_capture.release()
cv2.destroyAllWindows
