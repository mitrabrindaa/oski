import os
import requests

# Define the URL of the YOLOv5s.pt file
url = "https://github.com/ultralytics/yolov5s/releases/download/v6.0/yolov5s.onnx"

# Define the filename and path to save the file
filename = "yolov5s.onnx"
path = os.path.join(os.getcwd(), filename)

# Download the file using the requests library
response = requests.get(url)

# Save the file to the specified path
with open(path, "wb") as f:
    f.write(response.content)

# Print a message to confirm that the file was downloaded successfully
print(f"{filename} downloaded successfully and saved to {path}")
