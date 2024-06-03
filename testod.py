import torch
import cv2


def run_yolov5_object_detection():
    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5:v6.0", "yolov5s")

    # Set model to inference mode
    model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully

    # Get the frame rate from the camera
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the frame rate is zero
    if fps == 0:
        print("Error: Unable to retrieve frame rate from the camera.")
        return

    # Set target FPS
    target_fps = 30
    frame_skip = int(fps / target_fps)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Skip frames to achieve target FPS
        if frame_count % frame_skip == 0:
            # Resize frame for faster processing
            resized_frame = cv2.resize(frame, (1280, 800))

            # Mirror the frame
            mirrored_frame = cv2.flip(resized_frame, 1)  # 1 for horizontal flip

            # Inference
            results = model(mirrored_frame)

            # Display results
            cv2.imshow("YOLOv5 Object Detection", results.render()[0])

        frame_count += 1

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Run the YOLOv5 object detection function
run_yolov5_object_detection()
