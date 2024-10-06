import cv2
import cvzone
import time
from ultralytics import YOLO

# Loading both models
yolo10 = YOLO("yolov10n.pt")
yolo11 = YOLO("yolo11n.pt")

# Lists to store nums for respective measurements
inferenceList = []
fpsList = []
confidenceList = []

cap = cv2.VideoCapture(0)

while True:
    frameCaptured, frame = cap.read()

    if not frameCaptured:
        break

    startTime = time.time()

    results = yolo11(frame)

    processedFrame = results[0].plot()

    inferenceTime = time.time() - startTime
    fps = 1 / inferenceTime

    inferenceList.append(inferenceTime)
    fpsList.append(fps)

    # Get confidence scores for all detected objects
    confidences = [box.conf[0] * 100 for box in results[0].boxes]
    if confidences:
        avgConfidence = sum(confidences) / len(confidences)
    else:
        avgConfidence = 0.0  # Handle case where no objects are detected

    confidenceList.append(avgConfidence)

    cvzone.putTextRect(processedFrame, f"FPS: {fps:.2f}", (10, 30), 2)
    cvzone.putTextRect(processedFrame, f"Inference Time: {inferenceTime:.4f} s", (10, 70), 2)

    cv2.imshow("YOLO Live Webcam", processedFrame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

avgFPS = sum(fpsList) / len(fpsList)
avgInference = sum(inferenceList) / len(inferenceList)
avgConfidence = sum(confidenceList) / len(confidenceList)

print(f"Average FPS: {avgFPS:.2f}")
print(f"Average Inference Time: {avgInference:.4f}")
print(f"Average Confidence: {avgConfidence:.2f}")

# YOLO 10 results
# FPS: 13.04
# INFERENCE: 0.0796s
# CONF: 81.95%

# YOLO 11 RESULTS
# FPS: 19.10
# Inference: 0.0546s
# Conf: 70.05%