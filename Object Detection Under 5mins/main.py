import cv2
import cvzone

from ultralytics import YOLO

objectModel = YOLO("yolov10n.pt")

# Capturing webcam video
cap = cv2. VideoCapture(0)

while True:
    frameCaptured, frame = cap.read()

    # Breaks out of loop if no frame is captured
    if not frameCaptured:
        break
    
    objects = objectModel(frame)

    for object in objects:

        boxes = object.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].numpy().astype("int")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)

            classNum = int(box.cls[0])
            className = objectModel.names[classNum]

            confidence = box.conf[0]
            confidence = confidence * 100

            cvzone.putTextRect(frame, f"{className} | {confidence:.2f}% confident.", [x1 + 8, y1 - 12], scale=2)

    cv2.imshow("frame", frame)
    keyPress = cv2.waitKey(10)

    # Break out of the loop if Esc is pressed.
    if keyPress == 27:
        break


cap.release()
cv2.destroyAllWindows()