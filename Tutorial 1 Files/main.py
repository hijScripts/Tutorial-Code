# Imports required for object detection
import cvzone
import cv2

# Import specifically required to use YOLO model.
from ultralytics import YOLO

# Sets up YoloV10 as the object processing model
objectModel = YOLO("yolov10n.pt")

# Gets default webcam footage
cap = cv2.VideoCapture(0)

# Infinitely captures frames from our webcam feed until Esc pressed or program interrupted
while True:

    # Gets status of frame being captured and the frame itself
    bool, frame = cap.read()

    # Breaks out of loop if no frame
    if not bool:
        break
    
    # Getting a list of objects
    objects = objectModel(frame)

    # Going over objects
    for object in objects:

        # Assigning all bounding boxes for the object to a variable
        boxes = object.boxes

        for box in boxes:

            # Gets top left and bottom right corner position
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype("int")

            # Gets number of class then matches to associated name.
            classNum = int(box.cls[0])
            className = objectModel.names[classNum]

            # Gets confidence value and then multiplies by 100 to get a percentage between 0-100
            confidence = box.conf[0]
            confidence = confidence * 100

            # Rectangle to outline object then text to display info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)
            cvzone.putTextRect(frame, f"{className} | {confidence:.2f}%", [x1 + 8, y1 - 12], scale=2)

    # Shows frame and waits 1ms before user input does anything
    cv2.imshow("frame", frame)
    keyPress = cv2.waitKey(1)

    # Breaks out of while loop if Esc key is pressed. 27 is ASCII code for Esc.
    if keyPress == 27:
        print("Esc key pressed.")
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
