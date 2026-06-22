import cv2
from ultralytics import YOLO

model = YOLO("/home/vivek/Downloads/spills/models/best.pt")

cap = cv2.VideoCapture(4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()