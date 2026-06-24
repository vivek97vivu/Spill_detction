import cv2
import supervision as sv
from rfdetr import RFDETRBase

MODEL_PATH = "/home/vivek/Downloads/checkpoint_best_ema(2).pth"

print("Loading model...")
model = RFDETRBase.from_checkpoint(MODEL_PATH)
print("Loaded!")

cap = cv2.VideoCapture(4)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    detections = model.predict(frame, threshold=0.5)

    labels = [
        f"{conf:.2f}"
        for conf in detections.confidence
    ]

    annotated = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )

    cv2.imshow("RF-DETR", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()