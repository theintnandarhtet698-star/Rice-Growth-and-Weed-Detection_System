from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("rice_seg.pt")   # rice vs weed segmentation model

def analyze_growth(image_path, day):
    img = cv2.imread(image_path)
    results = model.predict(img)

    # Extract rice mask
    for r in results:
        masks = r.masks
        if masks is not None:
            for m in masks.xy:   # contour points
                x, y, w, h = cv2.boundingRect(m.astype(np.int32))
                rice_height_px = h

                # Example conversion (200px = 30cm)
                cm_per_pixel = 30 / 200
                rice_height_cm = rice_height_px * cm_per_pixel

                # Threshold check
                if day == 7:
                    threshold = 10
                elif day == 15:
                    threshold = 25
                elif day == 30:
                    threshold = 45
                else:
                    threshold = 0

                growth_status = "Normal"
                advice = "No weed detected"

                if rice_height_cm < threshold:
                    # Check weed detection
                    for c in r.boxes.cls.cpu().numpy():
                        if int(c) == 1:  # weed class
                            growth_status = "Below threshold"
                            advice = "Weed removal required"

                return rice_height_cm, growth_status, advice

    return None, "No rice detected", "Check image quality"
