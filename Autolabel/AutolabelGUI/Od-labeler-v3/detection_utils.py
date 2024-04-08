import cv2 as cv
import os

def draw_boxes_on_image(annotated_frame, detections, image_path, output_path=None, log_update=None):
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_with_boxes.jpg'

    for cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name in detections:
        x1 = int(x_scaled - w_scaled / 2)
        y1 = int(y_scaled - h_scaled / 2)
        x2 = int(x_scaled + w_scaled / 2)
        y2 = int(y_scaled + h_scaled / 2)
        label = class_name
        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(annotated_frame, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv.imwrite(output_path, annotated_frame)
    if log_update:
        log_update(f"Saved image with bounding boxes to {output_path}")

    return annotated_frame