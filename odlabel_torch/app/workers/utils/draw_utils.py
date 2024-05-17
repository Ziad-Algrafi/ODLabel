import cv2 as cv
import random

def draw_boxes_on_image(annotated_frame, detections, class_colors, output_path):
    for (
        cls_id,
        x_scaled,
        y_scaled,
        w_scaled,
        h_scaled,
        class_name,
        confidence,
        *_,
    ) in detections:
        x1 = int(x_scaled - w_scaled / 2)
        y1 = int(y_scaled - h_scaled / 2)
        x2 = int(x_scaled + w_scaled / 2)
        y2 = int(y_scaled + h_scaled / 2)

        if cls_id not in class_colors:
            class_colors[cls_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

        color = class_colors[cls_id]

        img_height, img_width = annotated_frame.shape[:2]
        
        object_width = x2 - x1
        object_height = y2 - y1
        size_ratio = min(object_width, object_height) / min(img_width, img_height)
        font_scale = min(0.8, 0.4 + size_ratio * 0.6)
        text_thickness = max(1, int(font_scale * 2))

        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        caption = f"{class_name} {confidence*100:.2f}%"
        (tw, th), _ = cv.getTextSize(
            text=caption,
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            thickness=text_thickness,
        )
        th = int(th * 1.2)
        tw = int(tw * 1.1)

        bg_x1 = max(x1, 0)
        bg_y1 = max(y1 - th, 0)
        bg_x2 = min(x1 + tw, img_width)
        bg_y2 = y1

        cv.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv.putText(
            annotated_frame,
            caption,
            (x1, y1),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv.LINE_AA,
        )

    cv.imwrite(output_path, annotated_frame)
    return annotated_frame