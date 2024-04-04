import cv2 as cv
import os

def infer(model, img, conf_threshold, log_update, image_file):

    annotated_frame= img.copy()
    img_height, img_width = img.shape[:2]
    results = model(img, conf=conf_threshold, imgsz=640)

    detections = []
    names = model.names 
    if results[0].boxes is not None:
        log_update(f"Results for {image_file}:") 
        boxes = results[0].boxes.xywhn.cpu()
        clss = results[0].boxes.cls.int().cpu().tolist()
        for box, class_id in zip(boxes, clss):
            x, y, w, h = box
            cls_id = class_id
            class_name = names[cls_id]
            x_scaled = x * img_width
            y_scaled = y * img_height
            w_scaled = w * img_width
            h_scaled = h * img_height

            detections.append((cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name))
    else:
        log_update(f"No Results for {image_file}:")

    return detections, annotated_frame

def slice_and_infer(model, img, conf_threshold, log_update, image_file, slice_size):
    annotated_frame = img.copy()
    height, width = img.shape[:2]
    slices = []
    num_slices = 0

    for y in range(0, height, slice_size // 2):
        for x in range(0, width, slice_size // 2):
            slice_y = min(y + slice_size, height)
            slice_x = min(x + slice_size, width)
            slice = img[y:slice_y, x:slice_x]
            slices.append((slice, x, y, slice_x - x, slice_y - y))
            num_slices += 1

    log_update(f"Number of slices: {num_slices}")
    detections = []
    names = model.names 
    for slice, x_offset, y_offset, slice_width, slice_height in slices:
        results = model(slice, conf=conf_threshold, imgsz=slice_size)[0]
        if results.boxes is not None:
            log_update(f"Results for {image_file}:")
            for box in results.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                cls_id = box.cls[0].int().item()
                class_name = names[cls_id]
                x_pixel = x * slice_width
                y_pixel = y * slice_height
                w_pixel = w * slice_width
                h_pixel = h * slice_height
                x_global = x_pixel + x_offset
                y_global = y_pixel + y_offset
                x_scaled = x_global
                y_scaled = y_global
                w_scaled = w_pixel
                h_scaled = h_pixel
                detections.append((cls_id, x_scaled, y_scaled, w_scaled, h_scaled, class_name))
        else:
            log_update(f"No Results for {image_file}:")

    return detections, annotated_frame

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