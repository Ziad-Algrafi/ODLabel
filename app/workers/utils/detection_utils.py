def infer(model, img, conf_threshold, log_update, image_file, iou_threshold):
    annotated_frame = img.copy()
    img_height, img_width = img.shape[:2]
    results = model(
        img, conf=conf_threshold, imgsz=640, iou=iou_threshold, verbose=False
    )
    detections = []
    names = model.names
    if results[0].boxes is not None:
        for box, class_id, score in zip(
            results[0].boxes.xywhn.cpu(),
            results[0].boxes.cls.int().cpu().tolist(),
            results[0].boxes.conf.cpu(),
        ):
            x, y, w, h = box
            cls_id = class_id
            confidence = float(score)
            class_name = names[cls_id]
            x_scaled = x * img_width
            y_scaled = y * img_height
            w_scaled = w * img_width
            h_scaled = h * img_height
            detections.append(
                (
                    cls_id,
                    x_scaled,
                    y_scaled,
                    w_scaled,
                    h_scaled,
                    class_name,
                    confidence,
                    img_width,
                    img_height,
                )
            )

    log_update_with_detections(log_update, detections, image_file)

    return detections, annotated_frame


def slice_and_infer(
    model,
    img,
    conf_threshold,
    log_update,
    image_file,
    slice_size,
    iou_threshold,
    terminate,
):
    annotated_frame = img.copy()
    img_height, img_width = img.shape[:2]
    slices = []
    num_slices = 0
    for y in range(0, img_height, slice_size // 2):
        for x in range(0, img_width, slice_size // 2):
            slice_y = min(y + slice_size, img_height)
            slice_x = min(x + slice_size, img_width)
            slice = img[y:slice_y, x:slice_x]
            slices.append((slice, x, y, slice_x - x, slice_y - y))
            num_slices += 1

    log_update(f"Number of slices: {num_slices}")
    detections = []
    names = model.names
    for i, (slice, x_offset, y_offset, slice_width, slice_height) in enumerate(
        slices, start=1
    ):
        results = model(
            slice,
            conf=conf_threshold,
            imgsz=slice_size,
            iou=iou_threshold,
            verbose=False,
        )[0]
        if results.boxes is not None:
            log_update(f"Slice {i} has detections.")
            for box in results.boxes:
                x, y, w, h = box.xywhn[0].tolist()
                cls_id = box.cls[0].int().item()
                confidence = box.conf[0].cpu()
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
                detections.append(
                    (
                        cls_id,
                        x_scaled,
                        y_scaled,
                        w_scaled,
                        h_scaled,
                        class_name,
                        confidence,
                        img_width,
                        img_height,
                    )
                )
        else:
            log_update(f"Slice {i} has no detections.")

    detections = apply_nms(detections, iou_threshold)
    log_update_with_detections(log_update, detections, image_file)
    return detections, annotated_frame


def apply_nms(detections, iou_threshold):
    boxes = []
    for det in detections:
        (
            cls_id,
            x_scaled,
            y_scaled,
            w_scaled,
            h_scaled,
            class_name,
            confidence,
            img_width,
            img_height,
        ) = det
        x1 = x_scaled
        y1 = y_scaled
        x2 = x_scaled + w_scaled
        y2 = y_scaled + h_scaled
        boxes.append([x1, y1, x2, y2, cls_id, confidence])

    sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    selected_boxes = []
    while len(sorted_boxes) > 0:
        selected_boxes.append(sorted_boxes[0])
        remaining_boxes = []
        for box in sorted_boxes[1:]:
            x1_a, y1_a, x2_a, y2_a, *_ = selected_boxes[-1]
            x1_b, y1_b, x2_b, y2_b, *_ = box
            intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(
                0, min(y2_a, y2_b) - max(y1_a, y1_b)
            )
            union = (
                (x2_a - x1_a) * (y2_a - y1_a)
                + (x2_b - x1_b) * (y2_b - y1_b)
                - intersection
            )
            if union == 0:
                iou = 0
            else:
                iou = intersection / union
            if iou < iou_threshold:
                remaining_boxes.append(box)
        sorted_boxes = remaining_boxes

    nms_detections = []
    for box in selected_boxes:
        x1, y1, x2, y2, cls_id, confidence = box
        w_scaled = x2 - x1
        h_scaled = y2 - y1
        x_scaled = x1
        y_scaled = y1
        nms_detections.append(
            (
                cls_id,
                x_scaled,
                y_scaled,
                w_scaled,
                h_scaled,
                class_name,
                confidence,
                img_width,
                img_height,
            )
        )

    return nms_detections


def log_update_with_detections(log_update, detections, image_file):
    valid_detections = [det for det in detections if det[5] != ""]
    if not valid_detections:
        log_update(f"No Results for image {image_file}")
    else:
        class_counts = {}
        for _, _, _, _, _, class_name, _, _, _ in valid_detections:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        log_message = f"Results for {image_file}:"
        for class_name, count in class_counts.items():
            plural_class_name = class_name + "s" if count > 1 else class_name
            log_message += f" {count} {plural_class_name}"
        log_update(log_message)
