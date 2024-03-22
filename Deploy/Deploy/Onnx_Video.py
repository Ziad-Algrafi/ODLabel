import cv2
import onnxruntime as ort
import numpy as np
import time

def parse_row(row):
    xc, yc, w, h = row[:4]
    x1 = (xc - w / 2) / 640 * img_width
    y1 = (yc - h / 2) / 640 * img_height
    x2 = (xc + w / 2) / 640 * img_width
    y2 = (yc + h / 2) / 640 * img_height
    prob = row[4:].max()
    class_id = row[4:].argmax()
    return [x1, y1, x2, y2, class_id, prob]

def intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return max(0, x2 - x1) * max(0, y2 - y1)

def union(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    return box1_area + box2_area - intersection(box1, box2)

def iou(box1, box2):
    return intersection(box1, box2) / union(box1, box2)

def nms(boxes, threshold):
    sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    selected_boxes = []
    while len(sorted_boxes) > 0:
        selected_boxes.append(sorted_boxes[0])
        sorted_boxes = [box for box in sorted_boxes if iou(box, selected_boxes[-1]) < threshold]
    return selected_boxes

# sess_options = ort.SessionOptions()
# sess_options.intra_op_num_threads = 16
# sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
# model = ort.InferenceSession("/home/ziad/best.onnx", providers=['CPUExecutionProvider'], sess_options=sess_options)



sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 0
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  
model = ort.InferenceSession("/home/ziad/best.onnx", providers=['CPUExecutionProvider'], sess_options=sess_options)

video_path = "/home/ziad/Files/Crowded2.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    success, frame = cap.read()

    img_height, img_width, _ = frame.shape
    img_with_boxes = frame.copy()

    if success:
        frame = cv2.resize(frame, (640, 640))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_data = np.array(frame, dtype=np.float32).transpose(2, 0, 1).reshape(1, 3, 640, 640)
        input_data = input_data / 255.0
     
        start_time = time.time()
        outputs = model.run(["output0"], {"images": input_data})
        end_time = time.time()
        
        inference_time_ms = (end_time - start_time) * 1000  
        print(f"Inference time for frame: {inference_time_ms:.2f} ms")

        output = outputs[0]
        output = output[0].transpose()
        row = output[0]

        boxes = [row for row in [parse_row(row) for row in output] if row[5] > 0.3]
      

        nms_boxes = nms(boxes, 0.7)

        for box in nms_boxes:
            x1, y1, x2, y2, class_id, prob = box
            print("class", class_id, f"prob: {prob:.2f}")
            cv2.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
         

        cv2.imshow("YOLOv8 Tracking", img_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

