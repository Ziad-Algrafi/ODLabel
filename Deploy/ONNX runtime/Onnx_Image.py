import onnxruntime as ort
import numpy as np
import cv2 as cv
import time

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

model = ort.InferenceSession("/home/ziad/yolov8x-worldv2.onnx", providers=['CPUExecutionProvider'])

inputs = model.get_inputs()
print(len(inputs))

input = inputs[0]

print("Name:", input.name)
print("Type:", input.type)
print("Shape:", input.shape)

img_original = cv.imread("/home/ziad/Cars_at_traffic_1206.jpeg")
img_height, img_width, _ = img_original.shape

img_with_boxes = img_original.copy()

img = cv.resize(img_original, (640, 640))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
input_img = np.array(img, dtype=np.float32).transpose(2, 0, 1).reshape(1, 3, 640, 640)
print(input_img.shape)
input_img = input_img / 255.0
input_img[0, 0, 0, 0]

outputs = model.get_outputs()
output = outputs[0]
print("Name:", output.name)
print("Type:", output.type)
print("Shape:", output.shape)

start_time = time.time()
outputs = model.run(["output0"], {"images": input_img})
end_time = time.time()

inference_time_ms = (end_time - start_time) * 1000  
print(f"Inference time for frame: {inference_time_ms:.2f} ms")

print(len(outputs))

output = outputs[0]
print(output.shape)

print(output.shape)
for i in range(output.shape[1]):
    print(f"Detection {i}: {output[:, i]}")
output = output[0].transpose()
print(output.shape)
row = output[0]

def parse_row(row):
    xc, yc, w, h = row[:4]
    x1 = (xc - w / 2) / 640 * img_width
    y1 = (yc - h / 2) / 640 * img_height
    x2 = (xc + w / 2) / 640 * img_width
    y2 = (yc + h / 2) / 640 * img_height
    prob = row[4:].max()
    class_id = row[4:].argmax()
    label = yolo_classes[class_id]
    return [x1, y1, x2, y2, label, prob]

    #IoU
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
#conf > 0.3
boxes = [row for row in [parse_row(row) for row in output] if row[5] > 0.4]
print(len(boxes))

#Non-Maximum Suppression
def nms(boxes, threshold):
    sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    selected_boxes = []
    while len(sorted_boxes) > 0:
        selected_boxes.append(sorted_boxes[0])
        sorted_boxes = [box for box in sorted_boxes if iou(box, sorted_boxes[0]) < threshold]
    return selected_boxes

#NMS with threshold 0.7
nms_boxes = nms(boxes, 0.7)

for box in nms_boxes:
    x1, y1, x2, y2, class_id, prob = box
    cv.rectangle(img_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    print("class", class_id, f"prob: {prob:.2f}")

cv.imshow("Image with Bounding Boxes", img_with_boxes)
cv.waitKey(0)
cv.destroyAllWindows()