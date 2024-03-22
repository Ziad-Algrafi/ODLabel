import onnx

model_path = "custom_yolov8x-worldv2.onnx"
model = onnx.load(model_path)

desired_classes_with_ids = {
    0: 'person', 1: 'bicycle', 2: 'car'
    
}

custom_names_str = ", ".join([f"{key}: '{value}'" for key, value in desired_classes_with_ids.items()])
custom_names_str = "{" + custom_names_str + "}"

found = False
for prop in model.metadata_props:
    if prop.key == "names":
        prop.value = custom_names_str
        found = True
        break
if not found:
    model.metadata_props.add(key="names", value=custom_names_str)

onnx.save(model, "custom_yolov8x-worldv2_modified.onnx")
