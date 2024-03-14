import cv2
import os

# This is the dictionary you need to add that correspond to your model 
class_names_dict = {0: 'Ambulance', 1: 'Firefighter', 2: 'Traffic_jam', 3: 'Car', 4: 'Police'}

def draw_boxes_on_image(image_path, labels_path, output_path):
    # Read the image
    img = cv2.imread(image_path)

    # Read the labels from the text file
    with open(labels_path, 'r') as txt_file:
        lines = txt_file.readlines()

    for line in lines:
        # Parse the line to get class_id and normalized coordinates (xywhn)
        parts = line.strip().split()
        if len(parts) != 5:  # Ensure five elements in the line sense the txt file contain {class_id} {box[0]} {box[1]} {box[2]} {box[3]
            continue

        class_id, x_center, y_center, width, height = map(float, parts)

        # Convert normalized coordinates to pixel values
        img_height, img_width, _ = img.shape
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)

        # Draw bounding box on the image
        color = (0, 0, 255)  # RED color
        thickness = 2
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # Adding the class name above the rectangle with red text background
        class_name = class_names_dict.get(int(class_id), "Unknown")
        text = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate the width and height of the red rectangle based on text size
        rect_width = text_size[0] + 10  
        rect_height = text_size[1] + 5  

        # Draw a red rectangle as the background for the text
        img = cv2.rectangle(img, (x_min, y_min - rect_height), (x_min + rect_width, y_min), (0, 0, 255), thickness=cv2.FILLED)

        text_position = (x_min + 5, y_min - 5)
        img = cv2.putText(img, text, text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Save the output image
    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + '_with_boxes.jpg'
    cv2.imwrite(output_path, img)

def process_images(images_directory, labels_directory, output_directory):
    for image_file in os.listdir(images_directory):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
            image_path = os.path.join(images_directory, image_file)
            labels_path = os.path.join(labels_directory, os.path.splitext(image_file)[0] + '.txt')
            output_path = os.path.join(output_directory, os.path.splitext(image_file)[0] + '_with_boxes.jpg')
            draw_boxes_on_image(image_path, labels_path, output_path)

if __name__ == "__main__":
    images_directory = '/home/ziad/Documents/test/images/'
    labels_directory = '/home/ziad/Documents/test/labels/'
    output_directory = '/home/ziad/Documents/test/Label_validation/'

    process_images(images_directory, labels_directory, output_directory)
