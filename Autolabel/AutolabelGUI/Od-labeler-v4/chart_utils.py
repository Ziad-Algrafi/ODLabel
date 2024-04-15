import os
import cv2 as cv
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def update_input_chart(images_folder, chart_colors):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png', 'jfif'))]
    resolutions = []
    blurred_images = 0
    grayed_images = 0
    black_white_images = 0
    corrupted_images = 0
    color_spaces = {}
    
    for image_file in image_files:
        img_path = os.path.join(images_folder, image_file)
        try:
            img = cv.imread(img_path)
            if img is not None:
                resolutions.append(img.shape[:2])
                
                # Check for blurred images
                if cv.Laplacian(img, cv.CV_64F).var() < 100:
                    blurred_images += 1
                
                # Check for grayscale images
                if len(img.shape) < 3 or img.shape[2] == 1:
                    grayed_images += 1
                
                # Check for black and white images
                if len(img.shape) == 3 and np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2]):
                    black_white_images += 1
                
                # Check color space
                color_space = img.shape[2] if len(img.shape) == 3 else 1
                color_spaces[color_space] = color_spaces.get(color_space, 0) + 1
        except:
            corrupted_images += 1
    
    num_images = len(resolutions)
    if resolutions:
        avg_resolution = tuple(map(int, np.mean(resolutions, axis=0)))
        min_resolution = tuple(map(int, np.min(resolutions, axis=0)))
        max_resolution = tuple(map(int, np.max(resolutions, axis=0)))
    else:
        avg_resolution = (0, 0)
        min_resolution = (0, 0)
        max_resolution = (0, 0)
    
    figure = Figure(figsize=(10, 8), dpi=100)
    figure.subplots_adjust(hspace=0.4, wspace=0.4)
    figure.set_facecolor('#1c1c1c')  # Set background color to dark gray
    
    # Plot number of images and image format distribution
    ax1 = figure.add_subplot(221)
    formats = [os.path.splitext(f)[1].lower() for f in image_files]
    format_counts = {}
    for fmt in formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    num_images_bar = ax1.bar(['Images'], [num_images], color=chart_colors['images'], width=0.4)
    format_bars = ax1.bar(format_counts.keys(), format_counts.values(), color=chart_colors['format'], width=0.4)
    ax1.set_ylabel('Number of Images', color='white')
    ax1.set_title('Input Images and Format Distribution', color='white')
    ax1.tick_params(axis='x', colors='white', rotation=45)
    ax1.tick_params(axis='y', colors='white')
    ax1.legend((num_images_bar, format_bars), ('Images', 'Format'), loc='upper right', facecolor='none', edgecolor='none', fontsize='small', labelcolor='white')
    
    # Plot resolution information
    ax2 = figure.add_subplot(222)
    ax2.bar(['Max', 'Min', 'Avg'], [max_resolution[0], min_resolution[0], avg_resolution[0]], color=chart_colors['resolution'])
    ax2.set_ylabel('Resolution', color='white')
    ax2.set_title('Image Resolution', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    
    # Plot image quality information
    ax3 = figure.add_subplot(223)
    ax3.bar(['Blurred', 'Grayscale', 'B&W', 'Corrupted'], [blurred_images, grayed_images, black_white_images, corrupted_images], color=chart_colors['quality'])
    ax3.set_ylabel('Number of Images', color='white')
    ax3.set_title('Image Quality', color='white')
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')
    
    # Plot color space distribution
    ax4 = figure.add_subplot(224, projection='3d')
    color_spaces_keys = list(color_spaces.keys())
    x = np.arange(len(color_spaces_keys))
    y = [0] * len(color_spaces_keys)
    z = list(color_spaces.values())

    dx = np.ones_like(x)
    dy = np.ones_like(x)
    dz = z

    ax4.bar3d(x, y, np.zeros_like(z), dx, dy, dz, color=chart_colors['format'])

    ax4.set_xticks(x)
    ax4.set_xticklabels(color_spaces_keys)
    ax4.set_yticks([])
    ax4.set_zlabel('Number of Images', color='white')
    ax4.set_title('Color Space Distribution', color='white')
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='z', colors='white')
    
    return figure

def update_output_charts(train_detections, val_detections):
    train_object_counts = {}
    train_confidences = []
    train_labels_sizes = {}
    for detection in train_detections:
        class_name = detection[5]
        confidence = detection[6]
        train_object_counts[class_name] = train_object_counts.get(class_name, 0) + 1
        train_confidences.append(confidence)
        train_labels_sizes[class_name] = train_labels_sizes.get(class_name, 0) + 1
    
    val_object_counts = {}
    val_confidences = []
    val_labels_sizes = {}
    for detection in val_detections:
        class_name = detection[5]
        confidence = detection[6]
        val_object_counts[class_name] = val_object_counts.get(class_name, 0) + 1
        val_confidences.append(confidence)
        val_labels_sizes[class_name] = val_labels_sizes.get(class_name, 0) + 1
    
    # Count of Detected Objects
    fig_count, ax_count = plt.subplots(figsize=(4, 3)) 
    x = np.arange(len(train_object_counts))
    width = 0.35
    ax_count.bar(x - width/2, train_object_counts.values(), width, label='Train')
    ax_count.bar(x + width/2, val_object_counts.values(), width, label='Val')
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(train_object_counts.keys())
    ax_count.set_xlabel('Class')
    ax_count.set_ylabel('Count')
    ax_count.set_title('Count of Detected Objects')
    ax_count.legend()
    
    # Detection Confidence Histogram
    fig_conf, ax_conf = plt.subplots(figsize=(5, 5))  
    ax_conf.hist(train_confidences, bins=20, alpha=0.5, label='Train')
    ax_conf.hist(val_confidences, bins=20, alpha=0.5, label='Val')
    ax_conf.set_xlabel('Confidence')
    ax_conf.set_ylabel('Frequency')
    ax_conf.set_title('Detection Confidence Histogram')
    ax_conf.legend()
    
    # Labels Correlogram
    fig_labels, ax_labels = plt.subplots(figsize=(4, 3)) 
    train_labels_sizes = dict(sorted(train_labels_sizes.items(), key=lambda x: x[1], reverse=True))
    val_labels_sizes = dict(sorted(val_labels_sizes.items(), key=lambda x: x[1], reverse=True))
    train_labels = list(train_labels_sizes.keys())
    train_sizes = list(train_labels_sizes.values())
    val_labels = list(val_labels_sizes.keys())
    val_sizes = list(val_labels_sizes.values())
    y_pos = np.arange(len(train_labels))
    ax_labels.barh(y_pos - 0.2, train_sizes, 0.4, label='Train')
    ax_labels.barh(y_pos + 0.2, val_sizes, 0.4, label='Val')
    ax_labels.set_yticks(y_pos)
    ax_labels.set_yticklabels(train_labels)
    ax_labels.invert_yaxis()
    ax_labels.set_xlabel('Size')
    ax_labels.set_ylabel('Label')
    ax_labels.set_title('Labels Correlogram')
    ax_labels.legend()
    
    return fig_count, fig_conf, fig_labels