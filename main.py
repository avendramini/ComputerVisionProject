from utils import *

images_path="dataset/train/images"
labels_path="dataset/train/labels"
dataset = load_yolo_dataset(images_path, labels_path)

class_names= ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']
class_colors = [
    (255, 0, 0),       # blue
    (238, 130, 238),   # violet
    (235, 206, 135),   # skyblue
    (255, 0, 255),     # fuchsia
    (211, 0, 148),     # darkviolet
    (0, 128, 0),       # green
    (42, 42, 165),     # brown
    (203, 192, 255),   # pink
    (0, 255, 255),     # yellow
    (144, 238, 144),   # lightgreen
    (0, 0, 255),       # red
    (255, 0, 0),       # blue
    (0, 165, 255),     # orange
    (255, 255, 0)      # cyan
]
show_image_with_labels(dataset[0][0], dataset[0][1], class_names, class_colors)