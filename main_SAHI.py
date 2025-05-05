from utils import *
from tracking import *
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
images_path="dataset/train/images"
labels_path="dataset/train/labels"
camera_datasets=split_and_sort_by_camera(images_path, labels_path) #



detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path='yolov12x.pt',
    confidence_threshold=0.7,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)


# Convert the BGR image to RGB
ok = cv2.cvtColor(camera_datasets[13][0][0], cv2.COLOR_BGR2RGB)
result = get_sliced_prediction(ok, detection_model,slice_height=640,slice_width=640, overlap_height_ratio=0.1, overlap_width_ratio=0.1)

result.export_visuals(export_dir="demo_data/", hide_conf=True)


object_prediction_list = result.object_prediction_list

import matplotlib.pyplot as plt

# Extract bounding boxes for balls and people (assuming class IDs for balls and people are known, e.g., 0 for person, 1 for ball)
selected_classes = [0, 32]  # Replace with actual class IDs for person and ball
filtered_predictions = [
    pred for pred in object_prediction_list if pred.category.id in selected_classes
]

# Plot the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(ok)
# Add bounding boxes to the plot
for pred in filtered_predictions:
    bbox = pred.bbox
    x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
    plt.gca().add_patch(
        plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
    )
    plt.text(
        x_min,
        y_min - 5,
        pred.category.name,
        color="red",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5),
    )

plt.axis("off")
plt.show()



