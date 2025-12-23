import os
from ultralytics import YOLO
import yaml
import torch
def create_dataset_yaml(train_path, val_path, class_names, save_path="dataset.yaml"):
    """Create dataset configuration file for YOLO training.

    Fix: use a stable root (dataset/) and relative subpaths expected by Ultralytics.
    """
    # Normalize provided paths
    train_abs = os.path.abspath(train_path)
    val_abs = os.path.abspath(val_path)

    # Heuristic root = common parent of train/val up to 'dataset'
    root_marker = 'dataset'
    parts = train_abs.split(os.sep)
    if root_marker in parts:
        idx = parts.index(root_marker)
        root = os.sep.join(parts[:idx + 1])  # .../dataset
    else:
        # fallback: parent dir of train images
        root = os.path.dirname(os.path.dirname(train_abs))

    # Build relative paths from root
    def rel_from_root(p):
        if p.startswith(root):
            rel = os.path.relpath(p, root).replace('\\', '/')
            return rel
        return p.replace('\\', '/')

    train_rel = rel_from_root(train_abs)
    val_rel = rel_from_root(val_abs)

    dataset_config = {
        'path': root.replace('\\', '/'),
        'train': train_rel,
        'val': val_rel,
        'nc': len(class_names),
        'names': class_names
    }

    with open(save_path, 'w') as f:
        yaml.dump(dataset_config, f, sort_keys=False)
    print(f"[INFO] Creato file dataset YAML: {save_path}")
    print(f"[INFO] Contenuto: {dataset_config}")
    return save_path

def fine_tune_yolo(model_size='yolov8n.pt', dataset_yaml='dataset.yaml', epochs=100, imgsz=640):
    """Fine-tune YOLO model on custom dataset"""
    
    # Load pre-trained YOLO model
    model = YOLO(model_size)
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        patience=50,
        batch=2,
        device=0  # Use GPU if available
        ,cls=1.5
    )
    
    return model, results

if __name__ == "__main__":
    # Configuration
    TRAIN_IMAGES_PATH = "./dataset/train/images"
    VAL_IMAGES_PATH = "./dataset/val/images"
    CLASS_NAMES = ['Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2', 'Refree_F', 'Refree_M', 'White_13', 'White_16', 'White_25', 'White_27', 'White_34']

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
        (0, 165, 255),     # orange
        (255, 255, 0)      # cyan
    ]
    
    # Create dataset configuration
    dataset_yaml = create_dataset_yaml(
        train_path=TRAIN_IMAGES_PATH,
        val_path=VAL_IMAGES_PATH,
        class_names=CLASS_NAMES
    )
    
    # Fine-tune the model
    model, results = fine_tune_yolo(
        model_size='yolo12n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        dataset_yaml=dataset_yaml,
        epochs=100,
        imgsz=1920
    )
    
    # Save the trained model
    model.save('last_runv81920.pt')
    
    print("Fine-tuning completed!")