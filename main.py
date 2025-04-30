from utils import *
from tracking import *
from ultralytics import YOLO
images_path="dataset/train/images"
labels_path="dataset/train/labels"
camera_datasets=split_and_sort_by_camera(images_path, labels_path) #


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
    (0, 165, 255),     # orange
    (255, 255, 0)      # cyan
]

path_videos=['dataset/video/out4.mp4', 'dataset/video/out13.mp4']
frame_videos={
    4:extract_frames_from_video(path_videos[0]),
    13:extract_frames_from_video(path_videos[1])
}

#show_side_by_side(camera_datasets[4][0][0],frame_videos[4][2], max_size=800)
#show_side_by_side(camera_datasets[13][0][0],frame_videos[13][2], max_size=800)

#Il terzo frame è il primo etichettato, poi si fa +5 ogni volta
#camera_datasets[13][0][0]->frame_videos[13][2]
#camera_datasets[13][1][0]->frame_videos[13][7]
#...

#print("Extracted Image:", image)
#print("YOLO Detected Labels:", labels)
#show_image_with_labels(image,labels)

model = YOLO("yolov8x.pt")  # o yolov8s.pt, yolov8m.pt, ecc.

offset = 2
cameras=[13]
prev_labels={4:[],13:[]}
for cam in cameras:

    prev_labels[cam].append(camera_datasets[cam][0][1])
    for i in range(offset+1,len(frame_videos[cam])):
        #usa old_labels e old_frame per fare tracking
        new_labels=find_labels(new_frame=frame_videos[cam][i],old_frame=frame_videos[cam][i-1],prev_labels=prev_labels[cam][-1],model=model)
        show_image_with_labels(frame_videos[cam][i],new_labels,class_names=class_names,class_colors=class_colors,max_size=800)
        prev_labels[cam].append(new_labels)
        if i==5:
            break
loss={4:0,13:0}
for cam in cameras:
    for i in range(len(camera_datasets[cam])):
        loss[cam]+=compare_labels(camera_datasets[cam][i][1],prev_labels[cam][offset+i*5])
    
    print(f"Loss for camera {cam}: {loss[cam]}")

