from utils import *
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
offset=2
old_labels=camera_datasets[13][0][1]
old_frame=frame_videos[13][offset]
for i in range(offset+1,len(frame_videos[13])):
    #usa old_labels e old_frame per fare tracking
    new_frame=frame_videos[13][i]






