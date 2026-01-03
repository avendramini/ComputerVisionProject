#!/bin/bash

# =============================================================================
# PIPELINE CONFIGURATION
# Modify values here to configure execution
# =============================================================================

# --- Base ---
CAMERAS="2 4 13"                # IDs of cameras to process (space-separated)
LABELS_DIR="dataset/infer_video" # Directory to read/save labels
INTERPOLATE=true                # true: enable interpolation, false: disable (--no-interpolate)
MAX_GAP=2                  # Maximum number of frames to interpolate
ANCHOR="center"                 # 'center' or 'bottom_center'
MIN_CAMS=2                      # Minimum number of cameras to triangulate a point
VISUALIZE=true                  # true: opens 3D visualizer at the end

# --- YOLO Inference ---
INFER_VIDEOS=true               # true: run inference on videos, false: use existing labels
VIDEO_DIR="dataset/video"       # Directory containing videos (outX.mp4) - used only if INFER_VIDEOS=true
INFER_IMAGES_DIR=""             # If specified, run inference on this image folder instead of videos
MODEL="weights/best.pt" # Path to YOLO model (.pt) - used only if INFER_VIDEOS=true or INFER_IMAGES_DIR is specified
DEVICE="0"                      # Device: 'cpu', '0', '0,1', 'auto' - used only if INFER_VIDEOS=true
CONF_THRES=0.3                 # Detection confidence threshold - used only if INFER_VIDEOS=true
IMGSZ=1920                      # Inference input size (e.g. 1920, 1280) - used only if INFER_VIDEOS=true

# --- Rectification ---
RECTIFY=true                    # true: enable distortion correction, false: disable (--no-rectify)

# --- Labels and Video Saving (without 2D tracking) ---
TRACKS_OUT_DIR="runs/tracks"   # Output directory for labels and videos
SAVE_TRACKS=true               # true: save JSON labels and generate visualization video

# --- Evaluation (Optional) ---
EVALUATE_LABELS=true            # true: calculate IoU metrics against GT
EVAL_GT_DIR="dataset/val/labels" # Ground Truth directory
EVAL_OUT_DIR="runs/eval"        # Metrics output directory

# =============================================================================
# COMMAND CONSTRUCTION
# =============================================================================

CMD="python pipeline.py"

# Base args
CMD="$CMD --cameras $CAMERAS"
CMD="$CMD --labels-dir $LABELS_DIR"
CMD="$CMD --max-gap $MAX_GAP"
CMD="$CMD --anchor $ANCHOR"
CMD="$CMD --min-cams $MIN_CAMS"

if [ "$INTERPOLATE" = false ]; then
    CMD="$CMD --no-interpolate"
fi

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

# Inference args
if [ "$INFER_VIDEOS" = true ]; then
    CMD="$CMD --infer-videos"
    CMD="$CMD --video-dir $VIDEO_DIR"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --device $DEVICE"
    CMD="$CMD --conf-thres $CONF_THRES"
    CMD="$CMD --imgsz $IMGSZ"
fi

if [ -n "$INFER_IMAGES_DIR" ]; then
    CMD="$CMD --infer-images-dir $INFER_IMAGES_DIR"
    CMD="$CMD --model $MODEL"
    CMD="$CMD --device $DEVICE"
    CMD="$CMD --conf-thres $CONF_THRES"
    CMD="$CMD --imgsz $IMGSZ"
fi

# Rectification args
if [ "$RECTIFY" = false ]; then
    CMD="$CMD --no-rectify"
fi

# Labels saving args
CMD="$CMD --tracks-out-dir $TRACKS_OUT_DIR"

if [ "$SAVE_TRACKS" = true ]; then
    CMD="$CMD --save-tracks"
fi

# Evaluation args
if [ "$EVALUATE_LABELS" = true ]; then
    CMD="$CMD --evaluate-labels"
    CMD="$CMD --eval-gt-dir $EVAL_GT_DIR"
    CMD="$CMD --eval-out-dir $EVAL_OUT_DIR"
fi

# =============================================================================
# EXECUTION
# =============================================================================

echo "Executing command:"
echo "$CMD"
echo "----------------------------------------------------------------"

$CMD