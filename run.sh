#!/bin/bash

# =============================================================================
# CONFIGURAZIONE PIPELINE
# Modifica qui i valori per configurare l'esecuzione
# =============================================================================

# --- Base ---
CAMERAS="2 4 13"                # ID delle camere da processare (separate da spazio)
LABELS_DIR="dataset/infer_video" # Directory dove leggere/salvare le labels
INTERPOLATE=true                # true: attiva interpolazione, false: disattiva (--no-interpolate)
MAX_GAP=10                      # Numero massimo di frame da interpolare
ANCHOR="center"                 # 'center' o 'bottom_center'
MIN_CAMS=2                      # Minimo numero di camere per triangolare un punto
VISUALIZE=true                  # true: apre il visualizzatore 3D alla fine

# --- Inferenza YOLO ---
INFER_VIDEOS=true               # true: esegue inferenza su video, false: usa labels esistenti
VIDEO_DIR="dataset/video"       # Directory contenente i video (outX.mp4)
INFER_IMAGES_DIR=""             # Se specificato, esegue inferenza su questa cartella di immagini invece che sui video
MODEL="weights/best.pt" # Path del modello YOLO (.pt)
DEVICE="0"                      # Device: 'cpu', '0', '0,1', 'auto'
CONF_THRES=0.25                 # Soglia confidenza detection
IMGSZ=1920                    # Dimensione input inferenza (es. 1920, 1280)

# --- Rettificazione ---
RECTIFY=true                    # true: attiva correzione distorsione, false: disattiva (--no-rectify)

# --- Tracking 2D (Opzionale) ---
TRACK2D=true                   # true: abilita tracking 2D
TRACK_IOU=0.0                   # Soglia IoU per tracking
TRACK_MAX_AGE=15                # Max frame persi prima di chiudere traccia
TRACKS_OUT_DIR="runs/tracks"    # Output directory tracce
SAVE_TRACKS=true               # true: salva file JSON delle tracce

# --- Valutazione (Opzionale) ---
EVALUATE_LABELS=true            # true: calcola metriche IoU contro GT
EVAL_GT_DIR="dataset/val/labels" # Directory Ground Truth
EVAL_OUT_DIR="runs/eval"        # Output directory metriche

# =============================================================================
# COSTRUZIONE COMANDO
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

# Inferenza args
if [ "$INFER_VIDEOS" = true ]; then
    CMD="$CMD --infer-videos"
fi

if [ -n "$INFER_IMAGES_DIR" ]; then
    CMD="$CMD --infer-images-dir $INFER_IMAGES_DIR"
fi

CMD="$CMD --video-dir $VIDEO_DIR"
CMD="$CMD --model $MODEL"
CMD="$CMD --device $DEVICE"
CMD="$CMD --conf-thres $CONF_THRES"
CMD="$CMD --imgsz $IMGSZ"

# Rettificazione args
if [ "$RECTIFY" = false ]; then
    CMD="$CMD --no-rectify"
fi

# Tracking args
if [ "$TRACK2D" = true ]; then
    CMD="$CMD --track2d"
fi
CMD="$CMD --track-iou $TRACK_IOU"
CMD="$CMD --track-max-age $TRACK_MAX_AGE"
CMD="$CMD --tracks-out-dir $TRACKS_OUT_DIR"

if [ "$SAVE_TRACKS" = true ]; then
    CMD="$CMD --save-tracks"
fi

# Valutazione args
if [ "$EVALUATE_LABELS" = true ]; then
    CMD="$CMD --evaluate-labels"
fi
CMD="$CMD --eval-gt-dir $EVAL_GT_DIR"
CMD="$CMD --eval-out-dir $EVAL_OUT_DIR"

# =============================================================================
# ESECUZIONE
# =============================================================================

echo "Esecuzione comando:"
echo "$CMD"
echo "----------------------------------------------------------------"

$CMD