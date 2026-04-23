#!/bin/bash
# Beim ersten Start: Pre-trained Modell herunterladen falls nicht vorhanden

MODEL_DIR="/app/models"
MODEL_FILE="$MODEL_DIR/yolov8l-1024-seg.pt"
TRAINED_MODEL="$MODEL_DIR/timbervision_yolov8/weights/best.pt"

if [ ! -f "$MODEL_FILE" ] && [ ! -f "$TRAINED_MODEL" ]; then
    echo "============================================"
    echo "  TimberVision: Modell nicht gefunden."
    echo "  Lade Pre-trained Modell herunter..."
    echo "============================================"
    cd /app/scripts && python3 train.py download-model --model-type seg
    if [ $? -ne 0 ]; then
        echo "WARNUNG: Modell-Download fehlgeschlagen. Server startet ohne Modell."
        echo "Manuell herunterladen: python3 train.py download-model"
    fi
else
    echo "TimberVision Modell gefunden. Starte Server..."
fi

echo "Starte TimberVision Inference Server auf Port 8300..."
exec uvicorn inference_server:app --host 0.0.0.0 --port 8300
