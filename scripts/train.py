#!/usr/bin/env python3
"""
YOLOv8 Training fuer Holzstamm-Segmentierung.

Nutzt TimberVision Dataset (WACV 2025) im YOLO-Format.
Klassen: cut (Schnittflaeche), side (Seitenflaeche), trunk (Gesamtstamm)

Training-Optionen:
  1. Von Grund auf mit TimberVision trainieren (empfohlen fuer beste Ergebnisse)
  2. TimberVision Pre-trained Modelle direkt verwenden (schnellster Weg)
  3. Fine-Tuning auf eigenem Dataset basierend auf TimberVision-Modell
"""

import sys
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "yolo_format"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"

MODELS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# TimberVision Pre-trained Modelle (GitHub Releases)
TV_MODEL_BASE = "https://github.com/timbervision/timbervision/releases/download/v1.0.0"
TV_MODELS = {
    'seg': 'yolov8l-1024-seg.pt',   # Instance Segmentation (Large, 1024px)
    'obb': 'yolov8l-1024-obb.pt',   # Oriented Object Detection (Large, 1024px)
}

# Training-Konfiguration
PROJECT_NAME = 'timbervision_yolov8'


def download_pretrained(model_type='seg'):
    """
    Lade vortrainiertes TimberVision-Modell herunter.
    Kann direkt fuer Inference oder als Basis fuer Fine-Tuning genutzt werden.
    """
    model_name = TV_MODELS.get(model_type)
    if not model_name:
        print(f"FEHLER: Unbekannter Modelltyp '{model_type}'. Verfuegbar: {list(TV_MODELS.keys())}")
        sys.exit(1)

    model_path = MODELS_DIR / model_name
    if model_path.exists():
        print(f"Modell bereits vorhanden: {model_path}")
        return model_path

    url = f"{TV_MODEL_BASE}/{model_name}"
    print(f"Lade TimberVision Modell: {model_name}")
    print(f"  URL: {url}")

    try:
        urlretrieve(url, str(model_path), reporthook=_download_progress)
        print(f"\n  Gespeichert: {model_path}")
        return model_path
    except (URLError, OSError) as e:
        print(f"\nDownload fehlgeschlagen: {e}")
        print(f"Manuell herunterladen: {url}")
        return None


def _download_progress(block_num, block_size, total_size):
    """Fortschrittsanzeige"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        print(f"\r  {percent:.1f}% ({mb:.0f} MB)", end="", flush=True)


def train(epochs=100, imgsz=640, batch=8, device='cpu', base_model='yolov8n-seg.pt',
          resume=False):
    """
    YOLOv8 Segmentation Training auf TimberVision Dataset.

    Args:
        epochs: Anzahl Trainings-Epochen
        imgsz: Bildgroesse (TimberVision nutzt 1024, aber 640 ist schneller)
        batch: Batch-Groesse (abhaengig von GPU-RAM)
        device: 'cpu', '0' (erste GPU), '0,1' (Multi-GPU)
        base_model: Basis-Modell (yolov8n/s/m/l/x-seg.pt oder TimberVision)
        resume: Training fortsetzen
    """
    dataset_yaml = DATA_DIR / "dataset.yaml"

    if not dataset_yaml.exists():
        print(f"FEHLER: {dataset_yaml} nicht gefunden!")
        print("Zuerst: python3 download_dataset.py")
        sys.exit(1)

    # Modell laden
    if resume:
        last_model = MODELS_DIR / PROJECT_NAME / 'weights' / 'last.pt'
        if not last_model.exists():
            print(f"FEHLER: {last_model} nicht gefunden fuer Resume!")
            sys.exit(1)
        model = YOLO(str(last_model))
        print(f"Setze Training fort von: {last_model}")
    else:
        model = YOLO(base_model)

    print(f"\n{'=' * 50}")
    print(f"  Training Start")
    print(f"{'=' * 50}")
    print(f"  Modell:     {base_model}")
    print(f"  Dataset:    {dataset_yaml}")
    print(f"  Epochen:    {epochs}")
    print(f"  Bildgroesse: {imgsz}")
    print(f"  Batch:      {batch}")
    print(f"  Device:     {device}")
    print(f"  Klassen:    cut, side, trunk")
    print(f"{'=' * 50}\n")

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR),
        name=PROJECT_NAME,
        exist_ok=True,
        # Augmentation (optimiert fuer Holzstapel-Bilder)
        augment=True,
        hsv_h=0.015,       # Farbton leicht variieren
        hsv_s=0.7,         # Saettigung variieren (Rinde, Schnittflaeche)
        hsv_v=0.4,         # Helligkeit (Schatten im Wald)
        degrees=10,         # Leichte Rotation
        translate=0.1,      # Verschiebung
        scale=0.5,          # Skalierung
        flipud=0.5,         # Vertikal spiegeln
        fliplr=0.5,         # Horizontal spiegeln
        mosaic=1.0,         # Mosaic Augmentation
        mixup=0.1,          # MixUp
        # Training
        patience=20,        # Early Stopping nach 20 Epochen ohne Verbesserung
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = MODELS_DIR / PROJECT_NAME / 'weights' / 'best.pt'
    print(f"\n{'=' * 50}")
    print(f"  Training abgeschlossen!")
    print(f"  Best Model: {best_path}")
    print(f"{'=' * 50}")

    return results


def finetune(epochs=50, imgsz=640, batch=4, device='cpu'):
    """
    Fine-Tuning basierend auf TimberVision Pre-trained Modell.
    Ideal wenn wenige eigene Bilder vorhanden sind.
    """
    print("=== Fine-Tuning mit TimberVision Basis-Modell ===\n")

    model_path = download_pretrained('seg')
    if not model_path:
        print("Kann Pre-trained Modell nicht laden.")
        print("Starte stattdessen Training von Grund auf...")
        return train(epochs=epochs, imgsz=imgsz, batch=batch, device=device)

    return train(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        base_model=str(model_path),
    )


def export_models():
    """Exportiere trainiertes Modell als ONNX und TFLite fuer Mobile"""
    best_model = MODELS_DIR / PROJECT_NAME / 'weights' / 'best.pt'

    if not best_model.exists():
        print(f"FEHLER: {best_model} nicht gefunden! Zuerst trainieren.")
        sys.exit(1)

    model = YOLO(str(best_model))
    print(f"\n=== Export: {best_model.name} ===\n")

    # ONNX Export
    print("1. ONNX Export...")
    model.export(format='onnx', imgsz=640, simplify=True)
    print("   ONNX exportiert")

    # TFLite Export
    print("2. TFLite Export...")
    try:
        model.export(format='tflite', imgsz=640)
        print("   TFLite exportiert")
    except Exception as e:
        print(f"   TFLite Export fehlgeschlagen: {e}")
        print("   ONNX kann alternativ verwendet werden.")

    # Kopiere Exports in exports/ Verzeichnis
    weights_dir = MODELS_DIR / PROJECT_NAME / 'weights'
    for ext in ['*.onnx', '*.tflite']:
        for f in weights_dir.glob(ext):
            dst = EXPORTS_DIR / f.name
            shutil.copy2(f, dst)
            print(f"   Kopiert: {dst}")


def validate():
    """Validiere das trainierte Modell"""
    best_model = MODELS_DIR / PROJECT_NAME / 'weights' / 'best.pt'

    if not best_model.exists():
        print(f"FEHLER: {best_model} nicht gefunden!")
        sys.exit(1)

    model = YOLO(str(best_model))
    dataset_yaml = DATA_DIR / "dataset.yaml"

    print(f"\n=== Validation: {best_model.name} ===\n")
    results = model.val(data=str(dataset_yaml))

    print(f"\n{'=' * 40}")
    print(f"  Segmentation Metriken")
    print(f"{'=' * 40}")
    print(f"  mAP50:    {results.seg.map50:.4f}")
    print(f"  mAP50-95: {results.seg.map:.4f}")

    # Pro-Klasse Metriken wenn verfuegbar
    if hasattr(results.seg, 'ap_class_index') and results.seg.ap_class_index is not None:
        class_names = {0: "cut", 1: "side", 2: "trunk"}
        print(f"\n  Pro Klasse:")
        for i, cls_idx in enumerate(results.seg.ap_class_index):
            name = class_names.get(int(cls_idx), f"class_{cls_idx}")
            print(f"    {name:8s}: mAP50={results.seg.ap50[i]:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv8 Holzstamm-Training (TimberVision)')
    parser.add_argument('action',
                        choices=['train', 'finetune', 'export', 'validate', 'download-model'],
                        default='train', nargs='?',
                        help='Aktion (default: train)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Bildgroesse (640 schnell, 1024 wie TimberVision Original)')
    parser.add_argument('--device', default='cpu',
                        help='Device: cpu, 0 (GPU), 0,1 (Multi-GPU)')
    parser.add_argument('--model', default='yolov8n-seg.pt',
                        help='Basis-Modell (n=nano, s=small, m=medium, l=large)')
    parser.add_argument('--resume', action='store_true',
                        help='Training fortsetzen')
    parser.add_argument('--model-type', choices=['seg', 'obb'], default='seg',
                        help='TimberVision Modelltyp fuer Download')
    args = parser.parse_args()

    if args.action == 'train':
        train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device=args.device, base_model=args.model, resume=args.resume)
    elif args.action == 'finetune':
        finetune(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                 device=args.device)
    elif args.action == 'export':
        export_models()
    elif args.action == 'validate':
        validate()
    elif args.action == 'download-model':
        path = download_pretrained(args.model_type)
        if path:
            print(f"\nModell bereit: {path}")
