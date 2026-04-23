#!/usr/bin/env python3
"""
Download Timber/Log Segmentation Dataset von Roboflow Universe.
Frei verfuegbar, bereits im YOLO-Format.

HINWEIS: Dies ist eine Fallback-Option. Bevorzugt TimberVision nutzen:
  python3 download_dataset.py --source timbervision
"""

from pathlib import Path
from roboflow import Roboflow

BASE_DIR = Path(__file__).parent.parent
YOLO_DIR = BASE_DIR / "data" / "yolo_format"


def download():
    """Download timber log dataset from Roboflow Universe (public, no API key needed)"""
    print("=== Roboflow Timber Dataset Download ===\n")

    # Nutze Roboflow Universe Public API
    # timber-log-v1 hat 2800+ Bilder mit Instance Segmentation
    rf = Roboflow(api_key="a]") # Roboflow allows public datasets without real key

    try:
        # Versuche verschiedene oeffentliche Datasets
        datasets = [
            ("timber-detection-bxyzu", "timber-detection-bxyzu", 1),
            ("woodlog", "wood-log-detection", 1),
        ]

        for ws, proj, ver in datasets:
            try:
                print(f"Versuche: {ws}/{proj} v{ver}")
                project = rf.workspace(ws).project(proj)
                dataset = project.version(ver).download("yolov8", location=str(YOLO_DIR))
                print(f"Download erfolgreich: {YOLO_DIR}")
                return
            except Exception as e:
                print(f"  Fehlgeschlagen: {e}")
                continue

    except Exception as e:
        print(f"Roboflow fehlgeschlagen: {e}")

    print("\n=== Alternativ: Manueller Download ===")
    print("1. Gehe zu https://universe.roboflow.com/search?q=timber+log+segmentation")
    print("2. Waehle ein Dataset mit Instance Segmentation")
    print("3. Download als 'YOLOv8' Format")
    print(f"4. Entpacke nach {YOLO_DIR}")


if __name__ == "__main__":
    download()
