#!/usr/bin/env python3
"""
Dataset-Management fuer WVB Holzmengen-Erkennung.

Primaere Datenquelle: TimberVision (WACV 2025)
- >2000 Bilder aus Oesterreich mit 51.000+ Stamm-Komponenten
- Instance Segmentation + Oriented Object Detection
- Klassen: cut (0), bound (1), side (2), trunk (3)
- Download: Zenodo (4.3 GB ZIP)
- Format: Bereits YOLOv8 Instance-Segmentation kompatibel

Fallback-Optionen:
- Roboflow: python3 download_roboflow.py
- HuggingFace: log-pile Starter-Dataset (wenige Bilder, ohne Labels)
- TimberSeg 1.0: Manuell von Mendeley (220 Bilder, Kanada)

TimberVision Klassen → WVB Mapping:
  cut (0)   = Schnittflaeche (Stirnseite des Stamms)
  bound (1) = Rueckseitige Begrenzung (nicht sichtbar)
  side (2)  = Seitenflaeche (Mantelflaeche/Rinde)
  trunk (3) = Gesamter Stamm (OBB ueber cut+side)
"""

import os
import json
import shutil
import random
import zipfile
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
YOLO_DIR = DATA_DIR / "yolo_format"
TIMBERVISION_DIR = DATA_DIR / "timbervision"

# TimberVision Zenodo Download
ZENODO_URL = "https://zenodo.org/records/14825846/files/timbervision.zip?download=1"
ZENODO_MD5 = "05e676243e1ed50976fb0f1860ed35ee"
ZENODO_ZIP = DATA_DIR / "timbervision.zip"

# TimberVision Original-Klassen
TV_CLASSES = {0: "cut", 1: "bound", 2: "side", 3: "trunk"}

# Mapping fuer unser Training (wir nutzen cut + side, ignorieren bound)
# Fuer Volumen-Berechnung sind cut (Durchmesser) und side (Laenge) relevant
WVB_CLASSES = {0: "cut", 1: "side", 2: "trunk"}
TV_TO_WVB = {0: 0, 2: 1, 3: 2}  # cut→0, side→1, trunk→2 (bound wird ignoriert)


def setup_yolo_dirs():
    """Erstelle YOLO Verzeichnisstruktur"""
    for split in ['train', 'val', 'test']:
        (YOLO_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)


def verify_md5(filepath, expected_md5):
    """Pruefe MD5 Checksumme einer Datei"""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def download_progress(block_num, block_size, total_size):
    """Fortschrittsanzeige fuer urlretrieve"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  {percent:.1f}% ({mb_down:.0f}/{mb_total:.0f} MB)", end="", flush=True)
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\r  {mb_down:.0f} MB heruntergeladen", end="", flush=True)


def download_timbervision():
    """
    Lade TimberVision Dataset von Zenodo herunter.
    4.3 GB ZIP mit Bildern und YOLOv8-Labels.
    """
    print("=== TimberVision Dataset Download (Zenodo) ===\n")
    print("Quelle: https://zenodo.org/records/14825846")
    print("Paper:  WACV 2025 - Steininger et al.")
    print(f"Ziel:   {TIMBERVISION_DIR}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Pruefen ob bereits entpackt
    if TIMBERVISION_DIR.exists() and any(TIMBERVISION_DIR.rglob("*.jpg")):
        img_count = len(list(TIMBERVISION_DIR.rglob("*.jpg")))
        print(f"TimberVision bereits vorhanden ({img_count} Bilder)")
        return True

    # ZIP herunterladen falls noetig
    if ZENODO_ZIP.exists():
        print(f"ZIP bereits vorhanden: {ZENODO_ZIP}")
        print("Pruefe MD5 Checksumme...")
        if verify_md5(ZENODO_ZIP, ZENODO_MD5):
            print("  MD5 OK")
        else:
            print("  MD5 FALSCH - lade erneut herunter...")
            ZENODO_ZIP.unlink()

    if not ZENODO_ZIP.exists():
        print(f"Lade herunter: {ZENODO_URL}")
        print("  (ca. 4.3 GB - das dauert je nach Verbindung)\n")
        try:
            urlretrieve(ZENODO_URL, str(ZENODO_ZIP), reporthook=download_progress)
            print("\n  Download abgeschlossen!")

            print("Pruefe MD5 Checksumme...")
            if not verify_md5(ZENODO_ZIP, ZENODO_MD5):
                print("  WARNUNG: MD5 stimmt nicht ueberein!")
                print("  Datei koennte beschaedigt sein.")
        except (URLError, OSError) as e:
            print(f"\nDownload fehlgeschlagen: {e}")
            print("\nAlternativ manuell herunterladen:")
            print(f"  URL: {ZENODO_URL}")
            print(f"  Speichern als: {ZENODO_ZIP}")
            return False

    # Entpacken
    print(f"\nEntpacke nach {TIMBERVISION_DIR}...")
    try:
        with zipfile.ZipFile(str(ZENODO_ZIP), 'r') as zf:
            zf.extractall(str(DATA_DIR))
        print("  Entpacken abgeschlossen!")

        # Pruefen ob ein Unterverzeichnis erstellt wurde
        if not TIMBERVISION_DIR.exists():
            # Suche nach entpacktem Verzeichnis
            for d in DATA_DIR.iterdir():
                if d.is_dir() and d.name != "yolo_format" and any(d.rglob("*.jpg")):
                    d.rename(TIMBERVISION_DIR)
                    print(f"  Umbenannt: {d.name} → timbervision/")
                    break

        return True
    except zipfile.BadZipFile:
        print("  FEHLER: ZIP-Datei ist beschaedigt!")
        ZENODO_ZIP.unlink()
        return False


def remap_label_line(line, class_mapping):
    """
    Konvertiere eine YOLO-Label-Zeile mit neuem Klassen-Mapping.

    Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    Returns: Konvertierte Zeile oder None wenn Klasse ignoriert wird.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    orig_class = int(parts[0])
    if orig_class not in class_mapping:
        return None  # Klasse wird ignoriert (z.B. bound)

    new_class = class_mapping[orig_class]
    return f"{new_class} " + " ".join(parts[1:])


def convert_timbervision_to_yolo():
    """
    Konvertiere TimberVision Dataset ins WVB YOLO-Format.

    TimberVision hat bereits YOLOv8-kompatible Labels.
    Wir muessen nur:
    1. Klassen remappen (bound entfernen, IDs anpassen)
    2. Splits aus train.txt/val.txt/test.txt uebernehmen
    3. Bilder + Labels in YOLO-Struktur kopieren
    """
    print("\n=== Konvertiere TimberVision → WVB YOLO-Format ===\n")

    if not TIMBERVISION_DIR.exists():
        print(f"FEHLER: {TIMBERVISION_DIR} nicht gefunden!")
        return 0

    # Finde Verzeichnisse
    img_dir = find_subdir(TIMBERVISION_DIR, "images")
    lbl_dir = find_subdir(TIMBERVISION_DIR, "labels")

    if not img_dir or not lbl_dir:
        print("FEHLER: images/ oder labels/ Verzeichnis nicht gefunden!")
        print(f"  Gesucht in: {TIMBERVISION_DIR}")
        return 0

    print(f"Bilder:  {img_dir}")
    print(f"Labels:  {lbl_dir}")

    # Splits laden (train.txt, val.txt, test.txt)
    splits = load_splits(TIMBERVISION_DIR)
    if not splits:
        print("Keine Split-Dateien gefunden. Erstelle 80/10/10 Split...")
        splits = create_random_splits(img_dir)

    setup_yolo_dirs()

    total = 0
    stats = {"train": 0, "val": 0, "test": 0}
    class_counts = {v: 0 for v in WVB_CLASSES.values()}

    for split_name, filenames in splits.items():
        for filename in filenames:
            # Bild finden
            img_src = find_image_file(img_dir, filename)
            if not img_src:
                continue

            # Label finden
            stem = img_src.stem
            lbl_src = find_label_file(lbl_dir, stem)

            # Bild kopieren
            dst_img = YOLO_DIR / 'images' / split_name / img_src.name
            shutil.copy2(img_src, dst_img)

            # Labels konvertieren (Klassen remappen)
            dst_lbl = YOLO_DIR / 'labels' / split_name / f"{stem}.txt"
            if lbl_src and lbl_src.exists():
                converted_lines = convert_label_file(lbl_src, TV_TO_WVB)
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(converted_lines))

                # Statistik
                for line in converted_lines:
                    cls_id = int(line.split()[0])
                    cls_name = WVB_CLASSES.get(cls_id, "unknown")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            else:
                dst_lbl.touch()

            stats[split_name] += 1
            total += 1

    print(f"\nKonvertiert: {total} Bilder")
    for split, count in stats.items():
        print(f"  {split}: {count}")
    print(f"\nKlassen-Verteilung:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count} Instanzen")

    return total


def find_subdir(base, name):
    """Finde ein Unterverzeichnis (auch verschachtelt)"""
    direct = base / name
    if direct.is_dir():
        return direct
    for d in base.rglob(name):
        if d.is_dir():
            return d
    return None


def find_image_file(img_dir, filename):
    """Finde ein Bild anhand des Dateinamens"""
    # Direkt
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        candidate = img_dir / (Path(filename).stem + ext)
        if candidate.exists():
            return candidate
    # Rekursiv
    for match in img_dir.rglob(Path(filename).stem + ".*"):
        if match.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return match
    return None


def find_label_file(lbl_dir, stem):
    """Finde eine Label-Datei anhand des Stems"""
    candidate = lbl_dir / f"{stem}.txt"
    if candidate.exists():
        return candidate
    for match in lbl_dir.rglob(f"{stem}.txt"):
        return match
    return None


def load_splits(base_dir):
    """Lade Train/Val/Test Splits aus .txt Dateien"""
    splits = {}
    for split_name in ['train', 'val', 'test']:
        txt_path = base_dir / f"{split_name}.txt"
        if not txt_path.exists():
            # Suche rekursiv
            for match in base_dir.rglob(f"{split_name}.txt"):
                txt_path = match
                break

        if txt_path.exists():
            with open(txt_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            # Extrahiere Dateinamen (kann relative Pfade enthalten)
            filenames = [Path(line).stem for line in lines]
            splits[split_name] = filenames
            print(f"  {split_name}: {len(filenames)} Eintraege")

    return splits if splits else None


def create_random_splits(img_dir, train_ratio=0.8, val_ratio=0.1):
    """Erstelle zufaellige Splits wenn keine .txt Dateien vorhanden"""
    images = sorted([f.stem for f in img_dir.rglob("*")
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    random.seed(42)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }


def convert_label_file(label_path, class_mapping):
    """Konvertiere alle Zeilen einer Label-Datei mit Klassen-Remapping"""
    converted = []
    with open(label_path) as f:
        for line in f:
            remapped = remap_label_line(line, class_mapping)
            if remapped:
                converted.append(remapped)
    return converted


def download_huggingface_starter():
    """Lade Starter-Dataset von HuggingFace (Fallback, wenige Bilder)"""
    print("\n=== HuggingFace Log-Pile Starter Dataset ===\n")

    try:
        from datasets import load_dataset
        ds = load_dataset("murcmurc/log-pile-images-v0")
        print(f"Train: {len(ds['train'])} Bilder")
        if 'test' in ds:
            print(f"Test: {len(ds['test'])} Bilder")

        setup_yolo_dirs()
        count = 0

        for split_name, hf_split in [('train', 'train'), ('val', 'test')]:
            if hf_split not in ds:
                continue
            for i, item in enumerate(ds[hf_split]):
                img = item.get('image')
                if img is None:
                    continue

                img_path = YOLO_DIR / 'images' / split_name / f'hf_{i:04d}.jpg'
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(str(img_path), 'JPEG', quality=90)

                # Leere Label-Datei (muss noch annotiert werden)
                label_path = YOLO_DIR / 'labels' / split_name / f'hf_{i:04d}.txt'
                label_path.touch()
                count += 1

        print(f"\n{count} Bilder gespeichert (noch nicht annotiert)")
        return count

    except Exception as e:
        print(f"HuggingFace Download fehlgeschlagen: {e}")
        return 0


def create_yaml(classes=None):
    """Erstelle YOLO dataset.yaml"""
    if classes is None:
        classes = WVB_CLASSES

    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(classes.items()))

    yaml_content = f"""# WVB Holzmengen-Erkennung Dataset
# Basiert auf TimberVision (WACV 2025) - Steininger et al.
# Instance Segmentation fuer Holzstamm-Erkennung

path: {YOLO_DIR.absolute()}
train: images/train
val: images/val
test: images/test

# Klassen (gemappt von TimberVision)
# cut = Schnittflaeche (Stirnseite, fuer Durchmesser-Messung)
# side = Seitenflaeche (Mantelflaeche, fuer Laengen-Messung)
# trunk = Gesamter Stamm (OBB)
names:
{names_block}
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"\nDataset YAML: {yaml_path}")


def print_dataset_info():
    """Zeige Informationen zum fertigen Dataset"""
    print(f"\n{'=' * 50}")
    print("Dataset-Uebersicht")
    print(f"{'=' * 50}")
    print(f"Pfad: {YOLO_DIR}")

    for split in ['train', 'val', 'test']:
        img_dir = YOLO_DIR / 'images' / split
        lbl_dir = YOLO_DIR / 'labels' / split
        if img_dir.exists():
            n_img = len(list(img_dir.glob('*')))
            n_lbl = len([f for f in lbl_dir.glob('*.txt') if f.stat().st_size > 0])
            print(f"  {split:5s}: {n_img:5d} Bilder, {n_lbl:5d} mit Labels")

    yaml_path = YOLO_DIR / "dataset.yaml"
    if yaml_path.exists():
        print(f"\nYAML: {yaml_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='WVB Dataset Setup - TimberVision Download & Konvertierung'
    )
    parser.add_argument(
        '--source', choices=['timbervision', 'huggingface', 'all'],
        default='timbervision',
        help='Datenquelle (default: timbervision)'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Ueberspringe Download, nur Konvertierung'
    )
    parser.add_argument(
        '--keep-all-classes', action='store_true',
        help='Behalte alle 4 TimberVision-Klassen (inkl. bound)'
    )
    args = parser.parse_args()

    print("=== WVB Dataset Setup ===\n")

    total = 0

    if args.source in ['timbervision', 'all']:
        # TimberVision herunterladen
        if not args.skip_download:
            success = download_timbervision()
            if not success:
                print("\nTimberVision Download fehlgeschlagen.")
                print("Alternativen:")
                print("  1. Manuell herunterladen:")
                print(f"     {ZENODO_URL}")
                print(f"     Speichern als: {ZENODO_ZIP}")
                print("  2. Roboflow nutzen: python3 download_roboflow.py")
                if args.source == 'timbervision':
                    exit(1)

        # Konvertieren
        if args.keep_all_classes:
            # Alle 4 Klassen behalten (cut, bound, side, trunk)
            global TV_TO_WVB_ALL
            TV_TO_WVB_ALL = {0: 0, 1: 1, 2: 2, 3: 3}
            total += convert_timbervision_to_yolo()
            create_yaml(TV_CLASSES)
        else:
            total += convert_timbervision_to_yolo()
            create_yaml()

    if args.source in ['huggingface', 'all']:
        total += download_huggingface_starter()
        if args.source == 'huggingface':
            create_yaml({0: "log"})

    print(f"\n=== Fertig! {total} Bilder total ===")

    if total > 0:
        print_dataset_info()
    else:
        print("\nKeine Bilder verarbeitet. Optionen:")
        print("1. TimberVision:  python3 download_dataset.py --source timbervision")
        print("2. Roboflow:      python3 download_roboflow.py")
        print("3. HuggingFace:   python3 download_dataset.py --source huggingface")
        print("4. Manuell:       Bilder in ml/data/yolo_format/images/train/ ablegen")
