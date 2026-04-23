#!/usr/bin/env python3
"""
Holzvolumen-Berechnung aus YOLO-Segmentierungsergebnissen.

Unterstuetzt TimberVision-Klassen:
  cut (0)   = Schnittflaeche → praeziser Durchmesser aus Ellipsen-Fit
  side (1)  = Seitenflaeche  → Stammlaenge
  trunk (2) = Gesamtstamm    → Fallback wenn keine Komponenten erkannt

Workflow:
1. YOLOv8 erkennt Stamm-Komponenten (Instance Segmentation)
2. Schnittflaechen (cut) → Durchmesser via Ellipsen-Fit auf Maske
3. Seitenflaechen (side) → Sichtbare Stammlaenge
4. Referenz-Kalibrierung (LKW oder Messlatte) liefert px→cm Faktor
5. Volumen = Summe(pi * (d/2)^2 * L) pro Stamm
6. Umrechnung in Festmeter (FM) oder Raummeter (RM)

Umrechnungsfaktoren (Standardwerte):
- Festmeter (fm) = tatsaechliches Holzvolumen
- Raummeter (rm) = gestapeltes Holz inkl. Zwischenraeume
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# Umrechnungsfaktoren RM → FM
UMRECHNUNGSFAKTOREN = {
    'fichte_rundholz': 0.70,
    'buche_rundholz': 0.70,
    'laerche_rundholz': 0.70,
    'kiefer_rundholz': 0.68,
    'fichte_scheitholz': 0.65,
    'buche_scheitholz': 0.65,
    'brennholz_gemischt': 0.55,
    'industrieholz': 0.60,
}

# Standard-LKW Ladungsmasse (Innen) in cm
LKW_REFERENZEN = {
    'standard_lkw': {'laenge_cm': 700, 'breite_cm': 245, 'hoehe_cm': 260},
    'kurzholz_lkw': {'laenge_cm': 600, 'breite_cm': 245, 'hoehe_cm': 260},
    'langholz_lkw': {'laenge_cm': 1200, 'breite_cm': 245, 'hoehe_cm': 260},
    'traktor_anhaenger': {'laenge_cm': 500, 'breite_cm': 200, 'hoehe_cm': 200},
}

# TimberVision Klassen-IDs (nach WVB-Mapping)
CLS_CUT = 0    # Schnittflaeche
CLS_SIDE = 1   # Seitenflaeche
CLS_TRUNK = 2  # Gesamtstamm


@dataclass
class StammErgebnis:
    """Ergebnis fuer einen erkannten Stamm"""
    id: int
    durchmesser_cm: float
    laenge_cm: float
    flaeche_cm2: float
    volumen_fm: float
    konfidenz: float
    bbox: list  # [x1, y1, x2, y2] normalisiert
    quelle: str = "bbox"  # "cut_ellipse", "cut_bbox", "bbox" - wie Durchmesser ermittelt


@dataclass
class VolumenErgebnis:
    """Gesamt-Ergebnis der Volumenschaetzung"""
    anzahl_staemme: int
    volumen_fm: float
    volumen_rm: float
    holzart: str
    umrechnungsfaktor: float
    referenz_typ: str
    px_pro_cm: float
    konfidenz_gesamt: float
    staemme: list = field(default_factory=list)
    methode: str = 'timbervision_segmentation'
    klassen_info: dict = field(default_factory=lambda: {
        'cut': 0, 'side': 0, 'trunk': 0
    })

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def fit_ellipse_to_mask(mask_data, bbox_xyxy):
    """
    Fitte eine Ellipse an die Segmentierungsmaske.
    Gibt (minor_axis, major_axis) in Pixeln zurueck oder None.

    Die Schnittflaeche ist naherungsweise elliptisch.
    Der Minor-Axis ist der Stamm-Durchmesser (bei schraegemSchnitt).
    """
    try:
        import cv2
    except ImportError:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    # Maske auf BBox zuschneiden
    crop = mask_data[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Konturen finden
    binary = (crop > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Groesste Kontur nehmen
    largest = max(contours, key=cv2.contourArea)

    # Mindestens 5 Punkte fuer Ellipsen-Fit
    if len(largest) < 5:
        return None

    ellipse = cv2.fitEllipse(largest)
    # ellipse = ((cx, cy), (width, height), angle)
    width, height = ellipse[1]

    if width <= 0 or height <= 0:
        return None

    minor = min(width, height)
    major = max(width, height)
    return minor, major


class VolumeCalculator:
    """Berechnet Holzvolumen aus YOLO-Segmentierungsergebnissen"""

    def __init__(self, referenz_typ='standard_lkw', holzart='fichte_rundholz'):
        self.referenz_typ = referenz_typ
        self.holzart = holzart
        self.umrechnungsfaktor = UMRECHNUNGSFAKTOREN.get(holzart, 0.65)

    def berechne_px_pro_cm(self, img_width, img_height, referenz_breite_px=None):
        """
        Berechne Pixel-zu-cm Faktor anhand bekannter LKW-Masse.
        """
        ref = LKW_REFERENZEN.get(self.referenz_typ)
        if not ref:
            raise ValueError(f"Unbekannter Referenz-Typ: {self.referenz_typ}")

        breite_cm = ref['breite_cm']
        if breite_cm <= 0:
            raise ValueError(f"Ungueltige Referenz-Breite: {breite_cm}")

        if referenz_breite_px and referenz_breite_px > 0:
            return referenz_breite_px / breite_cm
        else:
            if img_width <= 0:
                raise ValueError(f"Ungueltige Bildbreite: {img_width}")
            estimated_breite_px = img_width * 0.80
            return estimated_breite_px / breite_cm

    def berechne_aus_segmentierung(self, yolo_result, img_width, img_height,
                                    referenz_breite_px=None, stamm_laenge_cm=None):
        """
        Berechne Volumen aus YOLO Segmentierungsergebnis.
        Nutzt TimberVision-Klassen fuer praezisere Messung:
        - cut-Masken fuer Durchmesser (Ellipsen-Fit)
        - side-Masken fuer Stammlaenge
        - trunk als Fallback
        """
        px_cm = self.berechne_px_pro_cm(img_width, img_height, referenz_breite_px)
        ref = LKW_REFERENZEN.get(self.referenz_typ, LKW_REFERENZEN['standard_lkw'])
        default_laenge = stamm_laenge_cm or ref['laenge_cm']

        empty_result = VolumenErgebnis(
            anzahl_staemme=0, volumen_fm=0, volumen_rm=0,
            holzart=self.holzart, umrechnungsfaktor=self.umrechnungsfaktor,
            referenz_typ=self.referenz_typ, px_pro_cm=px_cm, konfidenz_gesamt=0,
        )

        if not hasattr(yolo_result, 'masks') or yolo_result.masks is None:
            return empty_result
        if not hasattr(yolo_result, 'boxes') or yolo_result.boxes is None:
            return empty_result

        boxes = yolo_result.boxes
        masks = yolo_result.masks

        if len(boxes) == 0:
            return empty_result

        # Detektionen nach Klasse gruppieren
        klassen_counts = {'cut': 0, 'side': 0, 'trunk': 0}
        cut_detections = []   # (bbox, mask, conf)
        side_detections = []
        trunk_detections = []

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            mask = masks.data[i].cpu().numpy() if masks.data is not None else None

            det = (bbox, mask, conf, i)

            if cls_id == CLS_CUT:
                cut_detections.append(det)
                klassen_counts['cut'] += 1
            elif cls_id == CLS_SIDE:
                side_detections.append(det)
                klassen_counts['side'] += 1
            elif cls_id == CLS_TRUNK:
                trunk_detections.append(det)
                klassen_counts['trunk'] += 1
            else:
                # Unbekannte Klasse als trunk behandeln
                trunk_detections.append(det)

        # Volumen berechnen - Strategie abhaengig von erkannten Klassen
        staemme = []
        total_fm = 0

        if cut_detections:
            # Beste Methode: Durchmesser aus Schnittflaechen
            staemme, total_fm = self._berechne_aus_cuts(
                cut_detections, side_detections, px_cm, default_laenge,
                img_width, img_height
            )
        elif trunk_detections or side_detections:
            # Fallback: Durchmesser aus BBox
            all_dets = trunk_detections + side_detections
            staemme, total_fm = self._berechne_aus_bbox(
                all_dets, px_cm, default_laenge, img_width, img_height
            )

        total_rm = total_fm / self.umrechnungsfaktor if self.umrechnungsfaktor > 0 else 0
        avg_conf = np.mean([s.konfidenz for s in staemme]) if staemme else 0

        return VolumenErgebnis(
            anzahl_staemme=len(staemme),
            volumen_fm=round(total_fm, 2),
            volumen_rm=round(total_rm, 2),
            holzart=self.holzart,
            umrechnungsfaktor=self.umrechnungsfaktor,
            referenz_typ=self.referenz_typ,
            px_pro_cm=round(px_cm, 4),
            konfidenz_gesamt=round(float(avg_conf), 3),
            staemme=[asdict(s) for s in staemme],
            klassen_info=klassen_counts,
        )

    def _berechne_aus_cuts(self, cut_dets, side_dets, px_cm, default_laenge,
                            img_width, img_height):
        """
        Berechne Volumen aus Schnittflaechen-Detektionen.
        Durchmesser wird aus Ellipsen-Fit der Maske berechnet.
        """
        staemme = []
        total_fm = 0

        for idx, (bbox, mask, conf, orig_idx) in enumerate(cut_dets):
            # Durchmesser aus Ellipsen-Fit der Schnittflaeche
            durchmesser_px = None
            quelle = "cut_bbox"

            if mask is not None:
                ellipse_result = fit_ellipse_to_mask(mask, bbox)
                if ellipse_result:
                    minor_px, major_px = ellipse_result
                    # Minor-Axis = Durchmesser (bei schraegemSchnitt)
                    durchmesser_px = minor_px
                    quelle = "cut_ellipse"

            # Fallback: Durchmesser aus BBox der Schnittflaeche
            if durchmesser_px is None:
                bbox_w = bbox[2] - bbox[0]
                bbox_h = bbox[3] - bbox[1]
                durchmesser_px = min(bbox_w, bbox_h)

            if durchmesser_px <= 0 or px_cm <= 0:
                continue

            durchmesser_cm = durchmesser_px / px_cm

            # Stammlaenge: Versuche aus zugehoeriger side-Detektion
            laenge_cm = self._finde_stammlaenge(
                bbox, side_dets, px_cm, default_laenge
            )

            # Volumen berechnen
            radius_cm = durchmesser_cm / 2
            volumen_cm3 = math.pi * radius_cm ** 2 * laenge_cm
            volumen_fm = volumen_cm3 / 1_000_000
            flaeche_cm2 = math.pi * radius_cm ** 2

            stamm = StammErgebnis(
                id=idx,
                durchmesser_cm=round(durchmesser_cm, 1),
                laenge_cm=round(laenge_cm, 1),
                flaeche_cm2=round(flaeche_cm2, 1),
                volumen_fm=round(volumen_fm, 4),
                konfidenz=round(conf, 3),
                bbox=[float(b) for b in bbox / max(img_width, img_height)],
                quelle=quelle,
            )
            staemme.append(stamm)
            total_fm += volumen_fm

        return staemme, total_fm

    def _finde_stammlaenge(self, cut_bbox, side_dets, px_cm, default_laenge):
        """
        Finde die Stammlaenge aus einer zugehoerigen side-Detektion.
        Matching ueber raeumliche Naehe der BBoxen.
        """
        if not side_dets:
            return default_laenge

        cut_center_x = (cut_bbox[0] + cut_bbox[2]) / 2
        cut_center_y = (cut_bbox[1] + cut_bbox[3]) / 2

        best_side = None
        best_dist = float('inf')

        for side_bbox, _, _, _ in side_dets:
            side_center_x = (side_bbox[0] + side_bbox[2]) / 2
            side_center_y = (side_bbox[1] + side_bbox[3]) / 2
            dist = math.sqrt(
                (cut_center_x - side_center_x) ** 2 +
                (cut_center_y - side_center_y) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_side = side_bbox

        if best_side is not None:
            # Laengere Seite der side-BBox = Stammlaenge
            side_w = best_side[2] - best_side[0]
            side_h = best_side[3] - best_side[1]
            laenge_px = max(side_w, side_h)
            laenge_cm = laenge_px / px_cm

            if laenge_cm >= 50:  # Mindestens 50cm
                return min(laenge_cm, default_laenge)

        return default_laenge

    def _berechne_aus_bbox(self, detections, px_cm, default_laenge,
                            img_width, img_height):
        """
        Fallback: Berechne Volumen aus BBox (wenn keine cut-Klasse verfuegbar).
        Kompatibel mit alten Modellen die nur 'log' erkennen.
        """
        staemme = []
        total_fm = 0

        for idx, (bbox, mask, conf, orig_idx) in enumerate(detections):
            bbox_height_px = bbox[3] - bbox[1]
            bbox_width_px = bbox[2] - bbox[0]

            if bbox_height_px <= 0 or bbox_width_px <= 0 or px_cm <= 0:
                continue

            # Kuerzere Seite = Durchmesser, laengere = sichtbare Laenge
            if bbox_height_px < bbox_width_px:
                durchmesser_px = bbox_height_px
                sichtbare_laenge_px = bbox_width_px
            else:
                durchmesser_px = bbox_width_px
                sichtbare_laenge_px = bbox_height_px

            durchmesser_cm = durchmesser_px / px_cm
            sichtbare_laenge_cm = sichtbare_laenge_px / px_cm

            laenge_cm = min(sichtbare_laenge_cm, default_laenge)
            if laenge_cm < 50:
                laenge_cm = default_laenge

            radius_cm = durchmesser_cm / 2
            volumen_cm3 = math.pi * radius_cm ** 2 * laenge_cm
            volumen_fm = volumen_cm3 / 1_000_000
            flaeche_cm2 = math.pi * radius_cm ** 2

            stamm = StammErgebnis(
                id=idx,
                durchmesser_cm=round(durchmesser_cm, 1),
                laenge_cm=round(laenge_cm, 1),
                flaeche_cm2=round(flaeche_cm2, 1),
                volumen_fm=round(volumen_fm, 4),
                konfidenz=round(conf, 3),
                bbox=[float(b) for b in bbox / max(img_width, img_height)],
                quelle="bbox",
            )
            staemme.append(stamm)
            total_fm += volumen_fm

        return staemme, total_fm

    def berechne_aus_polter_foto(self, yolo_result, img_width, img_height,
                                  polter_breite_cm=None, polter_hoehe_cm=None):
        """
        Alternative: Polter-Methode (Stapel von der Seite fotografiert).
        Volumen = Breite x Hoehe x Tiefe x Umrechnungsfaktor
        """
        empty = VolumenErgebnis(
            anzahl_staemme=0, volumen_fm=0, volumen_rm=0,
            holzart=self.holzart, umrechnungsfaktor=self.umrechnungsfaktor,
            referenz_typ=self.referenz_typ, px_pro_cm=0, konfidenz_gesamt=0,
            methode='polter_flaeche'
        )

        if not hasattr(yolo_result, 'masks') or yolo_result.masks is None:
            return empty
        if not hasattr(yolo_result, 'boxes') or len(yolo_result.boxes) == 0:
            return empty

        px_cm = self.berechne_px_pro_cm(img_width, img_height)
        boxes = yolo_result.boxes

        # Gesamte Polterflaeche aus allen Bounding Boxes
        all_boxes = boxes.xyxy.cpu().numpy()
        min_x = all_boxes[:, 0].min()
        min_y = all_boxes[:, 1].min()
        max_x = all_boxes[:, 2].max()
        max_y = all_boxes[:, 3].max()

        polter_breite_px = max_x - min_x
        polter_hoehe_px = max_y - min_y

        if polter_breite_cm is None:
            polter_breite_cm = polter_breite_px / px_cm
        if polter_hoehe_cm is None:
            polter_hoehe_cm = polter_hoehe_px / px_cm

        ref = LKW_REFERENZEN.get(self.referenz_typ, LKW_REFERENZEN['standard_lkw'])
        tiefe_cm = ref.get('laenge_cm', 400)

        rm = (polter_breite_cm * polter_hoehe_cm * tiefe_cm) / 1_000_000
        fm = rm * self.umrechnungsfaktor
        avg_conf = float(np.mean(boxes.conf.cpu().numpy()))

        return VolumenErgebnis(
            anzahl_staemme=len(boxes),
            volumen_fm=round(fm, 2),
            volumen_rm=round(rm, 2),
            holzart=self.holzart,
            umrechnungsfaktor=self.umrechnungsfaktor,
            referenz_typ=self.referenz_typ,
            px_pro_cm=round(px_cm, 4),
            konfidenz_gesamt=round(avg_conf, 3),
            methode='polter_flaeche',
        )


def inference_and_calculate(image_path, model_path=None,
                            referenz_typ='standard_lkw',
                            holzart='fichte_rundholz',
                            stamm_laenge_cm=None):
    """
    Convenience-Funktion: Bild → YOLO Inference → Volumen-Berechnung
    """
    from ultralytics import YOLO
    from PIL import Image

    if model_path is None:
        model_dir = Path(__file__).parent.parent / 'models'
        # Zuerst TimberVision-Modell suchen, dann Fallback
        candidates = [
            model_dir / 'timbervision_yolov8' / 'weights' / 'best.pt',
            model_dir / 'yolov8l-1024-seg.pt',  # TimberVision Pre-trained
            model_dir / 'timberseg_yolov8n' / 'weights' / 'best.pt',  # Legacy
        ]
        model_path = next((str(p) for p in candidates if p.exists()), None)
        if not model_path:
            raise FileNotFoundError(
                "Kein Modell gefunden! Optionen:\n"
                "  1. python3 train.py download-model\n"
                "  2. python3 train.py train\n"
                "  3. --model Pfad angeben"
            )

    model = YOLO(model_path)
    img = Image.open(image_path)
    img_w, img_h = img.size

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Ungueltige Bildgroesse: {img_w}x{img_h}")

    results = model(image_path, verbose=False)

    if not results:
        return VolumenErgebnis(
            anzahl_staemme=0, volumen_fm=0, volumen_rm=0,
            holzart=holzart,
            umrechnungsfaktor=UMRECHNUNGSFAKTOREN.get(holzart, 0.65),
            referenz_typ=referenz_typ, px_pro_cm=0, konfidenz_gesamt=0,
        )

    calc = VolumeCalculator(referenz_typ=referenz_typ, holzart=holzart)
    return calc.berechne_aus_segmentierung(
        results[0], img_w, img_h, stamm_laenge_cm=stamm_laenge_cm
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Holzvolumen-Berechnung')
    parser.add_argument('image', help='Pfad zum Bild')
    parser.add_argument('--model', help='Pfad zum YOLO Modell')
    parser.add_argument('--referenz', default='standard_lkw',
                        choices=LKW_REFERENZEN.keys())
    parser.add_argument('--holzart', default='fichte_rundholz',
                        choices=UMRECHNUNGSFAKTOREN.keys())
    parser.add_argument('--laenge', type=float, help='Stammlaenge in cm')
    args = parser.parse_args()

    ergebnis = inference_and_calculate(
        args.image,
        model_path=args.model,
        referenz_typ=args.referenz,
        holzart=args.holzart,
        stamm_laenge_cm=args.laenge,
    )

    print(ergebnis.to_json())
