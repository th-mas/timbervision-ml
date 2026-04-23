#!/usr/bin/env python3
"""
TimberVision ML Inference Server.
Holzstamm-Erkennung und Volumenberechnung via YOLOv8 Instance Segmentation.

Starten: uvicorn inference_server:app --host 0.0.0.0 --port 8300
"""

import io
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from volume_calculator import inference_and_calculate, LKW_REFERENZEN, UMRECHNUNGSFAKTOREN

app = FastAPI(
    title="TimberVision ML Service",
    version="1.0.0",
    description="Holzstamm-Erkennung und Volumenschaetzung via YOLOv8 + TimberVision",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modell-Pfad: TimberVision trainiert → Pre-trained → Legacy Fallback
_MODELS_DIR = Path(__file__).parent.parent / 'models'
_MODEL_CANDIDATES = [
    _MODELS_DIR / 'timbervision_yolov8' / 'weights' / 'best.pt',
    _MODELS_DIR / 'yolov8l-1024-seg.pt',
    _MODELS_DIR / 'timberseg_yolov8n' / 'weights' / 'best.pt',
]
MODEL_PATH = str(next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[0]))

MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/webp'}


@app.post("/api/v1/estimate")
async def estimate_volume(
    image: UploadFile = File(...),
    referenz_typ: str = Form(default='standard_lkw'),
    holzart: str = Form(default='fichte_rundholz'),
    stamm_laenge_cm: float = Form(default=None),
):
    """
    Schaetze Holzvolumen aus einem Foto.

    - **image**: Foto der Holzladung (JPEG/PNG/WebP)
    - **referenz_typ**: LKW-Typ fuer Kalibrierung (standard_lkw, kurzholz_lkw, langholz_lkw, traktor_anhaenger)
    - **holzart**: Holzart fuer Umrechnungsfaktor (fichte_rundholz, buche_rundholz, etc.)
    - **stamm_laenge_cm**: Bekannte Stammlaenge in cm (optional)
    """
    if referenz_typ not in LKW_REFERENZEN:
        raise HTTPException(
            status_code=422,
            detail=f"Unbekannter referenz_typ: '{referenz_typ}'. Erlaubt: {list(LKW_REFERENZEN.keys())}"
        )
    if holzart not in UMRECHNUNGSFAKTOREN:
        raise HTTPException(
            status_code=422,
            detail=f"Unbekannte holzart: '{holzart}'. Erlaubt: {list(UMRECHNUNGSFAKTOREN.keys())}"
        )
    if stamm_laenge_cm is not None and (stamm_laenge_cm <= 0 or stamm_laenge_cm > 3000):
        raise HTTPException(
            status_code=422,
            detail=f"stamm_laenge_cm muss zwischen 1 und 3000 cm liegen"
        )

    content_type = image.content_type or ''
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Nicht erlaubter Bildtyp: '{content_type}'. Erlaubt: {sorted(ALLOWED_CONTENT_TYPES)}"
        )

    contents = await image.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Leere Datei uebermittelt")
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Bild zu gross: {len(contents)} Bytes. Maximum: {MAX_IMAGE_BYTES} Bytes"
        )

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()
    except Exception:
        raise HTTPException(status_code=422, detail="Ungueltige Bilddatei")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        ergebnis = inference_and_calculate(
            tmp_path,
            model_path=MODEL_PATH,
            referenz_typ=referenz_typ,
            holzart=holzart,
            stamm_laenge_cm=stamm_laenge_cm,
        )
        return ergebnis.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Inference fehlgeschlagen")
        raise HTTPException(status_code=500, detail="Inference fehlgeschlagen")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/v1/referenzen")
async def get_referenzen():
    """Verfuegbare LKW-Referenztypen mit Abmessungen"""
    return LKW_REFERENZEN


@app.get("/api/v1/holzarten")
async def get_holzarten():
    """Verfuegbare Holzarten mit Umrechnungsfaktoren (RM→FM)"""
    return UMRECHNUNGSFAKTOREN


@app.get("/health")
async def health():
    """Health Check - prueft ob Modell geladen ist"""
    model_exists = Path(MODEL_PATH).exists()
    return {
        "status": "ok" if model_exists else "model_missing",
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)
