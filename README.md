# TimberVision ML Service

Standalone Docker Microservice für Holzstamm-Erkennung und Volumenberechnung.  
Basiert auf **YOLOv8 Instance Segmentation** + **TimberVision Dataset** (WACV 2025).

## Features

- Automatische Erkennung von Holzstämmen auf LKW-Fotos
- Volumenberechnung in Festmeter (FM) und Raummeter (RM)
- 3 Erkennungsklassen: `cut` (Schnittfläche), `side` (Seitenansicht), `trunk` (Gesamtstamm)
- Pre-trained Modell wird beim ersten Start automatisch heruntergeladen
- REST API via FastAPI

## Schnellstart

```bash
# Service starten (lädt Modell automatisch beim ersten Start)
docker compose up -d

# Health Check
curl http://localhost:8300/health

# Foto analysieren
curl -X POST http://localhost:8300/api/v1/estimate \
  -F "image=@holzladung.jpg" \
  -F "holzart=fichte_rundholz" \
  -F "referenz_typ=standard_lkw"
```

## API

| Endpoint | Methode | Beschreibung |
|----------|---------|-------------|
| `/api/v1/estimate` | POST | Holzvolumen aus Foto schätzen |
| `/api/v1/referenzen` | GET | Verfügbare LKW-Typen |
| `/api/v1/holzarten` | GET | Holzarten + Umrechnungsfaktoren |
| `/health` | GET | Health Check |

### `POST /api/v1/estimate`

**Form-Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|-------------|
| `image` | file | — | Foto (JPEG/PNG/WebP, max. 20 MB) |
| `holzart` | string | `fichte_rundholz` | Holzart für Umrechnungsfaktor |
| `referenz_typ` | string | `standard_lkw` | LKW-Typ für Kalibrierung |
| `stamm_laenge_cm` | float | optional | Bekannte Stammläng in cm |

**Response:**

```json
{
  "anzahl_staemme": 24,
  "volumen_fm": 8.42,
  "volumen_rm": 12.03,
  "holzart": "fichte_rundholz",
  "umrechnungsfaktor": 0.70,
  "konfidenz_gesamt": 0.82,
  "referenz_typ": "standard_lkw",
  "methode": "timbervision_segmentation",
  "staemme": [...]
}
```

## Integration in Projekte

In `docker-compose.yml` des jeweiligen Projekts:

```yaml
services:
  timbervision-ml:
    build:
      context: /home/th/TimberVisionML
      dockerfile: Dockerfile
    container_name: timbervision-ml
    restart: unless-stopped
    ports:
      - "8300:8300"
    volumes:
      - ~/.timbervision/models:/app/models
      - ~/.timbervision/exports:/app/exports
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8300/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
```

Das Modell-Verzeichnis `~/.timbervision/models/` wird zwischen allen Projekten geteilt — einmal herunterladen, überall nutzen.

## Training

```bash
# TimberVision Dataset herunterladen (4.3 GB)
docker compose run --rm timbervision-ml python3 download_dataset.py

# Training starten (GPU empfohlen)
docker compose run --rm timbervision-ml python3 train.py train --epochs 100 --device 0

# Fine-Tuning auf Pre-trained Basis
docker compose run --rm timbervision-ml python3 train.py finetune --epochs 50

# Modell für Mobile exportieren (ONNX/TFLite)
docker compose run --rm timbervision-ml python3 train.py export
```

## Dataset

**TimberVision** (WACV 2025) — Steininger et al.  
>2000 Bilder aus Österreich, >51.000 Stamm-Komponenten, CC BY-NC-SA 4.0  
[Paper](https://arxiv.org/pdf/2501.07360v1) · [GitHub](https://github.com/timbervision/timbervision) · [Zenodo](https://zenodo.org/records/14825846)

## Struktur

```
TimberVisionML/
├── scripts/
│   ├── inference_server.py    # FastAPI Server
│   ├── volume_calculator.py   # Volumenberechnung
│   ├── train.py               # YOLOv8 Training + Export
│   └── download_dataset.py    # Dataset Download
├── Dockerfile
├── entrypoint.sh              # Auto-Download Modell beim ersten Start
├── requirements.txt
└── docker-compose.yml
```
