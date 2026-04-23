# TimberVisionML

## Zweck
Standalone ML Microservice fuer Holzstamm-Erkennung und Volumenberechnung.
Basiert auf YOLOv8 Instance Segmentation + TimberVision Dataset (WACV 2025).
Wird von mehreren Projekten als gemeinsamer Service genutzt.

## Verwendende Projekte
| Projekt | Pfad | ML_SERVICE_URL |
|---------|------|---------------|
| WVBTransportApp | /home/th/WVBTransportApp | http://ml:8300 |
| MengenmeldungsAppV2 | /home/th/MengenmeldungsAppV2 | http://timbervision-ml:8300 |

## Stack
- Python 3.11
- YOLOv8 (ultralytics)
- FastAPI + uvicorn
- Pillow, numpy, opencv (optional fuer Ellipsen-Fit)

## API Endpoints
| Endpoint | Methode | Beschreibung |
|----------|---------|-------------|
| /api/v1/estimate | POST | Holzvolumen aus Foto schaetzen |
| /api/v1/referenzen | GET | Verfuegbare LKW-Typen |
| /api/v1/holzarten | GET | Verfuegbare Holzarten + Umrechnungsfaktoren |
| /health | GET | Health Check |

## Modell-Verzeichnis
Gemeinsam genutzt: `~/.timbervision/models/`
Beim ersten Start wird das Pre-trained TimberVision Modell automatisch heruntergeladen.

## Befehle
```bash
# Standalone starten (fuer Tests)
docker compose up -d

# Nur Modell herunterladen
docker compose run --rm timbervision-ml python3 train.py download-model

# Training (eigene Daten)
docker compose run --rm timbervision-ml python3 train.py train --epochs 100 --device 0

# Fine-Tuning
docker compose run --rm timbervision-ml python3 train.py finetune --epochs 50

# Logs
docker compose logs -f timbervision-ml
```

## Klassen (TimberVision)
| ID | Klasse | Beschreibung | Nutzen |
|----|--------|-------------|--------|
| 0 | cut | Schnittflaeche (Stirnseite) | Durchmesser via Ellipsen-Fit |
| 1 | side | Seitenflaeche (Rinde) | Stammlaenge |
| 2 | trunk | Gesamter Stamm | Fallback |

## Docker Compose Integration (in anderen Projekten)
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
