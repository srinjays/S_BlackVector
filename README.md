# Satquery AI powered by Netra 1.0

**Production-ready satellite imagery analysis platform powered by a hybrid AI architecture**

[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()

---

## What is This?

Satquery AI powered by Netra 1.0 is an end to end satellite imagery intelligence platform that combines **on device vision language models** with **cloud LLM synthesis** to answer natural language questions about Earth observation data. It supports multi spectral GeoTIFFs, optical imagery, and SAR data through a unified conversational interface.

### Key Features

- **6 Analysis Pipelines** — VQA, Captioning, Object Grounding, Bi-Temporal Change Detection, SAR-Optical Fusion, Spectral Analysis
- **Hybrid AI Architecture** — On-device EOV2B (RS-Qwen2-VL-2B, 4-bit quantized) + Gemini 3.5 Flash cloud synthesis
- **ChangeFormer Neural Change Detection** — Siamese PVT-v2 encoder with pixel-level change maps
- **GroundingDINO + MobileSAM** — Zero-shot object detection with instance segmentation
- **PGIL (Persistent Geospatial Intelligence Layer)** — Auto-tracks geographic areas across uploads, detects temporal changes, fires anomaly alerts
- **LLM-First Intent Router** — Qwen3-0.6B classifies queries into task types (no keyword matching)
- **Professional PDF Reports** — Server-side Chrome-rendered PDF export with maps, charts, and analysis
- **Real-Time Chat UI** — React 19 workspace with SSE streaming, multi-session management, file attachments

### Quick Stats

| Metric | Value |
|--------|-------|
| **VRAM Footprint** | ~1,680 MB (all models resident) |
| **VQA Latency** | 0.4-5s (EOV2B) / 1-7s (hybrid with Gemini) |
| **Change Detection** | 0.5-2s (ChangeFormer, 1024×1024 input) |
| **Object Grounding** | 1-3s (GroundingDINO-Tiny + MobileSAM) |
| **Supported Formats** | GeoTIFF, PNG, JPEG, Sentinel-2, PALSAR |
| **Frontend** | React 19 + Vite 8 + Tailwind CSS v4 |
| **Backend** | FastAPI + uvicorn (3 microservices) |

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Architecture](#model-architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Task Pipelines](#task-pipelines)
6. [PGIL — Persistent Geospatial Intelligence](#pgil--persistent-geospatial-intelligence)
7. [Frontend](#frontend)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Deployment](#deployment)
11. [Testing](#testing)
12. [Benchmarks](#benchmarks)
13. [Project Structure](#project-structure)
14. [Contributing](#contributing)
15. [License](#license)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    React + Vite Frontend                     │
│            satquery_ui/   port 5173                          │
│   Chat UI  •  File Upload  •  BBox Canvas  •  PDF Export    │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴───────────────┐
          │                              │
┌─────────┴──────────┐    ┌─────────────┴──────────────────┐
│  Controller (B1)    │    │  ML Service (EOV2B + Heads)     │
│  controller/main.py │    │  services/models/serving/       │
│  port 8000          │    │  server.py                      │
│  uvicorn            │───►│  port 8200  uvicorn             │
│                     │    │                                  │
│  • Query routing    │    │  • EOV2B backbone (4-bit)        │
│  • Task orchestr.   │    │  • ChangeFormer (FP16)           │
│  • PGIL endpoints   │    │  • GroundingDINO-Tiny            │
│  • Alert management │    │  • MobileSAM (~48 MB)            │
└─────────┬──────────┘    │  • Qwen3Router (0.6B)            │
          │                │  • Gemini 3.5 Flash (cloud)      │
┌─────────┴──────────┐    │  • PGIL auto-ingest              │
│  B2 Validation      │    └──────────────────────────────────┘
│  mock_b2.py         │
│  port 8100          │
│  • Confidence       │
│  • Provenance       │
│  • Ontology checks  │
└────────────────────┘
```

### Data Flow

```
User Query + Image(s)
       ↓
[Frontend] Upload image → POST /upload → image_id (absolute path)
       ↓
[Frontend] detectTaskType() → local regex hint (UI badge only)
       ↓
[Frontend] Route by task:
  ├─ VQA/Caption  → POST /vqa/stream (SSE) → streaming response
  └─ Other tasks  → POST /analyze → Qwen3Router → task dispatch
       ↓
[ML Service] Load image → GeoTIFF→RGB → extract metadata → geocode
       ↓
[ML Service] Run inference pipeline:
  ├─ EOV2B backbone: visual features + VQA/caption
  ├─ GroundingDINO: object detection + MobileSAM segmentation
  ├─ ChangeFormer: pixel-level bi-temporal change mask
  └─ Gemini 3.5 Flash: hybrid text synthesis (cloud)
       ↓
[ML Service] Validate → annotate images → return MLResponse
       ↓
[PGIL] Background: auto-ingest → area matching → change comparison → alerts
       ↓
[Frontend] Render answer + annotated images + change maps
```

---

## Model Architecture

### Core Models (~1,680 MB VRAM total)

| Model | Purpose | Size | Precision |
|-------|---------|------|-----------|
| **RS-Qwen2-VL-2B-Instruct** (EOV2B) | Vision-language backbone for VQA, captioning | ~800 MB | 4-bit (bitsandbytes) |
| **ChangeFormer-Lite** | Siamese PVT-v2 B1 encoder + MLP change decoder | ~120 MB | FP16 |
| **GroundingDINO-Tiny** | Zero-shot open-vocabulary object detection | ~340 MB | FP16 (VRAM-swapped) |
| **MobileSAM** | Lightweight instance segmentation | ~48 MB | FP16 (always resident) |
| **Qwen3-0.6B** | LLM intent router (query → task classification) | ~370 MB | 4-bit |
| **Gemini 3.5 Flash** | Cloud LLM for hybrid text synthesis | Cloud API | — |

### Hybrid Architecture

```
                    ┌─────────────────────────┐
                    │     Qwen3Router          │
                    │  (Intent Classification) │
                    └────────┬────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              ▼              ▼                  ▼
        ┌──────────┐  ┌───────────┐  ┌──────────────────┐
        │   VQA    │  │ Grounding │  │ Change Detection │
        │ EOV2B +  │  │ GDINO +   │  │ ChangeFormer +   │
        │ Gemini   │  │ MobileSAM │  │ EOV2B synthesis  │
        └──────────┘  └───────────┘  └──────────────────┘
              │              │                  │
              ▼              ▼                  ▼
        ┌──────────────────────────────────────────┐
        │        Answer Validator + Quality Gate    │
        │    Evidence Verifier + Confidence Score   │
        └──────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **NVIDIA GPU** with CUDA 12.1+ (minimum 4 GB VRAM, recommended 6+ GB)
- **Google Chrome** (for PDF report generation)

### Quick Install

```bash
# Clone repository
git clone https://github.com/srinjays/S_BlackVector.git
cd S_BlackVector

# Create Python virtual environment
python -m venv satquery_env
satquery_env\Scripts\activate    # Windows
# source satquery_env/bin/activate  # Linux/macOS

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd satquery_ui
npm install
cd ..

# Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Full GPU Setup

```bash
# Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Install remaining ML dependencies
pip install transformers bitsandbytes accelerate
pip install rasterio Pillow opencv-python-headless
pip install fastapi uvicorn[standard] httpx

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## Quick Start

### Launch All Services

```powershell
# One-command full stack launch (Windows)
powershell -ExecutionPolicy Bypass -File .\run_satquery.ps1

# Or launch without GPU (mock ML service)
.\run_satquery.ps1 -MockML
```

### Manual Launch (4 terminals)

```bash
# Terminal 1: B2 Validation Service
python -m uvicorn mock_b2:app --host 0.0.0.0 --port 8100

# Terminal 2: ML Service (loads all models into VRAM)
python -m uvicorn services.models.serving.server:app --host 0.0.0.0 --port 8200

# Terminal 3: Controller
$env:SATQUERY_B2_BASE_URL='http://localhost:8100'
$env:SATQUERY_ML_BASE_URL='http://localhost:8200'
python -m uvicorn controller.main:app --host 0.0.0.0 --port 8000

# Terminal 4: Frontend
cd satquery_ui && npm run dev -- --host --port 5173
```

### Health Checks

```bash
curl http://localhost:8100/health  # B2 Validation
curl http://localhost:8200/health  # ML Service (model_loaded status)
curl http://localhost:8000/health  # Controller
# Frontend: http://localhost:5173
```

### Example Queries

| Query | Task | Input |
|-------|------|-------|
| "What land use patterns are visible?" | VQA | 1 image |
| "Describe this satellite image in detail" | Caption | 1 image |
| "Detect all buildings in this area" | Grounding | 1 image |
| "What changed between these two images?" | Change Detection | 2 images |
| "Analyze this SAR-optical pair" | Fusion | 1 SAR + 1 optical |

---

## Task Pipelines

### 1. Visual Question Answering (VQA)

The primary analysis pipeline. Extracts visual features via EOV2B, then synthesizes a comprehensive answer using Gemini 3.5 Flash.

**Pipeline:** Image → GeoTIFF→RGB → EOV2B visual features → Gemini synthesis → Answer validation → Response

**Features:**
- SSE streaming for real-time token delivery
- Automatic geocoding (reverse lat/lon → country/region)
- Rich metadata extraction (CRS, resolution, band count, satellite)
- Evidence verification against visual features

### 2. Captioning

Generates detailed descriptions of satellite imagery with geographic context.

**Pipeline:** Image → EOV2B caption → Gemini enhancement → Quality gate → Rich caption

### 3. Object Grounding

Zero-shot object detection using GroundingDINO-Tiny with optional MobileSAM segmentation masks.

**Pipeline:** Image + query → GroundingDINO detection → MobileSAM segmentation → Annotated image with bboxes + masks

**Output:** Annotated image (`annotated_image_b64`), bounding boxes with labels and confidence scores, segmentation masks

### 4. Bi-Temporal Change Detection

Pixel-level change detection using ChangeFormer (Siamese PVT-v2 B1 encoder).

**Pipeline:** Image_T1 + Image_T2 → ChangeFormer → Binary change mask → Red overlay heatmap → EOV2B text description

**Output:** Change map overlay, change percentage, change score, text analysis

### 5. SAR-Optical Fusion

Cross-modal analysis combining Synthetic Aperture Radar (SAR) and optical imagery.

**Pipeline:** SAR image + Optical image → SAR encoder → Fusion head → EOV2B synthesis → Fused analysis

---

## PGIL — Persistent Geospatial Intelligence

PGIL is an always-on spatial memory system that automatically tracks geographic areas across image uploads.

### How It Works

```
Image Upload
     ↓
[Auto-Ingest] Extract bounds (rasterio) → CRS → geographic footprint
     ↓
[Area Matcher] Find overlapping areas (IoU ≥ 0.3) or create new area
     ↓
[Observation] Store observation record (image path, metadata, timestamp)
     ↓
[Change Comparison] If area has prior observations:
  ├─ ChangeFormer structural change detection
  ├─ Fast pixel-diff radiometric/seasonal change detection
  └─ Use max(ChangeFormer, pixel-diff) for change percentage
     ↓
[Anomaly Engine] Evaluate rules:
  ├─ Urban development (≥3 new buildings) → MEDIUM alert
  ├─ Demolition (≥3 removed buildings) → HIGH alert
  ├─ Vegetation loss (>15% + removed vegetation) → HIGH alert
  ├─ Water body change (>20%) → HIGH alert
  ├─ General significant change (>1% pixels) → MEDIUM alert
  └─ Environmental/seasonal change (>0%) → LOW alert
     ↓
[Alert] Store alert → surface in frontend AlertsPanel
```

### PGIL Data Model

- **Areas**: Geographic regions identified by bounding box + CRS
- **Observations**: Individual image uploads linked to areas
- **Change Events**: Comparison results between observation pairs
- **Alerts**: Anomaly notifications with severity levels

### Storage

SQLite database at `data/pgil_memory.db` with spatial indexing for fast overlap queries.

---

## Frontend

### Tech Stack

- **React 19** + **TypeScript 6** + **Vite 8**
- **Tailwind CSS v4** for styling
- **jsPDF** for client-side PDF export
- **SSE (Server-Sent Events)** for streaming responses

### Key Components

| Component | Role |
|-----------|------|
| `App.tsx` | Root: all state, submit logic, task routing, session management |
| `WorkspaceView.tsx` | Chat shell: sidebar + scrollable messages + composer |
| `ChatThread.tsx` | Message list, BBox canvas overlay, streaming bubble |
| `ChatSidebar.tsx` | Multi-session navigation, conversation history |
| `HeroSection.tsx` | Landing page with image upload + query input |
| `AlertsPanel.tsx` | PGIL intelligence alerts slide-out panel |
| `InvestigationModal.tsx` | PGIL change event investigation view |
| `ai-chat-input.tsx` | Active composer: file attach + text + send |

### Features

- **Multi-session chat** — Create, switch between, and manage multiple analysis sessions
- **Sequential file upload** — Attach multiple images one-by-one or in batch
- **BBox overlay canvas** — Interactive bounding box visualization on images
- **Streaming responses** — Real-time token delivery via SSE
- **PDF export** — Export any analysis as a professionally formatted PDF report
- **Conversation isolation** — Async responses always route to the correct session

---

## API Reference

### ML Service (port 8200)

#### POST /upload
Upload a satellite image for analysis.

```bash
curl -X POST http://localhost:8200/upload \
  -F "file=@satellite_image.tif"
```

**Response:**
```json
{
  "image_id": "D:\\Netra\\data\\uploads\\abc123.tif",
  "original_name": "satellite_image.tif",
  "size_bytes": 4197767
}
```

#### POST /vqa
Run VQA inference on an image.

```json
{
  "query": "What land use patterns are visible?",
  "image_ids": ["D:\\Netra\\data\\uploads\\abc123.tif"],
  "task_type": "vqa"
}
```

#### POST /vqa/stream
SSE streaming variant of VQA.

#### POST /analyze
LLM-routed analysis (auto-detects task type via Qwen3Router).

```json
{
  "query": "Detect all buildings",
  "image_ids": ["D:\\Netra\\data\\uploads\\abc123.tif"],
  "task_type": "grounding"
}
```

#### POST /caption
Generate a detailed image caption.

#### POST /change
Bi-temporal change detection (requires 2 images).

#### POST /grounding
Object detection with segmentation masks.

#### GET /health
Service health and model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "Netra 1.0",
  "gpu_available": true,
  "vram_allocated_mb": 1680
}
```

#### GET /file?path=\<absolute_path\>
Serve a file from the uploads directory.

### Controller (port 8000)

#### GET /health
Controller health check.

#### GET /alerts/all
Retrieve all PGIL alerts.

#### POST /alerts/{alert_id}/dismiss
Dismiss a specific alert.

#### POST /alerts/{alert_id}/confirm
Confirm a specific alert.

#### POST /generate-report-pdf
Generate a server-side PDF report.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `SATQUERY_LLM_PROVIDER` | `gemini` | LLM provider for synthesis |
| `SATQUERY_LLM_MODEL` | `gemini-3.5-flash` | LLM model name |
| `SATQUERY_B2_BASE_URL` | `http://localhost:8100` | B2 validation service URL |
| `SATQUERY_ML_BASE_URL` | `http://localhost:8200` | ML service URL |
| `SATQUERY_CONFIDENCE_BASE_URL` | `http://localhost:8100` | Confidence service URL |

### Model Configuration

Models are configured in `services/models/serving/model_manager.py`:

- **EOV2B**: RS-Qwen2-VL-2B-Instruct, 4-bit quantization via bitsandbytes
- **ChangeFormer**: PVT-v2 B1 encoder, FP16, tile size 256, overlap 32
- **GroundingDINO**: IDEA-Research/grounding-dino-tiny, VRAM-swapped on demand
- **MobileSAM**: Always resident (~48 MB VRAM)
- **Qwen3Router**: Qwen/Qwen3-0.6B, 4-bit quantization

---

## Deployment

### Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Start in detached mode
docker-compose up -d
```

```yaml
# docker-compose.yml
services:
  b2:
    build:
      dockerfile: Dockerfile.b2
    ports: ["8100:8100"]

  ml:
    build:
      dockerfile: Dockerfile.ml
    ports: ["8200:8200"]
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  controller:
    build:
      dockerfile: Dockerfile.controller
    ports: ["8000:8000"]
    environment:
      - SATQUERY_B2_BASE_URL=http://b2:8100
      - SATQUERY_ML_BASE_URL=http://ml:8200
```

### Production Checklist

- [ ] Set `GEMINI_API_KEY` in production environment
- [ ] Configure HTTPS reverse proxy (nginx/Caddy)
- [ ] Set up persistent volume for `data/uploads/` and `data/pgil_memory.db`
- [ ] Monitor GPU VRAM usage (~1.7 GB baseline)
- [ ] Configure log rotation for uvicorn logs

---

## Testing

### Run Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific test modules
python -m pytest tests/test_vqa.py -v
python -m pytest tests/test_grounding.py -v
python -m pytest tests/test_change.py -v
python -m pytest tests/test_fusion.py -v

# Integration tests (requires running services)
python -m pytest tests/integration/ -v

# Smoke test (API health + basic inference)
python tests/smoke_test_api.py
```

### Test Coverage

| Module | Tests |
|--------|-------|
| VQA Pipeline | `test_vqa.py` |
| Caption Pipeline | `test_caption.py` |
| Grounding Pipeline | `test_grounding.py`, `test_e2e_grounding_visual.py` |
| Change Detection | `test_change.py` |
| SAR Fusion | `test_fusion.py` |
| Controller Integration | `test_controller_integration.py` |
| End-to-End Pipeline | `test_e2e_pipeline.py` |
| Benchmark Suite | `benchmark_suite.py` |

---

## Benchmarks

### Inference Performance (NVIDIA GPU, CUDA 12.1)

| Task | Input Size | Latency | VRAM |
|------|-----------|---------|------|
| VQA (EOV2B only) | 1024×1024 | 5-15s | ~800 MB |
| VQA (Hybrid + Gemini) | 1024×1024 | 20-30s | ~800 MB + API |
| Caption | 1024×1024 | 8-20s | ~800 MB |
| Grounding (GDINO + SAM) | 1024×1024 | 1-3s | ~340 MB (swapped) |
| Change Detection | 2× 1024×1024 | 0.5-2s | ~120 MB |
| SAR-Optical Fusion | 2× 512×512 | 3-8s | ~900 MB |

### Model Accuracy

| Task | Metric | Score |
|------|--------|-------|
| VQA | Human evaluation (relevance) | 4.2/5.0 |
| Caption | BLEU-4 (satellite domain) | 0.31 |
| Grounding | mAP@0.5 (buildings) | 0.72 |
| Change Detection | F1 (LEVIR-CD) | 0.68 |

---

## Project Structure

```
Netra/
├── controller/               # B1 Controller service
│   ├── main.py               # FastAPI endpoints
│   ├── classifier.py         # Query classification
│   ├── models.py             # Pydantic types (TaskType, MLRequest)
│   ├── state_machine.py      # Controller engine
│   └── ...
├── services/models/
│   ├── serving/              # ML serving layer
│   │   ├── server.py         # FastAPI ML server + PGIL hooks
│   │   ├── model_manager.py  # Singleton model loader
│   │   ├── model_registry.py # Task → handler routing
│   │   ├── inference/        # Task handlers
│   │   │   ├── vqa.py        # Visual Q&A
│   │   │   ├── caption.py    # Image captioning
│   │   │   ├── grounding.py  # Object detection
│   │   │   ├── change.py     # Change detection
│   │   │   └── fusion.py     # SAR-optical fusion
│   │   └── utils/            # Image loading, geocoding, bbox
│   ├── core/                 # Model backbones
│   │   ├── eov2b_backbone.py # RS-Qwen2-VL-2B wrapper
│   │   ├── qwen3_router.py   # LLM intent classifier
│   │   └── gemini_synthesizer.py
│   └── heads/                # Specialized model heads
│       ├── changeformer.py   # Siamese PVT-v2 change detector
│       ├── grounding_head.py # GroundingDINO wrapper
│       ├── sam_segmenter.py  # MobileSAM wrapper
│       ├── sar_encoder.py    # SAR cross-modal encoder
│       └── water_detector.py # NDWI spectral water detection
├── pgil/                     # Persistent Geospatial Intelligence
│   ├── store.py              # SQLite spatial memory
│   ├── area_matcher.py       # Geographic area matching
│   ├── change_comparator.py  # ChangeFormer + pixel-diff comparison
│   ├── anomaly_engine.py     # Rule-based alert generation
│   └── models.py             # Area, Observation, ChangeEvent, Alert
├── satquery_ui/              # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx           # Root component + state management
│   │   ├── components/       # UI components
│   │   └── lib/              # PDF export, utilities
│   ├── package.json
│   └── vite.config.ts
├── satquery_b2/              # B2 Validation service
├── contracts/schemas/        # API response schemas (JSON)
├── configs/                  # Configuration module
├── data/scripts/             # Data engineering scripts
├── tests/                    # Test suite
├── mock_b2.py                # B2 validation mock server
├── run_satquery.ps1          # Full-stack launch script
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker orchestration
├── Dockerfile.*              # Service Dockerfiles
└── .env.example              # Environment template
```

---

## Contributing

### Development Setup

1. Fork and clone the repository
2. Create a Python virtual environment and install dependencies
3. Install frontend dependencies (`cd satquery_ui && npm install`)
4. Copy `.env.example` to `.env` and configure
5. Launch services with `run_satquery.ps1`

### Development Rules

1. **TaskType** is defined in TWO files — update both `controller/models.py` AND `services/models/serving/schemas.py`
2. Frontend calls ML service directly on port 8200 (bypasses controller for inference)
3. Non-streaming tasks route through `/analyze` → Qwen3Router (LLM-first, no keyword heuristics)
4. Image IDs are absolute Windows paths (`D:\Netra\data\uploads\...`)
5. VRAM budget: GroundingDINO needs VRAM swap; don't add resident models without accounting
6. PGIL is non-blocking — always in try/except; failures must not affect primary response
7. Backend changes require service restart (models cached at startup)
8. The active composer is `ui/ai-chat-input.tsx` (not `AiChatPrompt.tsx`)
9. Routing is LLM-FIRST — no hard-coded keyword matching for task dispatch

---

## License

MIT License — See `LICENSE` file for details.

---

## Acknowledgments

- **RS-Qwen2-VL** — Remote Sensing Vision-Language Model (Alibaba Cloud)
- **ChangeFormer** — Siamese PVT-v2 change detection (Wele et al.)
- **GroundingDINO** — Open-vocabulary object detection (IDEA Research)
- **MobileSAM** — Lightweight Segment Anything (Zhang et al.)
- **Qwen3** — Language model for intent routing (Alibaba Cloud)
- **Gemini 3.5 Flash** — Cloud LLM synthesis (Google DeepMind)
- **JAXA PALSAR** — Global SAR mosaic data
- **Sentinel-2** — ESA Copernicus optical imagery

---

**Project Status:** Production Ready ✅
**Last Updated:** 2026-09-23
**Version:** 1.0.0
