# Drowning Detection System

[![CI](https://github.com/rajaaliakhtar/drowning-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/rajaaliakhtar/drowning-detector/actions/workflows/ci.yml)
[![Docker](https://github.com/rajaaliakhtar/drowning-detector/actions/workflows/docker-build.yml/badge.svg)](https://github.com/rajaaliakhtar/drowning-detector/actions/workflows/docker-build.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time drowning detection system using computer vision and deep learning. Processes RTSP camera streams or uploaded video, detects swimmers with YOLOv8 + ByteTrack, estimates pose via MediaPipe, classifies distress behaviour with an LSTM on 5-second pose sequences, and fires SMS/webhook alerts within 10 seconds.

---

## Architecture

```
Camera (RTSP) ──► YOLOv8 (person detection)
                      │
                      ▼
                 ByteTrack (multi-object tracking)
                      │
                      ▼
                 MediaPipe (pose estimation → 14 joints)
                      │
                      ▼
                 LSTM Classifier (5-sec sliding window)
                      │
                      ├──► Alert Engine (threshold + cooldown)
                      │         │
                      │         ├──► SMS (Twilio)
                      │         ├──► WebSocket (dashboard)
                      │         └──► S3 (30-sec incident clip)
                      │
                      └──► Passive Drowning Rule (velocity < 0.005 for 8s)
```

## Key Features

- **Sub-10-second alert latency** — two consecutive 5-second windows required before alert fires
- **Dual detection** — LSTM classifier + passive drowning stillness rule
- **Privacy by design** — no raw video stored; only 30-second clips around confirmed alerts
- **Production-ready** — Docker, GPU support, health checks, Prometheus metrics, Sentry integration
- **Water zone masking** — inference only runs inside configured pool boundaries

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Person Detection | YOLOv8 (Ultralytics) |
| Tracking | ByteTrack |
| Pose Estimation | MediaPipe |
| Behaviour Classification | PyTorch LSTM |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| Cache / Queue | Redis 7 |
| Monitoring | Prometheus + Grafana |
| Alerts | Twilio SMS + WebSocket |
| Storage | AWS S3 |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Deployment | AWS EC2 (g4dn.xlarge) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU + CUDA 12.1 (for inference)
- Git

### Local Development

```bash
# Clone
git clone https://github.com/rajaaliakhtar/drowning-detector.git
cd drowning-detector

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
make install-dev

# Copy environment file
cp .env.example .env
# Edit .env with your credentials

# Run API server
make api

# Run tests
make test
```

### Docker (Production)

```bash
# Start all services (API + PostgreSQL + Redis + Prometheus + Grafana)
make docker-up

# With GPU support (requires nvidia-docker)
docker-compose -f drowning_detector/docker/docker-compose.yml up --build

# Check health
curl http://localhost:8000/health

# View logs
make docker-logs

# Stop
make docker-down
```

---

## Project Structure

```
drowning_detector/
├── api/                    # FastAPI application
│   ├── main.py             # App entry point, health checks
│   ├── pipeline.py         # End-to-end inference pipeline
│   └── alert.py            # SMS + webhook alert logic
├── core/                   # Shared configuration & constants
│   ├── config.py           # Pydantic settings (env vars)
│   ├── constants.py        # Labels, joints, thresholds
│   └── logging.py          # Loguru structured logging
├── data/                   # Data directory (git-ignored)
│   ├── raw_video/          # Original downloads (read-only)
│   ├── clips/              # Clipped training videos
│   ├── poses/              # Extracted .npy pose files
│   └── camera_configs/     # Per-camera water zone masks
├── docker/                 # Container configuration
│   ├── Dockerfile          # Multi-stage production build
│   ├── docker-compose.yml  # Full stack orchestration
│   └── prometheus.yml      # Metrics scraping config
├── models/
│   ├── detector/           # YOLOv8 fine-tuned weights
│   └── classifier/         # LSTM model + checkpoints
├── scripts/                # Data pipeline scripts
├── notebooks/              # Jupyter exploration & training
├── tests/                  # Test suite
│   ├── unit/
│   ├── integration/
│   └── conftest.py
└── frontend/dashboard/     # React operator dashboard
```

---

## Data Pipeline

```bash
# 1. Clip raw videos into 5-second segments
python drowning_detector/scripts/clip_videos.py \
    --input data/raw_video/positive/ \
    --output data/clips/drowning/

# 2. Extract 14-joint pose sequences
python drowning_detector/scripts/extract_poses.py

# 3. Build annotation CSV
python drowning_detector/scripts/build_annotations.py

# 4. Verify dataset health
python drowning_detector/scripts/verify_dataset.py
```

## Training

```bash
# Train LSTM classifier
make train

# Monitor with TensorBoard
make tensorboard
# Open http://localhost:6006
```

### Evaluation Criteria

| Metric | Target | Priority |
|--------|--------|----------|
| Drowning Recall | >= 0.95 | **Primary** |
| False Positive Rate | <= 0.10 | Secondary |
| AUC-ROC | >= 0.92 | Secondary |
| Inference Latency | <= 500ms/frame | Operational |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check for load balancers |
| `GET` | `/ready` | Readiness probe (models + deps) |
| `POST` | `/api/v1/stream` | Start monitoring RTSP stream |
| `POST` | `/api/v1/upload` | Upload video for analysis |
| `WS` | `/ws/alerts/{camera_id}` | Real-time alert WebSocket |

All endpoints return: `{"status": "...", "data": {...}, "error": null}`

---

## Deployment (AWS)

```bash
# Target: EC2 g4dn.xlarge (1x T4 GPU, 16GB RAM)
# AMI: Deep Learning AMI (Ubuntu 20.04)

ssh -i keypair.pem ubuntu@<ec2-ip>
git clone https://github.com/rajaaliakhtar/drowning-detector
cd drowning-detector
cp .env.example .env  # fill in secrets
docker-compose up -d
```

Automated deployment via GitHub Actions on release tags. See [deploy.yml](.github/workflows/deploy.yml).

---

## Development

```bash
make help          # Show all available commands
make install-dev   # Install dev dependencies + pre-commit hooks
make format        # Auto-format code (black + isort + ruff)
make lint          # Run all linters
make test          # Run unit tests
make test-cov      # Run tests with coverage
make clean         # Remove build artifacts
```

### Branch Strategy

- `main` — production-ready, protected
- `develop` — integration branch
- `feature/*` — new features
- `fix/*` — bug fixes
- `release/*` — release preparation

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Raja Ali Akhtar** — CV Engineer, Lahore
Contact: rajaalaiakhtar07@gmail.com
