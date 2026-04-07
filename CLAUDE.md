# Drowning Detection System — Claude Code Context

## Project overview
Real-time drowning detection via computer vision. Processes RTSP streams or uploaded video,
detects swimmers using YOLOv8 + ByteTrack, runs pose estimation via MediaPipe, classifies
distress behaviour with an LSTM on 5-second pose sequences, and fires alerts within 10 seconds.

**Owner:** Raja Ali Akhtar — CV Engineer, Arrowx AI, Lahore  
**Stack:** Python 3.10+, PyTorch, YOLOv8, MediaPipe, FastAPI, Docker, AWS EC2

---

## Repo structure

```
drowning_detector/
  data/
    raw_video/              # original downloads — never modify
    clips/
      drowning/             # positive class (~400–600 clips)
      treading/             # hard negative (~600–800 clips)
      swimming/             # easy negative (~2000+ clips)
      splashing/            # easy negative (~500+ clips)
    poses/                  # extracted .npy files, one per clip
      drowning/
      treading/
      swimming/
      splashing/
    annotations.csv         # clip path, label, confidence, split
  scripts/
    clip_videos.py          # ffmpeg batch clipper
    extract_poses.py        # mediapipe pose extractor
    build_annotations.py    # build/update annotations.csv
    verify_dataset.py       # health check before training
  models/
    detector/               # YOLOv8 fine-tune weights + config
    classifier/             # LSTM model definition + checkpoints
  api/
    main.py                 # FastAPI app
    pipeline.py             # end-to-end inference pipeline
    alert.py                # SMS + webhook alert logic
  frontend/
    dashboard/              # React operator dashboard
  docker/
    Dockerfile
    docker-compose.yml
  notebooks/
    01_data_exploration.ipynb
    02_model_training.ipynb
    03_evaluation.ipynb
  tests/
  requirements.txt
  README.md
```

---

## Core conventions

### Python
- Python 3.10+. Type hints on all function signatures.
- Black formatting. Line length 100.
- Docstrings on every function that isn't a one-liner.
- No bare `except:` — always catch specific exceptions.
- Use `pathlib.Path` for all file paths, never `os.path.join` string concatenation.
- Log with `loguru` not `print`. Use `logger.info / logger.warning / logger.error`.

### Data pipeline
- Raw video is read-only. All processing writes to new directories.
- Pose sequences stored as `.npy`, shape `(T, 14, 3)` — 14 joints × (x, y, visibility).
- Joints index order: nose(0), L-shoulder(1), R-shoulder(2), L-elbow(3), R-elbow(4),
  L-wrist(5), R-wrist(6), L-hip(7), R-hip(8), L-knee(9), R-knee(10),
  L-ankle(11), R-ankle(12), head-centre(13).
- All coordinates normalised 0–1 relative to frame dimensions.
- Clip length standard: 5 seconds at 10 FPS = 50 frames per sequence.
- Pad short sequences with zeros at the end. Truncate long ones from the start.

### Labels
- 0 = normal (swimming, splashing)
- 1 = drowning (positive class)
- 2 = treading water (hard negative — evaluated separately in metrics)

### Model training
- Always set `torch.manual_seed(42)` and `np.random.seed(42)` at start of training script.
- Save checkpoints every 5 epochs to `models/classifier/checkpoints/`.
- Log to `runs/` using TensorBoard. Command: `tensorboard --logdir runs/`.
- Report: accuracy, precision, recall, F1, and AUC-ROC. Recall on class 1 (drowning) is the
  primary metric — a missed drowning is worse than a false alarm.
- Confusion matrix must be saved as an image artifact with every training run.

### API
- FastAPI. All endpoints return `{"status": ..., "data": ..., "error": null}`.
- RTSP stream input: POST `/api/v1/stream` with `{"rtsp_url": "...", "camera_id": "..."}`.
- Video upload input: POST `/api/v1/upload` with multipart form.
- Alert event: WebSocket at `/ws/alerts/{camera_id}`.
- Never log raw video frames — only metadata, timestamps, bounding boxes, scores.

### Docker
- One Dockerfile for the API service.
- `docker-compose.yml` runs API + PostgreSQL + Redis locally.
- GPU support via `--gpus all` flag — document this in README.

---

## Critical engineering decisions (do not change without discussion)

1. **Alert threshold = 0.75** — drowning probability must exceed 0.75 for two consecutive
   windows (10 seconds apart) before SMS fires. Reduces false positives without missing events.

2. **Passive drowning separate rule** — if a tracked swimmer ID has mean keypoint velocity
   < 0.005 for 8 consecutive seconds AND is in water zone, trigger alert regardless of LSTM.

3. **Water zone mask required** — each camera needs a polygon mask defining the water area.
   Stored in `data/camera_configs/{camera_id}.json`. Inference only runs inside this mask.

4. **ByteTrack for tracking** — do not switch to DeepSORT or BoTSORT without benchmarking.
   ByteTrack has lower ID-switch rate on crowded pool footage.

5. **No raw video stored by default** — only 30-second clips around alert events are saved
   to S3. Privacy by design. Can be enabled per-camera with explicit config flag.

---

## Environment variables

```
# .env (never commit this file)
DATABASE_URL=postgresql://user:pass@localhost:5432/drowning_db
REDIS_URL=redis://localhost:6379
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=drowning-detector-incidents
MODEL_WEIGHTS_PATH=models/classifier/best.pt
YOLO_WEIGHTS_PATH=models/detector/yolov8_pool.pt
ALERT_CONFIDENCE_THRESHOLD=0.75
```

---

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Data pipeline
python scripts/clip_videos.py --input data/raw_video/positive/ --output data/clips/drowning/
python scripts/extract_poses.py
python scripts/build_annotations.py
python scripts/verify_dataset.py          # run this before training

# Training
python models/classifier/train.py --epochs 50 --batch 32
tensorboard --logdir runs/

# API (local)
uvicorn api.main:app --reload --port 8000

# Docker
docker-compose up --build

# Tests
pytest tests/ -v
```

---

## Current phase

**Phase 1 — Data collection** (in progress)

- [ ] Download Kinetics-700 swimming class
- [ ] Download HMDB51 swim class
- [ ] YouTube mining — positive and hard negative class
- [ ] Pool simulation filming session (book ITU pool)
- [ ] Run extract_poses.py on all clips
- [ ] Run verify_dataset.py — confirm 400+ drowning clips before Phase 2

**Next:** Phase 2 — YOLOv8 fine-tuning + ByteTrack integration
