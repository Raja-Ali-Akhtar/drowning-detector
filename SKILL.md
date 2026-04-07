# Skill: Drowning Detection CV Pipeline

## When to use this skill
Load this skill whenever working on any part of the drowning detection system:
- Writing or debugging data pipeline scripts (clipping, pose extraction, annotation)
- Building or training the LSTM classifier or YOLOv8 detector
- Writing FastAPI inference endpoints
- Debugging video processing or MediaPipe issues
- Evaluating model performance

---

## Domain knowledge

### What drowning actually looks like (Instinctive Drowning Response)
The model must learn these specific signals — not generic "person in distress":
- Body orientation: vertical (not horizontal like swimming)
- Arms: pushing laterally downward at sides, not waving overhead
- Head: tilted back, mouth at or just above water surface
- Legs: little or no kicking visible
- No speech, no waving — the person cannot call for help
- Duration: 20–60 seconds before submersion
- Silent and looks calm from a distance — easily missed by human observers

### Hard negative: treading water
The most dangerous confusion class. Looks nearly identical:
- Also vertical body orientation
- Also arm movement at water level
- Key difference: controlled rhythm, head well above water, legs actively kicking
- If model confuses this with drowning → too many false positives → lifeguards ignore system

### Passive drowning (separate detection rule)
- Person already unconscious, face down or motionless underwater
- LSTM won't catch this — no pose = no features
- Handled by a separate stillness rule: tracked ID with velocity < 0.005 for 8+ seconds

---

## Pose sequence format

All pose data is stored as numpy arrays: shape `(T, 14, 3)`

```
T  = number of frames (target: 50 frames = 5 sec at 10fps)
14 = joints (see index below)
3  = (x, y, visibility) — x,y normalised 0-1 relative to frame
```

Joint index:
```
0:  nose
1:  left shoulder    2:  right shoulder
3:  left elbow       4:  right elbow
5:  left wrist       6:  right wrist
7:  left hip         8:  right hip
9:  left knee        10: right knee
11: left ankle       12: right ankle
13: head centre (computed as midpoint of ears/eyes)
```

Labels: `0 = normal`, `1 = drowning`, `2 = treading (hard negative)`

---

## Key computed features (for feature engineering)

These are derived from raw pose sequences and fed to the LSTM alongside raw keypoints:

```python
def compute_features(seq):
    """
    seq: np.array shape (T, 14, 3)
    returns: np.array shape (T, N_features)
    """
    features = []
    for t in range(len(seq)):
        kps = seq[t]  # (14, 3)
        vis = kps[:, 2]

        # Body verticality: angle of shoulder-hip vector from horizontal
        l_shoulder, r_shoulder = kps[1, :2], kps[2, :2]
        l_hip, r_hip = kps[7, :2], kps[8, :2]
        shoulder_mid = (l_shoulder + r_shoulder) / 2
        hip_mid = (l_hip + r_hip) / 2
        vec = shoulder_mid - hip_mid
        verticality = abs(np.arctan2(vec[0], vec[1]))  # 0=horizontal, pi/2=vertical

        # Head height relative to body (normalised)
        head_y = kps[0, 1]
        body_height = abs(shoulder_mid[1] - hip_mid[1]) + 1e-6
        head_position = (hip_mid[1] - head_y) / body_height

        # Arm lateral spread (wrists relative to hips)
        wrist_spread = abs(kps[5, 0] - kps[6, 0])

        # Mean visibility (low = swimmer partly submerged/obscured)
        mean_vis = np.mean(vis)

        features.append([verticality, head_position, wrist_spread, mean_vis])

    features = np.array(features)  # (T, 4)

    # Inter-frame velocity (append as additional features)
    velocity = np.diff(seq[:, :, :2], axis=0)  # (T-1, 14, 2)
    mean_vel = np.mean(np.linalg.norm(velocity, axis=2), axis=1, keepdims=True)  # (T-1, 1)
    mean_vel = np.vstack([[mean_vel[0]], mean_vel])  # pad first frame

    return np.hstack([features, mean_vel])  # (T, 5)
```

---

## LSTM architecture

```python
import torch
import torch.nn as nn

class DrowningLSTM(nn.Module):
    def __init__(self, input_size=47, hidden_size=128, num_layers=2, dropout=0.3):
        """
        input_size: 14 joints × 3 (x,y,vis) + 5 computed features = 47
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # [normal, drowning]
        )

    def forward(self, x):
        # x: (batch, T, input_size)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]       # take last timestep
        return self.classifier(last)
```

---

## Alert logic

```python
ALERT_THRESHOLD = 0.75
CONSECUTIVE_WINDOWS = 2  # must trigger in 2 consecutive 5-sec windows
STILLNESS_THRESHOLD = 0.005
STILLNESS_FRAMES = 80    # 8 seconds at 10fps

class AlertEngine:
    def __init__(self):
        self.tracker_history = {}  # {tracker_id: deque of probabilities}
        self.alerted_ids = {}      # {tracker_id: last_alert_timestamp}
        self.COOLDOWN = 60         # seconds between re-alerts for same person

    def update(self, tracker_id, drowning_prob, velocity, timestamp):
        if tracker_id not in self.tracker_history:
            self.tracker_history[tracker_id] = deque(maxlen=self.CONSECUTIVE_WINDOWS)

        self.tracker_history[tracker_id].append(drowning_prob)

        # Rule 1: LSTM threshold
        if all(p > ALERT_THRESHOLD for p in self.tracker_history[tracker_id]):
            return self._fire_alert(tracker_id, "lstm", timestamp)

        # Rule 2: Passive/stillness
        if velocity < STILLNESS_THRESHOLD:
            # tracked elsewhere with frame counter
            pass

        return None

    def _fire_alert(self, tracker_id, reason, timestamp):
        last = self.alerted_ids.get(tracker_id, 0)
        if timestamp - last < self.COOLDOWN:
            return None  # cooldown active
        self.alerted_ids[tracker_id] = timestamp
        return {"tracker_id": tracker_id, "reason": reason, "timestamp": timestamp}
```

---

## Common failure modes and fixes

| Problem | Symptom | Fix |
|---|---|---|
| Too many false positives | Alerts firing on treading water | Lower threshold temporarily, collect more treading clips, retrain |
| Poor pose detection | >30% zero frames in .npy files | Check camera angle — needs 45° overhead minimum. Discard clips with poor visibility. |
| ID switching in tracker | Same person gets new ID after occlusion | Tune ByteTrack `track_thresh` and `match_thresh`. Increase `track_buffer`. |
| LSTM not learning | Val loss stuck, recall near 0 | Check class imbalance — use `WeightedRandomSampler`. Verify labels are correct. |
| Slow inference | >2 sec per frame | Reduce YOLO input size to 416. Run MediaPipe on cropped bboxes only. Use TensorRT if on Jetson. |
| Water glare | YOLO misses swimmers | Add glare augmentation to training data. Fine-tune on pool-specific images. |

---

## Evaluation protocol

Always report these metrics together — never accuracy alone:

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate(y_true, y_pred, y_prob):
    print(classification_report(y_true, y_pred,
          target_names=["normal", "drowning"]))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    # Key numbers to report in README
    tn, fp, fn, tp = cm.ravel()
    print(f"False negative rate (missed drownings): {fn/(fn+tp):.3f}  ← must be < 0.05")
    print(f"False positive rate (false alarms):     {fp/(fp+tn):.3f}  ← target < 0.10")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.3f}")
```

**Pass criteria before deployment:**
- Drowning recall ≥ 0.95 (miss fewer than 1 in 20 drowning events)
- False positive rate ≤ 0.10 on test set
- AUC-ROC ≥ 0.92
- Inference latency ≤ 500ms per frame on target hardware

---

## AWS deployment (EC2 g4dn.xlarge)

```bash
# Instance: g4dn.xlarge — 1x T4 GPU, 16GB RAM, ~$0.526/hr on-demand
# AMI: Deep Learning AMI (Ubuntu 20.04)

# SSH in
ssh -i keypair.pem ubuntu@<ec2-ip>

# Pull repo and start
git clone https://github.com/rajaaliakhtar/drowning-detector
cd drowning-detector
cp .env.example .env  # fill in secrets
docker-compose up -d

# Expose port 8000 via security group
# Optional: Nginx reverse proxy + SSL via certbot
```

---

## References

- YOLOv8 docs: https://docs.ultralytics.com
- MediaPipe pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
- ByteTrack: https://github.com/ifzhang/ByteTrack
- Instinctive Drowning Response: Pia, F. (1974) — original research defining the behaviour pattern
- LSTM for action recognition: https://arxiv.org/abs/1411.4389
