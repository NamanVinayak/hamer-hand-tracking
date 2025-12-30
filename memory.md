# Memory: Egocentric Hand Tracking Pipeline

## Goal
Build a production-ready hand tracking system for egocentric (first-person) videos that:
- Processes videos uploaded to cloud storage
- Runs on RunPod serverless GPUs
- Returns 21-joint 3D hand coordinates (MANO format)
- Visualizes results in Rerun.io

---

## Journey So Far

### Phase 1: MediaPipe Attempt
- Started with MediaPipe for hand tracking on Mac
- Created `hand_tracker.py` for local processing
- **Problem**: MediaPipe drops frames on difficult hand poses
- **Decision**: Upgrade to HaMeR for more robust detection

### Phase 2: HaMeR Local Installation
- Installed HaMeR in Python 3.10 virtual environment (`hamer_venv`)
- Downloaded 6GB model checkpoints to `hamer/_DATA/`
- **Blocker**: HaMeR's pyrender requires OpenGL which fails on Mac (EGL/OSMesa missing)
- **Decision**: Move processing to cloud GPUs (RunPod)

### Phase 3: Cloud Pipeline Design
- Chose RunPod serverless for GPU processing
- Initially planned AWS S3 for storage
- **Problem**: User couldn't create AWS free account
- **Solution**: Switched to Cloudflare R2 (S3-compatible, 10GB free)

### Phase 4: Docker Image Creation
- Created Dockerfile with CUDA 12.1, PyTorch 2.2.0, HaMeR, 6GB models
- First attempt used `nvidia/cuda` base image
- **Problem**: RunPod user namespace error - "container ID 1000222 cannot be mapped"
- **Root Cause**: HaMeR's `fetch_demo_data.sh` uses `tar` without `--no-same-owner`, preserving UID 1000222 from the UT Austin researcher's environment
- **Failed Fix**: `RUN chown` in separate layer doesn't work - UID is already committed to earlier layer
- **Solution**: Inline the model download and extract with `--no-same-owner` flag
- **Working Base**: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

### Phase 5: Smart Layering Strategy
- Created 2-layer Docker approach:
  - `Dockerfile.base` - Heavy dependencies + models (build once, ~30 min)
  - `Dockerfile.handler` - Just handler.py (build fast, ~1 min)
- Created `build.sh` with commands: `base`, `handler`, `push`
- **Currently**: Building base image with RunPod's official base

### Phase 6: Runtime Errors & Fixes
- **Error 1**: Job failed with `numpy.dtype size changed, binary incompatibility`
- **Cause**: `xtcocotools` (ViTPose dependency) compiled against numpy 1.x, but container has numpy 2.x
- **Failed Fix 1**: `pip install numpy<2.0 --force-reinstall` at end - didn't work because runpod/boto3 install reinstalled numpy 2.x
- **Successful Fix**: `PIP_CONSTRAINT` environment variable
  - Creates `/tmp/numpy-constraint.txt` with `numpy==1.26.4`
  - Sets `ENV PIP_CONSTRAINT=/tmp/numpy-constraint.txt`
  - Prevents ANY pip command from installing numpy>=2.0
  - Added build-time verification: `assert numpy.__version__.startswith('1.')`

### Phase 7: Visualization & 2D Keypoints
- **Problem**: Initial visualizer showed 3D hands in wrong positions, far apart
- **Root Cause**: `joints_3d` is in hand-centric coordinates, not image coordinates
- **Solution**: Added `joints_2d` output to handler.py using perspective projection
  - Computes actual pixel coordinates for each joint
  - Uses HaMeR's camera parameters (focal_length, camera_center)
- **Handler Update**: `naman188/hamer-runpod:latest_8` now outputs:
  - `joints_3d`: 21x3 hand pose in hand-centric coords
  - `joints_2d`: 21x2 pixel coordinates for video overlay
  - `camera_t`: Camera translation for 3D positioning
  - `bbox`: Detection bounding box
- **3D Visualization**: Uses `camera_t` X/Y for relative positioning, `joints_3d` for hand shape
- **2D Overlay**: Uses `joints_2d` directly for accurate pixel positions

---

## Key Files

### Docker Setup (`hamer_runpod/`)
| File | Purpose |
|------|---------|
| `Dockerfile.base` | Base image with all dependencies and models |
| `Dockerfile.handler` | Thin layer with just handler.py |
| `build.sh` | Build script with `base`, `handler`, `push` commands |
| `handler.py` | RunPod serverless handler with R2 support + joints_2d |

### Client Scripts (`hamer_runpod/`)
| File | Purpose |
|------|---------|
| `hamer_client.py` | Upload to R2, trigger RunPod, save JSON |
| `hamer_visualizer.py` | Display 3D hand skeletons + 2D overlay in Rerun |
| `hamer_pipeline.py` | End-to-end pipeline for any video |

### Credentials (`.env` - gitignored)
- `S3_ENDPOINT_URL` - Cloudflare R2 endpoint
- `S3_ACCESS_KEY_ID` - R2 access key
- `S3_SECRET_ACCESS_KEY` - R2 secret key
- `RUNPOD_ENDPOINT_ID` - foah778z6qr2zd
- `RUNPOD_API_KEY` - RunPod API key

---

## Technical Decisions

| Decision | Reasoning |
|----------|-----------|
| HaMeR over MediaPipe | More robust on difficult poses, 3D mesh output |
| RunPod over local | HaMeR requires GPU, pyrender OpenGL fails on Mac |
| Cloudflare R2 over AWS S3 | Free tier, no AWS account issues |
| RunPod base image | Fixes user namespace UID mapping errors |
| 2-layer Docker | Fast iteration on handler changes |
| joints_2d in handler | Proper 2D projection using camera intrinsics |

---

## Current Status
- [x] HaMeR installed locally (for reference)
- [x] handler.py created with R2 support + joints_2d
- [x] Cloudflare R2 bucket: `egocentric-videos`
- [x] RunPod endpoint created: `foah778z6qr2zd`
- [x] Base Docker image: `naman188/hamer-runpod:base`
- [x] Production image: `naman188/hamer-runpod:latest_8`
- [x] End-to-end pipeline tested (04.mp4 @ 5fps, 10fps, 30fps)
- [x] Rerun visualizer with 3D hands + 2D overlay
- [ ] Fine-tune 3D hand positioning in visualizer

---

## Commands Reference

```bash
# Build base image (once, ~30 min)
cd hamer_runpod && ./build.sh base

# Build handler layer (fast, after code changes)
./build.sh handler

# Push to Docker Hub
./build.sh push

# Process a video
python hamer_client.py /path/to/video.mp4 --fps 30

# Visualize results
python hamer_visualizer.py video_hamer.json --video /path/to/video.mp4

# Open saved visualization
rerun hamer_final_v6.rrd
```

---

## Docker Images
- `naman188/hamer-runpod:base` - Base with all dependencies + models
- `naman188/hamer-runpod:latest_8` - Production image with joints_2d support

---

## Output JSON Format
```json
{
  "video": "04.mp4",
  "fps_processed": 10,
  "frames": [{
    "frame_idx": 0,
    "timestamp_ms": 0,
    "hands": [{
      "side": "left",
      "joints_3d": [[x,y,z], ...],  // 21 joints in hand-centric coords
      "joints_2d": [[x,y], ...],     // 21 joints in pixel coords
      "camera_t": [x, y, z],
      "bbox": [x1, y1, x2, y2]
    }]
  }]
}
```

---

*Last Updated: 2025-12-29*
