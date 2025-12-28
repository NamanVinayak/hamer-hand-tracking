# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **egocentric video hand tracking pipeline** for robotics AI training data. The project processes first-person POV videos to extract hand pose data, which is sold to robotics companies training humanoid robots and foundation models.

**Business Context**: This is part of an egocentric data agency leveraging India arbitrage (1/5 US costs) to collect and process video data from Western-style environments (hotels, facilities) in India.

## Core Architecture

### Two-Pipeline System

1. **MediaPipe Pipeline** (Local/Mac) - Lightweight, fast, local processing
   - Main script: [hand_tracker.py](hand_tracker.py)
   - Batch processing: [process_videos.py](process_videos.py)
   - Visualization: [rerun_visualizer.py](rerun_visualizer.py)
   - Output: MANO-21 compatible hand landmarks (21 joints per hand)

2. **HaMeR Pipeline** (Cloud/RunPod) - Production-grade, GPU-based, more robust
   - Location: [hamer_runpod/](hamer_runpod/)
   - Handler: [hamer_runpod/handler.py](hamer_runpod/handler.py)
   - Client: [hamer_runpod/hamer_client.py](hamer_runpod/hamer_client.py)
   - Storage: Cloudflare R2 (S3-compatible)
   - Endpoint: RunPod serverless GPU

### Pipeline Selection Criteria

- **Use MediaPipe when**: Testing locally, prototyping, or hands are clearly visible
- **Use HaMeR when**: Production processing, difficult hand poses, or need 3D mesh output
- **Known limitation**: MediaPipe drops frames on difficult/occluded hand poses → HaMeR is more robust

## Common Commands

### MediaPipe Pipeline (Local)

```bash
# Activate virtual environment
source .venv/bin/activate

# Process single video with visualization
python hand_tracker.py --input videos/sample.mp4 --output-json output/sample_hands.json --output-video output/sample_visualized.mp4

# Batch process directory
python process_videos.py --input videos/ --output output/ --visualize

# Visualize with Rerun.io
python rerun_visualizer.py --input videos/sample.mp4
```

### HaMeR Pipeline (Cloud)

```bash
# Activate HaMeR environment
source hamer_venv/bin/activate

# Build Docker images (two-layer strategy for fast iteration)
cd hamer_runpod
./build.sh base      # Build base image (~30 min, infrequent)
./build.sh handler   # Build handler layer (~1 min, after code changes)
./build.sh push      # Push to Docker Hub

# Process video via RunPod
cd hamer_runpod
python hamer_client.py /path/to/video.mp4 --fps 30

# Visualize HaMeR output
python hamer_visualizer.py video_hamer.json --video /path/to/video.mp4
```

## Critical Technical Details

### HaMeR Docker Image Fix

**Problem**: RunPod's user namespace requires all files owned by root/runpod user, but HaMeR's `fetch_demo_data.sh` preserves UID 1000222 from UT Austin researcher's build environment.

**Solution**: Inline model download in [Dockerfile.base](hamer_runpod/Dockerfile.base) with `tar --no-same-owner` flag.

### NumPy Version Constraint

HaMeR requires `numpy==1.26.4` because `xtcocotools` (ViTPose dependency) is compiled against numpy 1.x. Uses PIP_CONSTRAINT to globally lock numpy version:

```dockerfile
# Set constraint file EARLY to prevent ANY pip command from installing numpy>=2.0
ENV PIP_CONSTRAINT=/tmp/numpy-constraint.txt
RUN echo "numpy==1.26.4" > /tmp/numpy-constraint.txt

# Install numpy 1.26.4 FIRST
RUN pip install --no-cache-dir "numpy==1.26.4"

# Later: Rebuild xtcocotools against numpy 1.x
RUN pip install --no-cache-dir --force-reinstall xtcocotools
```

**Critical**: The constraint file prevents `pip install runpod boto3` (or any other package) from reinstalling numpy 2.x as a transitive dependency.

### RunPod Image Caching Gotcha

After pushing a new Docker image to Docker Hub, **you must restart RunPod workers** to pull the fresh image. Workers cache images aggressively.

Current production image: `naman188/hamer-runpod:latest_2`

### Storage Configuration

Uses Cloudflare R2 (not AWS S3) because:
- 10GB free tier
- S3-compatible API
- No AWS account creation issues

Credentials in `.env` (gitignored):
- `S3_ENDPOINT_URL` - R2 endpoint
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `RUNPOD_ENDPOINT_ID` - foah778z6qr2zd
- `RUNPOD_API_KEY`

## Output Format Standards

Both pipelines output **MANO-21 format** with 21 hand joints:

```
WRIST (0)
THUMB: CMC (1), MCP (2), IP (3), TIP (4)
INDEX: MCP (5), PIP (6), DIP (7), TIP (8)
MIDDLE: MCP (9), PIP (10), DIP (11), TIP (12)
RING: MCP (13), PIP (14), DIP (15), TIP (16)
PINKY: MCP (17), PIP (18), DIP (19), TIP (20)
```

JSON structure:
```json
{
  "metadata": {
    "video_id": "sample",
    "fps": 30,
    "resolution": [1920, 1080],
    "total_frames": 300,
    "frames_with_hands": 285,
    "detection_rate": 95.0,
    "joint_standard": "MANO_21"
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 0,
      "hands": {
        "left": {
          "detected": true,
          "confidence": 0.98,
          "landmarks_3d": [...],  // 21 joints
          "landmarks_2d": [...]   // pixel coordinates
        },
        "right": null
      }
    }
  ]
}
```

## Environment Setup

### MediaPipe Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements:
- mediapipe>=0.10.0
- opencv-python>=4.8.0
- numpy>=1.24.0

### HaMeR Environment
```bash
python3.10 -m venv hamer_venv
source hamer_venv/bin/activate
cd hamer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
bash fetch_demo_data.sh  # Downloads 6GB models
```

**Important**: Download `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de and place in `hamer/_DATA/data/mano/`

## Project Context Files

- [master.md](master.md) - Complete business plan, market analysis, tech stack decisions
- [memory.md](memory.md) - Technical journey, Docker fixes, RunPod setup history
- [logs.txt](logs.txt) - Recent session logs

## Data Deliverables (Customer Requirements)

For robotics customers, deliver:
1. ✅ Raw 4K/60fps egocentric video
2. ✅ Hand tracking JSON (21 joints per hand, 3D coordinates)
3. ✅ Object masks (planned: use Meta SAM 3)
4. ✅ Task labels (planned: Streamlit annotation tool)
5. ⚠️ Depth data (nice-to-have, not critical)
6. ⚠️ Body pose (nice-to-have, not critical)

Current status: Items 1-2 production-ready, items 3-4 planned, items 5-6 deferred based on customer demand.

## Development Workflow

### Adding New Hand Tracking Features

1. Prototype with MediaPipe locally ([hand_tracker.py](hand_tracker.py))
2. Test on sample videos in `videos/` directory
3. Once validated, port to HaMeR if needed for production robustness
4. Update Docker images and push to RunPod

### Modifying RunPod Handler

1. Edit [hamer_runpod/handler.py](hamer_runpod/handler.py)
2. Build handler layer only: `./build.sh handler` (fast)
3. Push: `./build.sh push`
4. **Critical**: Restart RunPod workers to pull fresh image
5. Test with `hamer_client.py`

### Testing Changes

```bash
# Test MediaPipe pipeline
python hand_tracker.py --input videos/test.mp4 --output-json output/test.json

# Test HaMeR locally (requires GPU)
cd hamer
python demo.py --img_folder example_data --out_folder demo_out --batch_size=48

# Test RunPod end-to-end
cd hamer_runpod
python hamer_client.py videos/test.mp4 --fps 5
```

## Known Issues & Workarounds

1. **MediaPipe frame drops on occlusion**: Upgrade to HaMeR for difficult poses
2. **HaMeR pyrender fails on Mac**: OpenGL/EGL missing → use RunPod GPU
3. **RunPod UID mapping errors**: Fixed in Dockerfile.base with `--no-same-owner`
4. **NumPy version conflicts**: Force numpy<2.0 after all installations
5. **RunPod image caching**: Always restart workers after pushing new images

## File Organization

```
.
├── hand_tracker.py          # MediaPipe tracker (MANO-21 output)
├── process_videos.py        # Batch processing script
├── rerun_visualizer.py      # 3D visualization with Rerun.io
├── requirements.txt         # MediaPipe dependencies
├── .venv/                   # MediaPipe virtual environment
├── hamer/                   # HaMeR repository (submodule)
├── hamer_venv/              # HaMeR local environment
├── hamer_runpod/            # RunPod serverless deployment
│   ├── Dockerfile.base      # Base image (models + dependencies)
│   ├── Dockerfile.handler   # Thin handler layer
│   ├── handler.py           # RunPod serverless handler
│   ├── hamer_client.py      # Client for triggering jobs
│   └── build.sh             # Build script (base/handler/push)
├── videos/                  # Input videos
├── output/                  # Processed JSON outputs
├── master.md                # Business plan & market research
└── memory.md                # Technical development history
```

## Competitor Context

When evaluating features or adding capabilities, be aware of:

- **Apple EgoDex**: 829 hours free dataset (May 2025) - sets quality baseline
- **Cortex AI**: YC F25 company doing similar egocentric data (3-month head start)
- **Sensei Robotics**: Marketplace model paying $30/hour for footage

**Differentiation strategy**: India arbitrage (1/5 cost), professional quality, hospitality domain focus.
