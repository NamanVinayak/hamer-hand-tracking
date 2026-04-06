# HaMeR Hand Tracking Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-supported-2496ED)
![RunPod](https://img.shields.io/badge/RunPod-serverless-8A2BE2)

🖐️ Egocentric hand tracking pipeline for robotics data collection — powered by HaMeR and RunPod serverless GPUs.

## Overview
This pipeline processes **egocentric (first-person perspective) video** to extract high-fidelity 3D hand mesh data using the HaMeR (Hand Mesh Recovery) model. It is designed for robotics teams collecting human demonstration data for training dexterous manipulation policies — the kind of data needed for learning-from-demonstration workflows (e.g. ALOHA, OpenVLA, ACT).

Heavy inference is offloaded to RunPod serverless GPUs, so you don't need on-premise GPU hardware to run it.

## Use Cases
- **Robot Learning from Demonstration (LfD)** — Capture hand poses from human operators to build manipulation training datasets
- **Teleoperation Data Collection** — Extract structured hand mesh data from egocentric video recordings
- **Dexterous Manipulation Research** — Generate 3D hand pose annotations at scale without a GPU cluster

## Features
- **Egocentric Video Support**: Optimized for first-person camera perspectives used in robotics data collection.
- **HaMeR Integration**: State-of-the-art 3D hand mesh recovery.
- **Serverless Processing**: Offload inference to RunPod — no local GPU required.
- **End-to-End Pipeline**: Video input → 3D pose extraction → structured output ready for downstream training.
- **Visualization**: Rerun-based visualizer to inspect hand tracking results out-of-the-box.
- **Docker Support**: Containerized environment for reproducible execution.

## Setup & Installation
1. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your RunPod and S3 credentials in your environment if you plan to use the serverless pipeline:
- `RUNPOD_API_KEY`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
