# HaMeR Hand Tracking Pipeline

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-supported-2496ED)
![RunPod](https://img.shields.io/badge/RunPod-serverless-8A2BE2)

🖐️ HaMeR-based hand tracking pipeline with RunPod serverless GPU integration.

## Overview
This repository contains a robust pipeline for hand tracking and pose estimation using the HaMeR (Hand Mesh Recovery) model. It is designed to process videos and extract high-fidelity 3D hand poses, with support for offloading heavy computation to serverless GPUs via RunPod.

## Features
- **HaMeR Integration**: State-of-the-art hand mesh recovery.
- **Serverless Processing**: Built-in scripts to deploy and execute inference on RunPod.
- **Video Processing**: End-to-end processing of video inputs to extracted pose data.
- **Visualization**: Tools (like Rerun visualizer) to inspect the 3D hand tracking results out-of-the-box.
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
