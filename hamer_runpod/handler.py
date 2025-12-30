"""
RunPod Serverless Handler for HaMeR Hand Mesh Recovery.
Receives Cloudflare R2 (or S3-compatible) video URL, processes with HaMeR, returns 21 joint coordinates per hand.
"""

import os
import sys
import tempfile
import cv2
import numpy as np
import torch
from pathlib import Path
from urllib.parse import urlparse

# Add HaMeR to path
sys.path.insert(0, '/app/hamer')

import runpod
import boto3

# ========== Global Models (loaded once on cold start) ==========
MODEL = None
MODEL_CFG = None
DETECTOR = None
CPM = None
DEVICE = None

def load_models():
    """Load all models on first call (cold start)."""
    global MODEL, MODEL_CFG, DETECTOR, CPM, DEVICE
    
    if MODEL is not None:
        return  # Already loaded
    
    print("="*50)
    print("Loading HaMeR models (cold start)...")
    print("="*50)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    # Import HaMeR components
    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    from vitpose_model import ViTPoseModel
    
    # SKIP download_models() - models are BAKED INTO the Docker image at /app/hamer/_DATA
    # Calling download_models() would re-download 5.7GB at runtime to ~/.cache/hamer/
    # which fails with UID permission errors on RunPod
    # download_models(CACHE_DIR_HAMER)  # DISABLED - models already in image
    
    # Load HaMeR
    print("Loading HaMeR model...")
    MODEL, MODEL_CFG = load_hamer(DEFAULT_CHECKPOINT)
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    # Load ViTDet detector
    print("Loading ViTDet detector...")
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    DETECTOR = DefaultPredictor_Lazy(detectron2_cfg)
    
    # Load ViTPose for hand keypoints
    print("Loading ViTPose...")
    CPM = ViTPoseModel(DEVICE)
    
    print("="*50)
    print("All models loaded successfully!")
    print("="*50)

def download_from_storage(storage_url: str, local_path: str):
    """Download file from Cloudflare R2 or S3-compatible storage."""
    parsed = urlparse(storage_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    # Get endpoint URL for R2 (or leave None for AWS S3)
    endpoint_url = os.environ.get('S3_ENDPOINT_URL')  # For R2: https://<account_id>.r2.cloudflarestorage.com
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get('S3_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('S3_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('S3_REGION', 'auto')
    )
    
    print(f"Downloading from {bucket}/{key}...")
    s3_client.download_file(bucket, key, local_path)
    print(f"Downloaded to {local_path}")

def process_frame(img_cv2: np.ndarray, frame_idx: int) -> dict:
    """Process a single frame with HaMeR pipeline."""
    global MODEL, MODEL_CFG, DETECTOR, CPM, DEVICE
    
    from hamer.utils import recursive_to
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils.renderer import cam_crop_to_full
    
    result = {
        "frame_idx": frame_idx,
        "hands": []
    }
    
    # Step 1: Detect humans
    det_out = DETECTOR(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]  # BGR to RGB
    
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
    
    if len(pred_bboxes) == 0:
        return result
    
    # Step 2: Detect hand keypoints with ViTPose
    vitposes_out = CPM.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )
    
    bboxes = []
    is_right = []
    
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]
        
        # Left hand
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), 
                    keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        
        # Right hand
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), 
                    keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)
    
    if len(bboxes) == 0:
        return result
    
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    
    # Step 3: Run HaMeR on detected hands
    dataset = ViTDetDataset(MODEL_CFG, img_cv2, boxes, right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    for batch in dataloader:
        batch = recursive_to(batch, DEVICE)
        with torch.no_grad():
            out = MODEL(batch)
        
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            side = "right" if batch['right'][n].item() else "left"
            
            # Get 3D joints (21 joints per hand)
            pred_keypoints_3d = out['pred_keypoints_3d'][n]  # [21, 3] - keep as tensor for projection
            
            # Get camera translation for world coordinates
            multiplier = (2 * batch['right'][n] - 1)
            pred_cam = out['pred_cam'][n].clone()
            pred_cam[1] = multiplier * pred_cam[1]
            
            box_center = batch["box_center"][n].float()
            box_size = batch["box_size"][n].float()
            img_size = batch["img_size"][n].float()
            scaled_focal_length = MODEL_CFG.EXTRA.FOCAL_LENGTH / MODEL_CFG.MODEL.IMAGE_SIZE * img_size.max()
            
            # Compute camera translation in full image space
            pred_cam_t_full = cam_crop_to_full(
                pred_cam.unsqueeze(0), 
                box_center.unsqueeze(0), 
                box_size.unsqueeze(0), 
                img_size.unsqueeze(0), 
                scaled_focal_length
            )
            
            # === Project 3D joints to 2D image coordinates ===
            # Build camera intrinsics
            focal_length = torch.tensor([[scaled_focal_length, scaled_focal_length]], device=DEVICE)
            camera_center = img_size.flip(0).unsqueeze(0) / 2  # [cx, cy]
            
            # Project: joints_3d + camera_t -> 2D
            joints_world = pred_keypoints_3d.unsqueeze(0) + pred_cam_t_full.unsqueeze(1)
            
            # Perspective projection: project to image plane
            # z = depth, project x,y using focal length
            z = joints_world[:, :, 2:3]  # [1, 21, 1]
            xy = joints_world[:, :, :2]   # [1, 21, 2]
            
            # Apply projection: x_2d = fx * x / z + cx
            joints_2d = xy / z * focal_length.unsqueeze(1) + camera_center.unsqueeze(1)
            joints_2d = joints_2d[0].detach().cpu().numpy()  # [21, 2]
            
            pred_cam_t = pred_cam_t_full.detach().cpu().numpy()[0]
            
            result["hands"].append({
                "side": side,
                "joints_3d": pred_keypoints_3d.detach().cpu().numpy().tolist(),  # 21 joints × 3 coords
                "joints_2d": joints_2d.tolist(),  # 21 joints × 2 coords (pixel positions!)
                "camera_t": pred_cam_t.tolist(),
                "bbox": boxes[n].tolist()
            })
    
    return result

def process_video(video_path: str, fps_sample: int = 5) -> list:
    """Process entire video and extract hand coordinates."""
    load_models()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(original_fps / fps_sample))
    
    print(f"Video: {total_frames} frames @ {original_fps:.1f} FPS")
    print(f"Sampling every {frame_skip} frames → {fps_sample} FPS output")
    
    results = []
    frame_idx = 0
    processed = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            result = process_frame(frame, frame_idx)
            result["timestamp_ms"] = (frame_idx / original_fps) * 1000
            results.append(result)
            processed += 1
            
            if processed % 10 == 0:
                print(f"Processed {processed} frames...")
        
        frame_idx += 1
    
    cap.release()
    print(f"Done! Processed {processed} frames total")
    return results

# ========== RunPod Handler ==========

def handler(job: dict) -> dict:
    """
    RunPod serverless handler.
    
    Input:
        {
            "input": {
                "s3_url": "s3://bucket/path/to/video.mp4",
                "fps": 5  # optional, default 5
            }
        }
    
    Output:
        {
            "frames": [
                {
                    "frame_idx": 0,
                    "timestamp_ms": 0.0,
                    "hands": [
                        {
                            "side": "left",
                            "joints_3d": [[x,y,z], ...],  # 21 joints
                            "camera_t": [tx, ty, tz],
                            "bbox": [x1, y1, x2, y2]
                        }
                    ]
                }
            ]
        }
    """
    try:
        job_input = job.get("input", {})
        storage_url = job_input.get("storage_url") or job_input.get("s3_url")  # Support both
        fps = job_input.get("fps", 5)
        
        if not storage_url:
            return {"error": "Missing storage_url in input"}
        
        # Create temp file for video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
        
        # Download from R2/S3
        download_from_storage(storage_url, video_path)
        
        # Process video
        results = process_video(video_path, fps_sample=fps)
        
        # Cleanup
        os.unlink(video_path)
        
        return {"frames": results, "total_frames": len(results)}
    
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# Start RunPod serverless
if __name__ == "__main__":
    print("Starting HaMeR RunPod Handler...")
    runpod.serverless.start({"handler": handler})
