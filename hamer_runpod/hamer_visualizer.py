"""
HaMeR Rerun Visualizer
Display hand tracking results from HaMeR in Rerun with 3D skeletons.
"""

import json
import cv2
import numpy as np
import rerun as rr
from pathlib import Path

# MANO 21 joint names and skeleton connections
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Skeleton connections (joint index pairs)
SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

# Colors for left/right hands
COLORS = {
    "left": (0.2, 0.6, 1.0),   # Blue
    "right": (1.0, 0.4, 0.2),  # Orange
}


def load_results(json_path: str) -> dict:
    """Load HaMeR results from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def visualize_hands(results: dict, video_path: str = None, app_name: str = "HaMeR Hand Tracking"):
    """
    Visualize HaMeR results in Rerun.
    
    Args:
        results: HaMeR output dict with 'frames' list
        video_path: Optional video file to show alongside
        app_name: Rerun application name
    """
    rr.init(app_name, spawn=True)
    
    frames = results.get("frames", [])
    if not frames:
        print("No frames in results")
        return
    
    print(f"Visualizing {len(frames)} frames...")
    
    # Open video if provided
    cap = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    for frame_data in frames:
        frame_idx = frame_data["frame_idx"]
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        hands = frame_data.get("hands", [])
        
        # Set Rerun time
        rr.set_time_seconds("time", timestamp_ms / 1000.0)
        rr.set_time_sequence("frame", frame_idx)
        
        # Log video frame if available
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rr.log("video/frame", rr.Image(frame_rgb))
        
        # Log each hand
        for hand in hands:
            side = hand["side"]
            joints_3d = np.array(hand["joints_3d"])  # [21, 3]
            color = COLORS.get(side, (0.5, 0.5, 0.5))
            
            # Log 3D points
            rr.log(
                f"hands/{side}/joints",
                rr.Points3D(
                    joints_3d,
                    colors=[color] * len(joints_3d),
                    radii=0.005
                )
            )
            
            # Log skeleton lines
            lines = []
            for i, j in SKELETON_CONNECTIONS:
                if i < len(joints_3d) and j < len(joints_3d):
                    lines.append([joints_3d[i], joints_3d[j]])
            
            if lines:
                rr.log(
                    f"hands/{side}/skeleton",
                    rr.LineStrips3D(
                        lines,
                        colors=[color] * len(lines),
                        radii=0.002
                    )
                )
            
            # Log joint labels for first frame only
            if frame_idx == frames[0]["frame_idx"]:
                for idx, name in enumerate(JOINT_NAMES):
                    if idx < len(joints_3d):
                        rr.log(
                            f"hands/{side}/labels/{name}",
                            rr.Points3D([joints_3d[idx]], labels=[name], radii=0.003)
                        )
    
    if cap is not None:
        cap.release()
    
    print("✅ Visualization complete! Check Rerun viewer.")


# ========== Main ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize HaMeR results in Rerun")
    parser.add_argument("json_file", help="Path to HaMeR results JSON")
    parser.add_argument("--video", "-v", help="Optional video file to overlay")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HaMeR Rerun Visualizer")
    print("=" * 50)
    
    results = load_results(args.json_file)
    visualize_hands(results, video_path=args.video)
