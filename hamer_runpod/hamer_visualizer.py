"""
HaMeR Rerun Visualizer v3 - Matching MediaPipe Style
Proper 3D visualization + 2D video overlay like the MediaPipe visualizer.

Usage:
    python hamer_visualizer.py 04_hamer_30fps.json --video videos/04.mp4
"""

import json
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from typing import Optional, List, Dict

# MANO skeleton
JOINT_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

# Apple EgoDex style colors
DOT_COLOR = [100, 255, 100]  # Lime green
LINE_COLOR = [50, 150, 255]  # Blue
LEFT_COLOR = [50, 150, 255]   # Blue for left
RIGHT_COLOR = [255, 100, 50]  # Orange for right


def load_results(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def transform_joints_for_3d(hand: dict, img_width: int, img_height: int) -> np.ndarray:
    """
    Transform HaMeR 3D joints to Rerun coordinate space.
    Use joints_3d for shape, camera_t X/Y for relative positioning.
    """
    joints = np.array(hand["joints_3d"])
    camera_t = np.array(hand["camera_t"])
    side = hand["side"]
    
    # Scale joints uniformly to preserve hand shape
    shape_scale = 2.0
    
    # Mirror X for right hand
    x_mult = -1.0 if side == "right" else 1.0
    
    # Use camera_t X/Y to position hands relative to each other
    # (ignore Z which is too large and would cause separation)
    pos_scale = 3.0  # Scale up small X/Y differences
    offset_x = camera_t[0] * pos_scale * x_mult
    offset_y = -camera_t[1] * pos_scale
    
    # Transform joints
    result = np.zeros_like(joints)
    result[:, 0] = joints[:, 0] * shape_scale * x_mult + offset_x
    result[:, 1] = -joints[:, 1] * shape_scale + offset_y
    result[:, 2] = -joints[:, 2] * shape_scale
    
    return result


def log_hand_3d(hand_key: str, positions: np.ndarray, side: str):
    """Log 3D hand skeleton in Apple EgoDex style."""
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    
    # Tiny dots
    radii = [0.008 if JOINT_NAMES[i].endswith("TIP") else 0.006 for i in range(21)]
    rr.log(
        f"world/{hand_key}/joints",
        rr.Points3D(positions, colors=[color] * 21, radii=radii)
    )
    
    # Thin lines
    segments = [[positions[s], positions[e]] for s, e in SKELETON_CONNECTIONS
                if s < len(positions) and e < len(positions)]
    rr.log(
        f"world/{hand_key}/skeleton",
        rr.LineStrips3D(segments, colors=[color], radii=0.004)
    )


def get_joints_2d(hand: dict) -> np.ndarray:
    """
    Get 2D joint coordinates - use joints_2d if available (from handler),
    otherwise fall back to bbox-based projection.
    """
    # Check if we have direct 2D coordinates from the handler
    if "joints_2d" in hand:
        return np.array(hand["joints_2d"])  # Already pixel coordinates!
    
    # Fallback: project 3D into bbox (less accurate)
    joints_3d = np.array(hand["joints_3d"])
    bbox = hand["bbox"]
    side = hand["side"]
    
    x1, y1, x2, y2 = bbox
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    xy = joints_3d[:, :2].copy()
    if side == "left":
        xy[:, 0] = -xy[:, 0]
    
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1
    xy_norm = (xy - xy_min) / xy_range
    
    pad = 0.05
    pts = np.zeros((len(joints_3d), 2))
    pts[:, 0] = x1 + bbox_w * (pad + xy_norm[:, 0] * (1 - 2*pad))
    pts[:, 1] = y1 + bbox_h * (pad + (1 - xy_norm[:, 1]) * (1 - 2*pad))
    
    return pts


def draw_hand_2d(frame: np.ndarray, pts: np.ndarray, side: str):
    """Draw hand skeleton on frame."""
    color = tuple(LEFT_COLOR if side == "left" else RIGHT_COLOR)
    
    for i, j in SKELETON_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(frame, tuple(pts[i].astype(int)), tuple(pts[j].astype(int)), color, 2, cv2.LINE_AA)
    
    for i, pt in enumerate(pts):
        r = 5 if JOINT_NAMES[i].endswith("TIP") else 4
        cv2.circle(frame, tuple(pt.astype(int)), r, color, -1, cv2.LINE_AA)


def visualize(results: dict, video_path: str = None, save_path: str = None):
    """Main visualization."""
    rr.init("HaMeR Hand Tracking", spawn=(save_path is None))
    
    # Layout like MediaPipe visualizer
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D Hands", origin="world"),
        rrb.Vertical(
            rrb.Spatial2DView(name="Video", origin="video"),
            rrb.TextDocumentView(name="Info", origin="info"),
        ),
        column_shares=[2, 1]
    )
    rr.send_blueprint(blueprint)
    
    frames = results.get("frames", [])
    total_frames = len(frames)
    hands_total = sum(len(f.get("hands", [])) for f in frames)
    
    rr.log("info/stats", rr.TextDocument(
        f"**Frames:** {total_frames}\n**Hands:** {hands_total}\n**Joints:** MANO-21",
        media_type=rr.MediaType.MARKDOWN
    ), static=True)
    
    cap = None
    img_w, img_h = 640, 480
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Visualizing {len(frames)} frames...")
    
    for idx, frame_data in enumerate(frames):
        frame_idx = frame_data["frame_idx"]
        ts = frame_data.get("timestamp_ms", frame_idx * 33.33)
        hands = frame_data.get("hands", [])
        
        rr.set_time_seconds("time", ts / 1000.0)
        rr.set_time_sequence("frame", frame_idx)
        
        # Get video frame
        frame_rgb = None
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process each hand
        for hand in hands:
            side = hand["side"]
            joints_3d = np.array(hand["joints_3d"])
            bbox = hand["bbox"]
            
            # 3D visualization
            pos_3d = transform_joints_for_3d(hand, img_w, img_h)
            log_hand_3d(side, pos_3d, side)
            
            # 2D overlay
            if frame_rgb is not None:
                pts_2d = get_joints_2d(hand)
                draw_hand_2d(frame_rgb, pts_2d, side)
        
        if frame_rgb is not None:
            rr.log("video/frame", rr.Image(frame_rgb))
        
        if idx % 50 == 0:
            print(f"  Frame {idx}/{len(frames)}")
    
    if cap:
        cap.release()
    
    if save_path:
        rr.save(save_path)
        print(f"✅ Saved to: {save_path}")
        print(f"   Open: rerun {save_path}")
    else:
        print("✅ Done!")


def visualize_comparison(results_a: dict, results_b: dict, video_path: str = None,
                        label_a: str = "5fps", label_b: str = "30fps"):
    """Compare two results side-by-side."""
    rr.init("HaMeR Comparison", spawn=True)
    
    blueprint = rrb.Horizontal(
        rrb.Spatial2DView(name=label_a, origin=f"video/{label_a}"),
        rrb.Spatial2DView(name=label_b, origin=f"video/{label_b}"),
    )
    rr.send_blueprint(blueprint)
    
    frames_a = {f.get("timestamp_ms", f["frame_idx"]*33.33): f for f in results_a.get("frames", [])}
    frames_b = {f.get("timestamp_ms", f["frame_idx"]*33.33): f for f in results_b.get("frames", [])}
    all_ts = sorted(set(frames_a.keys()) | set(frames_b.keys()))
    
    cap = None
    fps = 30
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
    
    for ts in all_ts:
        rr.set_time_seconds("time", ts / 1000.0)
        
        if cap:
            frame_idx = int(ts / 1000.0 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # A
                fa = frame_rgb.copy()
                if ts in frames_a:
                    for h in frames_a[ts].get("hands", []):
                        pts = get_joints_2d(h)
                        draw_hand_2d(fa, pts, h["side"])
                rr.log(f"video/{label_a}/frame", rr.Image(fa))
                
                # B
                fb = frame_rgb.copy()
                if ts in frames_b:
                    for h in frames_b[ts].get("hands", []):
                        pts = get_joints_2d(h)
                        draw_hand_2d(fb, pts, h["side"])
                rr.log(f"video/{label_b}/frame", rr.Image(fb))
    
    if cap:
        cap.release()
    print("✅ Comparison done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HaMeR Rerun Visualizer")
    parser.add_argument("json_files", nargs="+", help="HaMeR JSON (1 or 2 files)")
    parser.add_argument("--video", "-v", help="Video for overlay")
    parser.add_argument("--save", "-s", help="Save to .rrd")
    parser.add_argument("--labels", "-l", nargs=2, default=["5fps", "30fps"])
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("HaMeR Rerun Visualizer v3")
    print("=" * 50)
    
    if len(args.json_files) == 1:
        results = load_results(args.json_files[0])
        visualize(results, video_path=args.video, save_path=args.save)
    elif len(args.json_files) == 2:
        results_a = load_results(args.json_files[0])
        results_b = load_results(args.json_files[1])
        visualize_comparison(results_a, results_b, video_path=args.video,
                           label_a=args.labels[0], label_b=args.labels[1])
