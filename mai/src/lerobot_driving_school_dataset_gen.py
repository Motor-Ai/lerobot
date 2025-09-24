#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import imageio
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from collections import defaultdict

from mai.utils.dataset import smooth_traj_global  

# -------------------
# Config
# -------------------
REPO_ID = "mai/lerobot-driving-school"
ROOT_PATH = "/datasets/mai/lerobot-driving-school"
DATASET_NAME = "yaak-ai/lerobot-driving-school"
VIDEO_BASE_PATH = "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_left"
FPS = 10
FUTURE_STEPS = 50
CREATE_NEW_DATASET = True
N_FEATURES = 4  # [x, y, yaw, speed]

# -------------------
# Helpers
# -------------------

def ego_xy_geodesy(lat0, lon0, lat, lon):
    """Convert lat/lon to local ENU (x, y) in meters relative to reference (lat0, lon0)."""
    R_earth = 6378137.0
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = dlon * np.cos(lat0_rad) * R_earth
    y = dlat * R_earth
    return x, y

def convert_to_global(samples):
    """Extract raw traj, smoothed traj, targets, and speeds."""

    lat0, lon0 = samples[0]["observation.state.vehicle"][3:5]
    traj, all_speeds, targets_mid, targets_final = [], [], [], []

    for ex in samples:
        lat, lon, heading = ex["observation.state.vehicle"][3], ex["observation.state.vehicle"][4], ex["observation.state.vehicle"][1]
        x, y = ego_xy_geodesy(lat0, lon0, lat, lon)
        yaw = np.deg2rad(90 - heading)
        traj.append([x, y, yaw])
        all_speeds.append(ex["observation.state.vehicle"][0])

        target_mid_index = len(ex["observation.state.waypoints"][0]) // 2
        wp_lat_mid = ex["observation.state.waypoints"][1][target_mid_index // 2]
        wp_lon_mid = ex["observation.state.waypoints"][0][target_mid_index // 2]
        x_mid, y_mid = ego_xy_geodesy(lat0, lon0, wp_lat_mid, wp_lon_mid)

        wp_lat_final = ex["observation.state.waypoints"][1][target_mid_index]
        wp_lon_final = ex["observation.state.waypoints"][0][target_mid_index]
        x_final, y_final = ego_xy_geodesy(lat0, lon0, wp_lat_final, wp_lon_final)

        targets_mid.append([x_mid, y_mid])
        targets_final.append([x_final, y_final])

    traj = np.array(traj, dtype=np.float32)
    all_speeds = np.array(all_speeds, dtype=np.float32)
    traj_smooth = smooth_traj_global(traj=traj, speed=all_speeds)

    return all_speeds, traj, traj_smooth, np.stack([targets_mid, targets_final], axis=1)

def to_ego(traj, idx, speeds):
    """Convert future steps traj into ego frame relative to current idx."""
    if idx >= len(traj):
        return np.zeros((FUTURE_STEPS, 3)), np.ones(FUTURE_STEPS, dtype=np.int64)

    T0 = np.eye(3)
    x0, y0, yaw0 = traj[idx]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    R0 = np.array([[c, -s], [s, c]])
    padded, future = [], []

    for j in range(idx+1, idx+1+FUTURE_STEPS):
        if j >= len(traj):
            future.append([0, 0, 0, 0])
            padded.append(1)
        else:
            dx, dy = traj[j][0] - x0, traj[j][1] - y0
            rel = R0 @ np.array([dx, dy])
            dyaw = traj[j][2] - yaw0
            future.append([rel[0], rel[1], dyaw, speeds[j]])
            padded.append(0)

    return np.array(future, dtype=np.float32), np.array(padded, dtype=np.int64)
    """Simple rule-based classification."""
    dx, dy, dyaw = future[-1]
    lateral_shift = abs(dy)
    forward_shift = dx

    if forward_shift < 1.0:
        return "stop", "Stop the vehicle."
    elif lateral_shift > 3.0:
        return ("change lane to right", "Change lane to the right.") if dy > 0 else ("change lane to left", "Change lane to the left.")
    elif dx > 5.0:
        return "accelerate", "Accelerate forward."
    elif dx < -2.0:
        return "decelerate", "Decelerate."
    else:
        return "keep speed", "Keep the current speed."

# -------------------
# Main
# -------------------

def main():
    # Features
    features = {
        "observation.images.top": {"dtype": "video", 
                                   "shape": [3, 480, 640], 
                                   "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", 
                              "shape": (1, N_FEATURES), 
                              "names": ["x", "y", "yaw", "speed"]},  # [x, y, yaw, speed]
        "action": {"dtype": "float32", 
                   "shape": (FUTURE_STEPS, N_FEATURES), 
                   "names": ["x", "y", "yaw", "speed"]},
        "action_no_smoothing": {"dtype": "float32", 
                                "shape": (FUTURE_STEPS, N_FEATURES), 
                                "names": ["x", "y", "yaw", "speed"]},
        "action_is_padded": {"dtype": "int64", "shape": (FUTURE_STEPS,)},
        "targets": {"dtype": "float32", 
                    "shape": (2, 2), 
                    "names": ["x", "y"]},  # mid + final
    }

    if CREATE_NEW_DATASET:
        dataset = LeRobotDataset.create(REPO_ID, fps=FPS, features=features, root=ROOT_PATH, use_videos=True)
    else:
        dataset = LeRobotDataset(REPO_ID, root=ROOT_PATH)

    dataset_hf = load_dataset(DATASET_NAME)["train"] # Only train split available 


    # group by episode
    episodes = defaultdict(list)
    for ex in dataset_hf:
        episodes[ex["episode_index"]].append(ex)

    for ep_id, samples in tqdm(episodes.items(), desc="Episodes"):
        speeds, traj, traj_smooth, targets = convert_to_global(samples)
        video_path = f"{VIDEO_BASE_PATH}/episode_{samples[0]['episode_index']:06d}.mp4"
        reader = imageio.get_reader(video_path, "ffmpeg")

        for idx, ex in enumerate(samples):  
            image = reader.get_data(ex["frame_index"])
            image = torch.tensor(image).permute(2, 0, 1) # (C,H,W)
            image = F.interpolate(
                                image.unsqueeze(0),  # add batch dim → [1, 3, 1080, 1920]
                                size=(480, 640),     # target (H, W)
                                mode="bilinear",
                                align_corners=False
                            ).squeeze(0)  # back to [3, 480, 640]

            # state
            state = torch.tensor([[0.0, 0.0, 0.0, speeds[idx]]], dtype=torch.float32)

            # actions
            action, mask = to_ego(traj_smooth, idx, speeds)
            action_raw, _ = to_ego(traj, idx, speeds)

            frame = {
                "observation.images.top": image,
                "observation.state": state,
                "action": torch.tensor(action, dtype=torch.float32),
                "action_no_smoothing": torch.tensor(action_raw, dtype=torch.float32),
                "action_is_padded": torch.tensor(mask, dtype=torch.int64),
                "targets": torch.tensor(targets[idx], dtype=torch.float32),
            }

            dataset.add_frame(frame, task=ex['task.instructions'])

        dataset.save_episode()
        print(f"✅ Saved episode {ep_id}")

if __name__ == "__main__":
    main()
