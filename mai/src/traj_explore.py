import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
import imageio
from collections import defaultdict
from datasets import load_dataset

import math
import cv2 

from mai.utils.dataset import smooth_traj_global

def ego_xy_geodesy(lat0, lon0, lat, lon):
    """
    Convert (lat, lon) into local ENU coordinates (x_east, y_north) in meters,
    relative to a reference point (lat0, lon0).

    Args:
        lat0, lon0 : float
            Reference latitude and longitude in degrees (origin).
        lat, lon : float
            Target latitude and longitude in degrees.

    Returns:
        (x_east, y_north) in meters
    """
    # Earth radius (WGS-84 approximate)
    R = 6378137.0  

    # Convert degrees to radians
    lat0_rad = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)

    # ENU projection (flat Earth approx, good for small distances)
    x_east = dlon * math.cos(lat0_rad) * R
    y_north = dlat * R

    return x_east, y_north


# -------------------
# Config
# -------------------
DATASET_NAME = "yaak-ai/lerobot-driving-school"
VIDEO_BASE_PATH = "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_left"
OUTPUT_DIR = "./mai/outputs/gifs_corrected_traj"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_AS_MP4 = True


# -------------------
# Helpers
# -------------------

def group_by_episode(dataset_split):
    """Group HuggingFace dataset split by episode index."""
    episodes = defaultdict(list)
    for ex in dataset_split:
        episodes[ex["episode_index"]].append(ex)
    return episodes


def convert_episode_to_global(samples):
    """
    Convert episode into global-frame trajectory and per-timestep targets.

    Returns:
        traj: (N, 3) array of [x, y, yaw]
        targets_mid: (N, 2) array of [x_mid, y_mid]
        targets_final: (N, 2) array of [x_final, y_final]
    """
    if not samples:
        return (np.zeros((0, 3)), np.zeros((0, 2)), np.zeros((0, 2)))

    lat0 = samples[0]["observation.state.vehicle"][3]
    lon0 = samples[0]["observation.state.vehicle"][4]

    traj = []
    targets_mid = []
    targets_final = []
    all_speeds = []

    for ex in samples:
        # --- Ego pose ---
        lat = ex["observation.state.vehicle"][3]
        lon = ex["observation.state.vehicle"][4]
        heading_deg = ex["observation.state.vehicle"][1]
        x, y = ego_xy_geodesy(lat0, lon0, lat, lon)
        yaw = np.deg2rad(90 - heading_deg)  # convert dataset heading → math yaw
        traj.append([x, y, yaw])

        # --- Targets ---
        target_mid_index = len(ex["observation.state.waypoints"][0]) // 2

        # Mid waypoint
        wp_lat_mid = ex["observation.state.waypoints"][1][int(target_mid_index/2)]
        wp_lon_mid = ex["observation.state.waypoints"][0][int(target_mid_index/2)]
        x_mid, y_mid = ego_xy_geodesy(lat0, lon0, wp_lat_mid, wp_lon_mid)
        targets_mid.append([x_mid, y_mid])

        # Final waypoint
        wp_lat_final = ex["observation.state.waypoints"][1][target_mid_index]
        wp_lon_final = ex["observation.state.waypoints"][0][target_mid_index]
        x_final, y_final = ego_xy_geodesy(lat0, lon0, wp_lat_final, wp_lon_final)
        targets_final.append([x_final, y_final])
        all_speeds.append(ex["observation.state.vehicle"][0])

    
    all_speeds = np.array(all_speeds, dtype=np.float32)
    traj = np.array(traj, dtype=np.float32)
    traj_smooth = smooth_traj_global(traj=traj, speed=all_speeds)  

    
    return (
        all_speeds,
        traj, traj_smooth,
        [np.array(targets_mid, dtype=np.float32),
        np.array(targets_final, dtype=np.float32)]
    )



# -------------------
# GIF Creation
# -------------------

def create_episode_gif(ep_id, samples, traj, traj_smooth, targets, speeds, save_as_mp4=SAVE_AS_MP4):
    """
    Create a GIF or MP4 for one episode:
      - Left top: video frame
      - Left bottom: zoomed-in traj around ego pose
      - Right: full global trajectory

    Args:
        save_as_mp4 (bool): if True saves MP4, else saves GIF
    """
    video_path = f"{VIDEO_BASE_PATH}/episode_{ep_id:06d}.mp4"
    reader = imageio.get_reader(video_path, "ffmpeg")
    frames = []

    for idx, ex in enumerate(samples):
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # --- Left Top: video frame ---
        ax_frame = fig.add_subplot(gs[0, 0])
        frame = reader.get_data(ex["frame_index"])
        scale = 0.5
        h0, w0 = frame.shape[:2]
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ax_frame.imshow(frame)
        ax_frame.axis("off")
        ax_frame.set_title(f"Episode {ep_id} | Frame {idx}\nSpeed: {speeds[idx]:.1f} km/h")

        # --- Left Bottom: zoomed-in traj ---
        ax_zoom = fig.add_subplot(gs[1, 0])
        x, y, yaw = traj[idx]
        xs, ys, yaws = traj_smooth[idx]

        ax_zoom.plot(traj[:, 0], traj[:, 1], "k--", alpha=0.4, label="Raw traj")
        ax_zoom.plot(traj_smooth[:, 0], traj_smooth[:, 1], "g-", alpha=0.8, label="Smoothed traj")
        ax_zoom.quiver(x, y, np.cos(yaw), np.sin(yaw), scale=20, color="red", width=0.01, label="Ego raw")
        ax_zoom.quiver(xs, ys, np.cos(yaws), np.sin(yaws), scale=20, color="green", width=0.01, label="Ego smooth")

        ax_zoom.set_title("Zoomed-in trajectory")
        ax_zoom.set_aspect("equal", adjustable="box")
        zoom_size = 5
        ax_zoom.set_xlim([xs - zoom_size, xs + zoom_size])
        ax_zoom.set_ylim([ys - zoom_size, ys + zoom_size])
        ax_zoom.grid(True)
        ax_zoom.legend(loc="upper right")

        # --- Right: full global trajectory ---
        ax_full = fig.add_subplot(gs[:, 1])
        ax_full.plot(traj[:, 0], traj[:, 1], "k--", alpha=0.6, label="Raw traj")
        ax_full.plot(traj_smooth[:, 0], traj_smooth[:, 1], "g-", alpha=0.9, label="Smoothed traj")

        for i in range(len(targets)):
            ax_full.scatter(targets[i][idx, 0], targets[i][idx, 1],
                            c="blue", marker="x", s=80, label="Target")

        ax_full.quiver(x, y, np.cos(yaw), np.sin(yaw), scale=20, color="red", width=0.005, label="Ego raw")
        ax_full.quiver(xs, ys, np.cos(yaws), np.sin(yaws), scale=20, color="green", width=0.005, label="Ego smooth")

        ax_full.set_xlabel("East (m)")
        ax_full.set_ylabel("North (m)")
        ax_full.set_title("Global trajectory")
        ax_full.axis("equal")
        ax_full.grid(True)
        ax_full.legend()

        # Save frame
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        img = img[..., :3]
        frames.append(img)
        plt.close(fig)

    reader.close()

    # Save output
    if save_as_mp4:
        out_path = os.path.join(OUTPUT_DIR, f"episode_{ep_id:06d}.mp4")
        imageio.mimsave(out_path, frames, fps=10, format="mp4")
    else:
        out_path = os.path.join(OUTPUT_DIR, f"episode_{ep_id:06d}.gif")
        imageio.mimsave(out_path, frames, fps=10)

    print(f"Saved {out_path}")


# -------------------
# Main
# -------------------

if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME)
    train_ds = dataset["train"]
    episodes = group_by_episode(train_ds)

    print(f"Loaded {len(episodes)} episodes")

    for ep_id, samples in episodes.items():
        speeds, traj, traj_smooth, targets = convert_episode_to_global(samples)
        create_episode_gif(ep_id, samples, traj, traj_smooth, targets, speeds)
