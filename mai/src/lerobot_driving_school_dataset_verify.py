#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# -------------------
# Config
# -------------------
REPO_ID = "mai/lerobot-driving-school"
ROOT_PATH = "/datasets/mai/lerobot-driving-school"
OUTPUT_DIR = "./mai/outputs/lerobot_driving_school_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_AS_MP4 = True
FPS = 10


# -------------------
# Visualization
# -------------------

def create_episode_vis(ep_id, episode, save_as_mp4=SAVE_AS_MP4):
    """Create MP4/GIF for a single episode."""
    frames = []
    traj, traj_smooth, targets = [], [], []

    # Collect trajectories
    for frame in episode:
        action = frame["action"].numpy()
        action_raw = frame["action_no_smoothing"].numpy()
        target = frame["targets"].numpy()

        traj.append(action_raw[:, :2])
        traj_smooth.append(action[:, :2])
        targets.append(target)

    traj = np.array(traj)
    traj_smooth = np.array(traj_smooth)
    targets = np.array(targets)

    # Iterate frames for plotting
    for idx, frame in enumerate(episode):
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # --- Left Top: video frame ---
        ax_frame = fig.add_subplot(gs[0, 0])
        img = frame["observation.images.top"].numpy().transpose(1, 2, 0)
        ax_frame.imshow(img)
        ax_frame.axis("off")
        ax_frame.set_title(f"Episode {ep_id} | Frame {idx}")

        # --- Left Bottom: zoomed trajectory ---
        ax_zoom = fig.add_subplot(gs[1, 0])
        ax_zoom.plot(traj[idx, :, 0], traj[idx, :, 1], "k--", alpha=0.6, label="Raw traj")
        ax_zoom.plot(traj_smooth[idx, :, 0], traj_smooth[idx, :, 1], "g-", alpha=0.8, label="Smoothed traj")
        ax_zoom.scatter(targets[idx, :, 0], targets[idx, :, 1], c="blue", marker="x", s=60, label="Targets")
        ax_zoom.set_title("Zoomed-in trajectory")
        ax_zoom.set_aspect("equal", adjustable="box")
        ax_zoom.legend()
        ax_zoom.grid(True)

        # --- Right: full trajectory ---
        ax_full = fig.add_subplot(gs[:, 1])
        ax_full.plot(traj[:, :, 0].flatten(), traj[:, :, 1].flatten(), "k--", alpha=0.4, label="Raw traj")
        ax_full.plot(traj_smooth[:, :, 0].flatten(), traj_smooth[:, :, 1].flatten(), "g-", alpha=0.7, label="Smoothed traj")
        ax_full.scatter(targets[:, :, 0].flatten(), targets[:, :, 1].flatten(), c="blue", marker="x", s=40, label="Targets")
        ax_full.set_xlabel("X (m)")
        ax_full.set_ylabel("Y (m)")
        ax_full.set_title("Full trajectory")
        ax_full.axis("equal")
        ax_full.grid(True)
        ax_full.legend()

        # Convert plot to numpy frame
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img_np = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
        frames.append(img_np)
        plt.close(fig)

    # Save as MP4/GIF
    out_path = os.path.join(OUTPUT_DIR, f"episode_{ep_id:06d}.{'mp4' if save_as_mp4 else 'gif'}")
    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"✅ Saved {out_path}")


# -------------------
# Main
# -------------------

def main():
    dataset = LeRobotDataset(REPO_ID, root=ROOT_PATH, video_backend="pyav")
    print(f"Loaded dataset with {len(dataset)} frames across {dataset.num_episodes} episodes")

    current_ep = None
    buffer = []

    # Stream through dataset frame by frame
    for frame in tqdm(dataset, desc="Streaming frames"):
        ep_idx = frame["episode_index"].item()

        if current_ep is None:
            # start first episode
            current_ep = ep_idx

        if ep_idx != current_ep:
            # flush previous episode
            create_episode_vis(current_ep, buffer)
            buffer = []
            current_ep = ep_idx

        buffer.append(frame)

    # flush last episode
    if buffer:
        create_episode_vis(current_ep, buffer)


if __name__ == "__main__":
    main()
