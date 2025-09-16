import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
from collections import defaultdict
from datasets import load_dataset
from mai.utils.utils import target_in_ego, ego_xy_geodesy, ego_xy


# -------------------
# Config
# -------------------
DATASET_NAME = "yaak-ai/lerobot-driving-school"
VIDEO_BASE_PATH = "/home/user_lerobot/.cache/huggingface/lerobot/yaak-ai/lerobot-driving-school/videos/chunk-000/observation.images.front_left"

FUTURE_LENGTH = 50   # number of future frames to use for actions
VISUALIZE = True     # toggle visualization


# -------------------
# Processing functions
# -------------------

def process_episode(examples, future_length=FUTURE_LENGTH):
    """
    Process all samples in a single episode.
    For each sample:
      - Compute target_waypoint (last waypoint of current frame in ego frame).
      - Compute future ego positions (actions) relative to current frame.
      - Pad if fewer than `future_length` future steps are available.
    """
    processed = []

    for i, ex in enumerate(examples):
        # Current ego pose (anchor for ego frame)
        ego_lat = ex["observation.state.vehicle"][3]
        ego_lon = ex["observation.state.vehicle"][4]
        ego_heading_deg = ex["observation.state.vehicle"][1]

        # --- target_waypoint: last waypoint of CURRENT frame ---
        wp_lon = ex["observation.state.waypoints"][0][-1]
        wp_lat = ex["observation.state.waypoints"][1][-1]
        x_wp, y_wp = ego_xy(ego_lat, ego_lon, ego_heading_deg, wp_lat, wp_lon)
        target_waypoint = (x_wp, y_wp)

        # --- action: FUTURE ego positions relative to current ego ---
        actions, is_padded = [], []
        for j in range(future_length):
            idx = i + j + 1
            if idx < len(examples):
                next_ex = examples[idx]
                fut_lat = next_ex["observation.state.vehicle"][3]
                fut_lon = next_ex["observation.state.vehicle"][4]
                xf, yl = ego_xy(ego_lat, ego_lon, ego_heading_deg, fut_lat, fut_lon)
                actions.append([xf, yl])
                is_padded.append(False)
            else:
                actions.append([0.0, 0.0])
                is_padded.append(True)

        # --- Metadata ---
        episode_idx = ex["episode_index"]
        frame_idx = ex["frame_index"]
        video_path = f"{VIDEO_BASE_PATH}/episode_{episode_idx:06d}.mp4"

        prompt = (
            f"Current speed: {ex['observation.state.vehicle'][0]:.3f}. "
            f"Target (ego frame): x={target_waypoint[0]:.2f}, y={target_waypoint[1]:.2f}. "
            f"Next {future_length} actions are future ego positions in the current ego frame. "
            "Drive to the targets while adhering to traffic rules and regulations."
        )

        processed.append({
            "video_path": video_path,
            "frame_index": frame_idx,
            "episode_index": episode_idx,
            "speed": ex['observation.state.vehicle'][0],
            "target_waypoint": target_waypoint,
            "action": np.array(actions, dtype=np.float32),
            "is_padded": np.array(is_padded, dtype=np.bool_),
            "prompt": prompt,
            "target_text": ex['task.instructions'],
        })

    return processed


def group_by_episode(dataset_split):
    """
    Group HuggingFace dataset split by episode index.
    Returns a dict: {episode_index: [samples]}
    """
    episodes = defaultdict(list)
    for ex in dataset_split:
        episodes[ex["episode_index"]].append(ex)
    return episodes


# -------------------
# Visualization
# -------------------

def visualize_sample(sample, save_path="./mai/outputs/pngs/example_trajectory.png"):
    """
    Visualize a single processed sample:
      - Top subplot: the corresponding video frame.
      - Bottom subplot: future ego-frame trajectory.
    """
    # Extract frame using imageio
    reader = imageio.get_reader(sample["video_path"], "ffmpeg")
    frame = reader.get_data(sample["frame_index"])  # RGB numpy array
    reader.close()

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # --- Top: video frame ---
    axes[0].imshow(frame)
    axes[0].axis("off")
    axes[0].set_title(f"Video Frame {sample['frame_index']} from Episode {sample['episode_index']}")

    # --- Bottom: trajectory ---
    actions = sample["action"]
    mask = ~sample["is_padded"]

    axes[1].plot(actions[mask, 1], actions[mask, 0], "bo-", label="Future trajectory")
    # axes[1].scatter(*sample["target_waypoint"], color="green", s=100, label="Target waypoint")

    axes[1].set_xlabel("x_fwd (m)")
    axes[1].set_ylabel("y_left (m)")
    axes[1].set_title("Future Ego Trajectory")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = f"./mai/outputs/pngs/frame_{sample['frame_index']}_ep_{sample['episode_index']}.png"
    plt.savefig(save_path)
    # plt.show()
    print(f"Saved visualization to {save_path}")


# -------------------
# Main
# -------------------

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset(DATASET_NAME)

    # Group by episodes (train split only for now)
    train_ds = dataset["train"]
    episodes = group_by_episode(train_ds)

    # Process all episodes
    all_processed = []
    for ep_id, ep_samples in episodes.items():
        all_processed.extend(process_episode(ep_samples, FUTURE_LENGTH))

    print(f"Total processed samples: {len(all_processed)}")

    # Visualization (pick a random sample)
    if VISUALIZE:
        sample = random.choice(all_processed)
        visualize_sample(sample)
