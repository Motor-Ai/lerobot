# mai/utils/dataset.py

from __future__ import annotations
import math
from collections import deque
from typing import Dict, Iterable, List, Any
import numpy as np
from tqdm import tqdm

# -------------------------
# Keep only input fields you still want to carry through.
# We will OVERWRITE 'task' with a single language instruction.
# -------------------------
keep_keys_input = [
    # 'observation.images.front_left',
    # 'observation.images.map',
    'observation.state.vehicle',
    'observation.state.waypoints',
    'observation.state.timestamp',
    'action.continuous',
    'action.discrete',
    'timestamp',
    'frame_index',
    'episode_index',
    'index',
    'task_index',
    'task',                 # keep, but will be replaced
    'task.policy',          # keep unchanged
    'task.instructions',    # keep unchanged
]

# -------------------------
# Helpers
# -------------------------
def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

def _heading_to_rad(h: float) -> float:
    # Auto-detect degrees vs radians
    return math.radians(h) if abs(h) > 2 * math.pi else h

def _geodetic_local_xy(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> tuple[float, float]:
    # Equirectangular projection (city scale)
    R = 6371000.0
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    x = R * (lon - lon0) * math.cos(lat0)
    y = R * (lat - lat0)
    return x, y

def _classify_turn(dyaw_rad: float, left_deg=8.0, right_deg=-8.0) -> str:
    deg = math.degrees(dyaw_rad)
    if deg >= left_deg:
        return "turn_left"
    if deg <= right_deg:
        return "turn_right"
    return "go_straight"

def _classify_longitudinal(dspeed: float, accel_thr=1.0, decel_thr=-1.0) -> str:
    if dspeed >= accel_thr:
        return "accelerate"
    if dspeed <= decel_thr:
        return "decelerate"
    return "keep_speed"

# -------------------------
# Core
# -------------------------
def preprocess_dataset(
    dataset: Iterable[Dict[str, Any]],
    horizon_frames: int = 10,
) -> List[Dict[str, Any]]:
    """
    Produces per-sample:
      - 'action': np.float32 array of shape (H, 4) with columns:
          [x_rel, y_rel, dyaw_rel, speed]
        where:
          x_rel, y_rel  = position of future ego at t+k in ego(t) frame (meters)
          dyaw_rel      = yaw(t+k) - yaw(t) wrapped to [-pi, pi] (radians)
          speed         = absolute speed at t+k (m/s)
      - 'task': single natural-language instruction, e.g. "turn_left & accelerate"

    Tail frames without a full future horizon are dropped.
    """
    H = horizon_frames
    out: List[Dict[str, Any]] = []

    # Buffers across the sliding window
    state_buf = deque(maxlen=H + 1)   # (x, y, yaw, speed)
    sample_buf = deque(maxlen=H + 1)  # original samples

    lat0 = lon0 = None  # anchor for local projection

    # Index layout in observation.state.vehicle
    IDX_SPEED   = 0
    IDX_HEADING = 1
    IDX_LAT     = 3
    IDX_LON     = 4

    for sample in tqdm(dataset, desc="Preprocessing dataset", unit="frame"):
        sample_buf.append(sample)

        veh = _to_numpy(sample['observation.state.vehicle']).reshape(-1)
        speed  = float(veh[IDX_SPEED])
        yaw    = _wrap_pi(_heading_to_rad(float(veh[IDX_HEADING])))
        lat    = float(veh[IDX_LAT])
        lon    = float(veh[IDX_LON])

        if lat0 is None:
            lat0, lon0 = lat, lon
        x, y = _geodetic_local_xy(lat, lon, lat0, lon0)

        state_buf.append((x, y, yaw, speed))

        # Need H+1 frames to emit one (for the oldest in buffer)
        if len(state_buf) < H + 1:
            continue

        # Reference at time t (oldest in buffer)
        x0, y0, yaw0, v0 = state_buf[0]
        cos0, sin0 = math.cos(yaw0), math.sin(yaw0)

        # Prepare sequences for k=1..H
        xs   = np.array([s[0] for s in state_buf], dtype=np.float64)  # length H+1
        ys   = np.array([s[1] for s in state_buf], dtype=np.float64)
        yaws = np.array([s[2] for s in state_buf], dtype=np.float64)
        vs   = np.array([s[3] for s in state_buf], dtype=np.float64)

        dx = xs[1:] - x0                         # (H,)
        dy = ys[1:] - y0
        x_rel =  cos0 * dx + sin0 * dy           # (H,)
        y_rel = -sin0 * dx + cos0 * dy           # (H,)

        dyaw_seq = np.array([_wrap_pi(yaws[k] - yaw0) for k in range(1, H+1)], dtype=np.float64)  # (H,)
        speed_seq = vs[1:]                        # absolute speed at future steps (H,)

        # Stack into (H, 4): [x_rel, y_rel, dyaw_rel, speed]
        action = np.stack([x_rel, y_rel, dyaw_seq, speed_seq], axis=-1).astype(np.float32)

        # Build instruction using the END of the horizon (net dyaw and dspeed over H)
        dyaw_end   = float(dyaw_seq[-1])
        dspeed_end = float(speed_seq[-1] - v0)
        turn_label = _classify_turn(dyaw_end)
        lon_label  = _classify_longitudinal(dspeed_end)
        instruction = f"{turn_label} & {lon_label}"

        # Emit enriched sample
        # Emit enriched sample
        base = sample_buf[0]
        enriched = {k: base[k] for k in keep_keys_input if k in base}

        # Add current ego state in ego(t) frame: [x=0, y=0, yaw=0, speed=v0]
        enriched['observation.state'] = np.array([0.0, 0.0, 0.0, v0], dtype=np.float32)
        
        # (H,4) action = [x_rel, y_rel, dyaw_rel, speed]
        enriched['action'] = action.astype(np.float32)

        # Replace ONLY 'task' with our language instruction; keep task.policy / task.instructions untouched
        enriched['task'] = instruction

        out.append(enriched)

        # Slide window by 1
        sample_buf.popleft()
        state_buf.popleft()

    return out
