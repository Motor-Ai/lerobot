import math

def target_in_ego(lat0, lon0, heading_deg, lat_t, lon_t, alt0=None, alt_t=None):
    """
    Convert (lat_t, lon_t, alt_t) into ego-centric (x_fwd, y_left, z_up),
    given ego geodetic pose (lat0, lon0, heading_deg, alt0).

    heading_deg = degrees clockwise from North (0°=North, 90°=East).
    """

    R = 6378137.0  # WGS-84 radius
    lat0_rad = math.radians(lat0)
    dlat = math.radians(lat_t - lat0)
    dlon = math.radians(lon_t - lon0)

    de = dlon * math.cos(lat0_rad) * R  # East
    dn = dlat * R                       # North
    dz = (alt_t - alt0) if (alt0 is not None and alt_t is not None) else 0.0

    psi = math.radians(heading_deg)

    # Rotation North-East -> Ego (Forward, Left)
    x_fwd  =  math.cos(psi) * dn + math.sin(psi) * de
    y_left = -math.sin(psi) * dn + math.cos(psi) * de
    z_up   = dz

    return x_fwd, y_left, z_up

import numpy as np
import math
import pymap3d as pm  

def ego_xy_geodesy(lat0, lon0, heading_deg, lat_t, lon_t, alt0=None, alt_t=None):
    """
    Convert (lat_t, lon_t, alt_t) into ego-centric (x_fwd, y_left, z_up),
    given ego geodetic pose (lat0, lon0, heading_deg, alt0).

    heading_deg = degrees clockwise from North (0°=North, 90°=East).
    """

    # --- Step 1: geodetic -> ENU (east, north, up) in meters ---
    # If altitude is missing, set it to 0.0
    alt0 = alt0 if alt0 is not None else 0.0
    alt_t = alt_t if alt_t is not None else 0.0
    e, n, u = pm.geodetic2enu(lat_t, lon_t, alt_t, lat0, lon0, alt0)

    # --- Step 2: ENU -> ego (forward, left, up) ---
    psi = math.radians(heading_deg)  # heading CW from North
    x_fwd  =  math.cos(psi) * n + math.sin(psi) * e
    y_left = -math.sin(psi) * n + math.cos(psi) * e
    z_up   = u

    return float(x_fwd), float(y_left), float(z_up)

def ego_xy(lat0, lon0, heading_deg, lat_t, lon_t):
    import math, pymap3d as pm
    
    # geodetic -> ENU (east, north)
    e, n, _ = pm.geodetic2enu(lat_t, lon_t, 0.0, lat0, lon0, 0.0)
    
    # rotate ENU -> ego (forward, left)
    psi = math.radians(heading_deg)
    x_fwd  =  math.cos(psi) * n + math.sin(psi) * e
    y_left = -math.sin(psi) * n + math.cos(psi) * e
    
    return float(x_fwd), float(y_left)
