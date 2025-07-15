
import numpy as np

def compute_R1_3d(phantom_size=128):
    """
    Extend the 2D R1_local pattern into 3D by introducing
    variation along Z as well. Returns an array of shape
    (Z, Y, X, 3), where the last dim are the local R1 components
    in (x, y, z).
    
    """
    n = phantom_size
    # 1) Build 1D profiles along each axis
    # Y‐axis profiles (yy, yx = xy)
    yy = np.linspace(-1.7, 0.3, n)
    xy = np.ones(n)
    half = n // 2
    xy[:half]  = np.linspace(0.5, 1.0, half)
    xy[-half:] = np.linspace(1.0, 0.5, half)
    yx = xy.copy()

    # X‐axis profile
    xx = np.linspace(-1.0, 1.0, n)

    # Z‐axis profile (we choose a simple linear ramp here; you can
    # replace with any pattern you like, e.g. a cosine or the same
    # half‐ramp as XY)
    zz = np.linspace(-1.0, 1.0, n)
    zy = np.ones(n)
    zy[:half]  = np.linspace(0.5, 1.0, half)
    zy[-half:] = np.linspace(1.0, 0.5, half)

    # 2) Broadcast‐multiply to get three 3D volumes
    # R1_local_x varies in Z by zy, in Y by yx, in X by xx
    R1_x = (zy[:, None, None] *
            yx[None, :,   None] *
            xx[None, None, :])       # shape (Z, Y, X)

    # R1_local_y varies in Z by zy, in Y by yy, in X by xy
    R1_y = (zy[:, None, None] *
            yy[None, :,   None] *
            xy[None, None, :])       # shape (Z, Y, X)

    # R1_local_z varies in Z by zz, in Y by yy, in X by xx
    R1_z = (zz[:, None, None] *
            np.ones(n)[None, :, None] *
            np.ones(n)[None, None, :])       # shape (Z, Y, X)

    # 3) Stack into final array (Z, Y, X, 3)
    R1_local = np.stack([R1_z, R1_y, R1_x], axis=-1)
    return R1_local