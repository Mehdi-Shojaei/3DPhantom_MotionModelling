import numpy as np
# from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def phantom3d(definition: np.ndarray= None, n=128):
    """
    3D ellipsoid phantom, with per-ellipsoid rotation.
    definition: 
      - If string, falls back to unrotated 3D Shepp–Logan (no rotations applied).
      - Otherwise must be an (m×10) array: [A,a,b,c,x0,y0,z0, phi,theta,psi]
        phi,theta,psi in degrees (Z–Y–X Euler sequence).
    n: volume size (n,n,n).
    """
    # # Built-in (ignore rotations)
    # if isinstance(definition, str):
    #     # ... fall back to the previous phantom3d implementation ...
    #     from your_module import phantom3d
    #     return phantom3d(definition, n)

    if definition is None:
        E = np.array([
            [1140, 0.9,  0.6,   5,  0,   0, 0, 0, 0, 0], # skin/fat
            [320, 0.85, 0.55, 5, 0,   0, 0, 0, 0, 0], # muscle
            [1140, 0.75, 0.45, 5, 0,   0, 0, 0, 0, 0], # body
            [600, 0.2, 0.1, 0.75, 0.4,   0.18, -0.1, 20, 0, 0], #lk
            [600, 0.2, 0.1, 0.75, -0.35,   0.18, -0.15, -20, -8, 0], #rk
            [2000, 0.05, 0.05, 5, 0,   0.5, 0, 0, 0, 0], # sc
            [700, 0.35,  0.2,   1,  -0.45,   -0.1, 0.2, 45, 2, 0], # liver
            [850, 0.35,  0.1,   0.5,  0.25,   -0.1, 0.7, -45, 2, 0], # Stomach
            [800, 0.1,  0.1,   0.4,  0,   -0.05, 0.5, 0, 10, 5], # GTV
            ], dtype=float)

    # if definition is None:
    #     E = np.array(definition, float)
    if E.ndim!=2 or E.shape[1]!=10:
        raise ValueError("Need shape (m,10): [A,a,b,c,x0,y0,z0,phi,theta,psi]")

    # Create normalized grid
    coords = np.linspace(-1,1,n)
    Z, Y, X = np.meshgrid(coords, coords, coords, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (n³×3)

    vol = np.zeros(n**3, dtype=float)

    # Precompute degree→rad
    deg2rad = np.pi/180.0

    for (A, a, b, c, x0, y0, z0, phi, theta, psi) in E:
        # Build rotation matrix R = R_x(psi) @ R_y(theta) @ R_z(phi)
        p = phi * deg2rad
        t = theta * deg2rad
        s = psi * deg2rad
        Rz = np.array([[ np.cos(p), -np.sin(p), 0],
                       [ np.sin(p),  np.cos(p), 0],
                       [        0,         0,  1]])
        Ry = np.array([[ np.cos(t), 0, np.sin(t)],
                       [        0,  1,        0],
                       [-np.sin(t), 0, np.cos(t)]])
        Rx = np.array([[1,         0,          0],
                       [0, np.cos(s), -np.sin(s)],
                       [0, np.sin(s),  np.cos(s)]])
        R = Rx @ Ry @ Rz

        # Translate & rotate all points
        centered = pts - np.array([x0, y0, z0])
        primed = centered @ R.T  # shape (n³,3)

        # Ellipsoid test
        vals = (primed[:,0]/a)**2 + (primed[:,1]/b)**2 + (primed[:,2]/c)**2
        mask = (vals <= 1.0)
        vol[mask] = A

    return vol.reshape((n, n, n))
