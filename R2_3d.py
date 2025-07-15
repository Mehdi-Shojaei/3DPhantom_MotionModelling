import numpy as np

def compute_R2_3d(
    shape: int = 128,
    frac: float = 40,
    scale: tuple[float, float, float] = (1.6, 1.3, 6.0)
) -> np.ndarray:
    """
    Compute a 3D R2 field on a regular grid of given `shape`, where
      dist = sqrt(((x-cx)/sx)^2 + ((y-cy)/sy)^2 + ((z-cz)/sz)^2)
      R2_x = (1 - dist/frac)  for dist < frac, else 0
      R2_y = -R2_x / 3
      R2_z =  2 * R2_x
    and then stack [R2_z, R2_y, R2_x] in the last axis.

    Parameters
    ----------
    shape : int
        Number of voxels along each axis (produces a shape×shape×shape volume).
    frac : float
        Cut‐off radius in the same units as the coordinate grid.
    scale : (sx, sy, sz)
        Tuple of scaling factors for x, y, and z axes in the distance metric.

    Returns
    -------
    R2_3d : ndarray, shape (shape, shape, shape, 3)
        The computed R2 field, with components ordered (z, y, x).
    """
    # coordinate grid
    coord = np.arange(shape)
    Z, Y, X = np.meshgrid(coord, coord, coord, indexing='ij')

    # centers
    cx = (shape - 1) / 2.0
    cy = (shape - 1) / 2.0
    cz = shape - 1.0

    # unpack scales
    sx, sy, sz = scale

    # compute normalized distance
    dist = np.sqrt(
        ((X - cx) / sx) ** 2 +
        ((Y - cy) / sy) ** 2 +
        ((Z - cz) / sz) ** 2
    )

    # mask inside radius
    mask = dist < frac

    # build R2_x and then the other components
    R2_x = np.zeros((shape, shape, shape), dtype=float)
    R2_x[mask] = 1.0 - dist[mask] / frac

    R2_y = -R2_x / 3.0
    R2_z =  R2_x * 2.0

    # stack in (z, y, x) order on a new last axis
    R2_3d = np.stack([R2_z, R2_y, R2_x], axis=-1)

    return R2_3d
