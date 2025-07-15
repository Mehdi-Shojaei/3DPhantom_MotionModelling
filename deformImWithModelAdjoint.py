import numpy as np

def deformImWithModelAdjoint(I, R1, R2, s1, s2,
                               def_xs=None, def_ys=None, def_zs=None):
    """
    Adjoint of the 3D deformation model using trilinear interpolation.

    Parameters
    ----------
    I       : ndarray, shape (D, H, W)
              Input 3D volume (complex or real).
    R1, R2  : ndarray, shape (D, H, W, 3)
              Motion model fields for two surrogate signals.
    s1, s2  : scalars
              Surrogate signal values at this timepoint.
    def_xs, def_ys, def_zs : 1D arrays or None
              Optional indices along X, Y, Z to deform. If None, uses full range.

    Returns
    -------
    I_def   : ndarray, shape (D, H, W)
              The adjoint-warped volume.
    weights : ndarray, shape (D, H, W)
              Sum of interpolation weights at each voxel.
    """
    D, H, W = I.shape

    # default to full volume if no region specified
    if ((def_xs is None) != (def_ys is None)) \
    or ((def_xs is None) != (def_zs is None)):
        raise ValueError("Must specify all of def_xs, def_ys, def_zs or none.")
    if def_xs is None:
        # set all three to full ranges
        def_xs = np.arange(W)
        def_ys = np.arange(H)
        def_zs = np.arange(D)
    else:
        # safely convert all three to int arrays
        def_xs = np.array(def_xs, int)
        def_ys = np.array(def_ys, int)
        def_zs = np.array(def_zs, int)

    # build 3D meshgrid in (Z,Y,X) order
    Zg, Yg, Xg = np.meshgrid(def_zs, def_ys, def_xs, indexing='ij')

    # compute total displacements
    TX = s1 * R1[Zg, Yg, Xg, 0] + s2 * R2[Zg, Yg, Xg, 0]
    TY = s1 * R1[Zg, Yg, Xg, 1] + s2 * R2[Zg, Yg, Xg, 1]
    TZ = s1 * R1[Zg, Yg, Xg, 2] + s2 * R2[Zg, Yg, Xg, 2]

    # warped floating-point coordinates
    def_X0 = Xg + TX
    def_Y0 = Yg + TY
    def_Z0 = Zg + TZ

    # out-of-bounds mask
    oob = (
        (def_X0 < 0) | (def_X0 >= W-1) |
        (def_Y0 < 0) | (def_Y0 >= H-1) |
        (def_Z0 < 0) | (def_Z0 >= D-1)
    )

    # flatten arrays
    def_Xf = def_X0.ravel()
    def_Yf = def_Y0.ravel()
    def_Zf = def_Z0.ravel()
    I_flat = I.ravel()
    oob_flat = oob.ravel()

    # only keep valid points
    valid = ~oob_flat
    Xf = def_Xf[valid]
    Yf = def_Yf[valid]
    Zf = def_Zf[valid]
    Ivals = I_flat[valid]

    # integer floors
    Xi = np.floor(Xf).astype(int)
    Yi = np.floor(Yf).astype(int)
    Zi = np.floor(Zf).astype(int)

    # fractional parts
    wx = Xf - Xi
    wy = Yf - Yi
    wz = Zf - Zi

    # base index in flattened volume
    base = Zi * (H*W) + Yi * W + Xi

    # compute the 8 corner indices
    idx000 = base
    idx100 = base + 1
    idx010 = base + W
    idx110 = base + W + 1
    idx001 = base + H*W
    idx101 = base + H*W + 1
    idx011 = base + H*W + W
    idx111 = base + H*W + W + 1

    # trilinear weights
    w000 = (1 - wx) * (1 - wy) * (1 - wz)
    w100 =  wx      * (1 - wy) * (1 - wz)
    w010 = (1 - wx) * wy      * (1 - wz)
    w110 =  wx      * wy      * (1 - wz)
    w001 = (1 - wx) * (1 - wy) * wz
    w101 =  wx      * (1 - wy) * wz
    w011 = (1 - wx) * wy      * wz
    w111 =  wx      * wy      * wz

    # allocate flattened outputs
    size = D * H * W
    I_def_flat   = np.zeros(size, dtype=I.dtype)
    weights_flat = np.zeros(size, dtype=float)

    # scatter-add contributions
    np.add.at(I_def_flat,   idx000, Ivals * w000)
    np.add.at(I_def_flat,   idx100, Ivals * w100)
    np.add.at(I_def_flat,   idx010, Ivals * w010)
    np.add.at(I_def_flat,   idx110, Ivals * w110)
    np.add.at(I_def_flat,   idx001, Ivals * w001)
    np.add.at(I_def_flat,   idx101, Ivals * w101)
    np.add.at(I_def_flat,   idx011, Ivals * w011)
    np.add.at(I_def_flat,   idx111, Ivals * w111)

    np.add.at(weights_flat, idx000, w000)
    np.add.at(weights_flat, idx100, w100)
    np.add.at(weights_flat, idx010, w010)
    np.add.at(weights_flat, idx110, w110)
    np.add.at(weights_flat, idx001, w001)
    np.add.at(weights_flat, idx101, w101)
    np.add.at(weights_flat, idx011, w011)
    np.add.at(weights_flat, idx111, w111)

    # reshape back to 3D
    I_def   = I_def_flat.reshape((D, H, W))
    weights = weights_flat.reshape((D, H, W))

    return I_def, weights
