import numpy as np
from scipy import ndimage

def deformImWithModel(I_ref, R1, R2, s1, s2, def_xs=None, def_ys=None, def_zs=None):
    """
    Deform image I_ref using motion model (R1, R2) and surrogate signals (s1, s2).
    def_xs, def_ys, def_zs: optional sequences of x, y, and z indices specifying a sub-region to deform (if provided, must specify all).
    Returns: (I_def, T_X, T_Y, T_Z) - the deformed image and displacement fields in X, Y, and Z.
    """
    
    D, H, W = I_ref.shape[:3]
    if (def_xs is None) != (def_ys is None):
        raise ValueError("Must specify both def_xs and def_ys or neither.")
    if def_xs is None:
        def_xs = np.arange(W)
        def_ys = np.arange(H)
        def_zs = np.arange(D)
    else:
        def_xs = np.array(def_xs)
        def_ys = np.array(def_ys)
        def_zs = np.array(def_zs)
        
        
    # Coordinate grid for region
    Zg, Yg, Xg = np.meshgrid(def_zs, def_ys, def_xs, indexing='ij') 
    
    # sample the motion fields
    T_X = (s1 * R1[Zg, Yg, Xg, 2] +
           s2 * R2[Zg, Yg, Xg, 2])
    T_Y = (s1 * R1[Zg, Yg, Xg, 1] +
           s2 * R2[Zg, Yg, Xg, 1])
    T_Z = (s1 * R1[Zg, Yg, Xg, 0] +
           s2 * R2[Zg, Yg, Xg, 0])
    
    
    # 3) compute deformed coordinates
    def_X0 = Xg.astype(float) + T_X
    def_Y0 = Yg.astype(float) + T_Y
    def_Z0 = Zg.astype(float) + T_Z
    
    
    # map_coordinates expects a shape (3, Npoints) array in the same
    #    order as the volume axes: (z_indices, y_indices, x_indices)
    coords = np.vstack([
        def_Z0.ravel(),
        def_Y0.ravel(),
        def_X0.ravel()
    ])
    
    
# sample the (complex) phantom volume
    real_part = ndimage.map_coordinates(
        np.real(I_ref), coords, order=1, mode='constant', cval=0.0)
    imag_part = ndimage.map_coordinates(
        np.imag(I_ref), coords, order=1, mode='constant', cval=0.0)
    out_flat = real_part + 1j * imag_part

    # 6) reshape back to (D, H, W)
    I_def = out_flat.reshape((D, H, W))

    return I_def, T_Z, T_Y, T_X
