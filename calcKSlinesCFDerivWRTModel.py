import numpy as np
from deformImWithModel import deformImWithModel 

def calcKSlinesCFDerivWRTModel(diff_KS, I_ref, R1, R2, S1, S2, KS_lines):
    """
    3D extension of cost‐function derivative w.r.t. motion model (R1, R2).
    diff_KS: ndarray, shape (D, N)   – k‐space difference for each TR
    I_ref:   ndarray, shape (D, H, W) – reference volume
    R1,R2:   ndarray, shape (D, H, W, 3) – motion fields for two surrogates
    S1,S2:   length‐N arrays of surrogate values
    KS_lines: length‐N list of x‐indices in k-space to compare
    Returns
    -------
      dC_by_dR1, dC_by_dR2 : each ndarray (D, H, W, 3)
    """
    D, H, W = I_ref.shape

    # 1) compute spatial gradients of the reference volume
    #    finite differences along each axis:
    I_ref_dx = np.zeros_like(I_ref)
    I_ref_dy = np.zeros_like(I_ref)
    I_ref_dz = np.zeros_like(I_ref)

    # derivative w.r.t x (axis=2)
    I_ref_dx[...,1:-1] = (I_ref[...,2:] - I_ref[...,:-2]) * 0.5
    I_ref_dx[...,0]    =  I_ref[...,1] - I_ref[...,0]
    I_ref_dx[...,-1]   =  I_ref[...,-1] - I_ref[...,-2]

    # derivative w.r.t y (axis=1)
    I_ref_dy[:,1:-1,:] = (I_ref[:,2:,:] - I_ref[:,:-2,:]) * 0.5
    I_ref_dy[:,0,:]    =  I_ref[:,1,:] - I_ref[:,0,:]
    I_ref_dy[:,-1,:]   =  I_ref[:,-1,:] - I_ref[:,-2,:]

    # derivative w.r.t z (axis=0)
    I_ref_dz[1:-1,:,:] = (I_ref[2:,:,:] - I_ref[:-2,:,:]) * 0.5
    I_ref_dz[0,:,:]    =  I_ref[1,:,:] - I_ref[0,:,:]
    I_ref_dz[-1,:,:]   =  I_ref[-1,:,:] - I_ref[-2,:,:]

    # 2) prepare output gradient arrays
    dC_by_dR1 = np.zeros((D, H, W, 3), dtype=np.float64)
    dC_by_dR2 = np.zeros((D, H, W, 3), dtype=np.float64)

    # 3) loop over each acquired k‐space line
    for n, ky in enumerate(KS_lines):
        # a) embed diff_KS[:,n] into a 3D k-space volume
        F_diff = np.zeros((D, H, W), dtype=np.complex128)
        # broadcast the D×1 vector across all y:
        F_diff[:, :, ky] = diff_KS[:, :, n]
        # F_diff[:, :, ky] = diff_KS[:,n][:,None]

        # b) undo shift & inverse 3D FFT
        F_un = np.fft.ifftshift(F_diff, axes=(0,1,2))
        I_diff = np.fft.ifftn(F_un, axes=(0,1,2))

        # c) warp the three gradient volumes forward
        #    assume deformImWithModel takes (volume, R1,R2,s1,s2) → warped volume
        I_def_dx, _, _, _ = deformImWithModel(I_ref_dx, R1, R2, S1[n], S2[n])
        I_def_dy, _, _, _ = deformImWithModel(I_ref_dy, R1, R2, S1[n], S2[n])
        I_def_dz, _, _, _ = deformImWithModel(I_ref_dz, R1, R2, S1[n], S2[n])

        # d) zero out any NaNs
        I_def_dx = np.nan_to_num(I_def_dx)
        I_def_dy = np.nan_to_num(I_def_dy)
        I_def_dz = np.nan_to_num(I_def_dz)

        # e) split real/imag parts
        Rd = I_diff.real
        Id = I_diff.imag

        Rdx = I_def_dx.real;  I_dx = I_def_dx.imag
        Rdy = I_def_dy.real;  I_dy = I_def_dy.imag
        Rdz = I_def_dz.real;  I_dz = I_def_dz.imag

        # f) accumulate into dC_by_dR1 and dC_by_dR2
        #    each component scaled by the surrogate values
        dC_by_dR1[..., 0] += (Rd*Rdx + Id*I_dx) * S1[n]
        dC_by_dR1[..., 1] += (Rd*Rdy + Id*I_dy) * S1[n]
        dC_by_dR1[..., 2] += (Rd*Rdz + Id*I_dz) * S1[n]

        dC_by_dR2[..., 0] += (Rd*Rdx + Id*I_dx) * S2[n]
        dC_by_dR2[..., 1] += (Rd*Rdy + Id*I_dy) * S2[n]
        dC_by_dR2[..., 2] += (Rd*Rdz + Id*I_dz) * S2[n]

    return dC_by_dR1, dC_by_dR2
