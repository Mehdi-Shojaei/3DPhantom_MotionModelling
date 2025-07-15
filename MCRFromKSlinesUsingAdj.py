import numpy as np
import matplotlib.pyplot as plt
from deformImWithModelAdjoint import deformImWithModelAdjoint

def MCRFromKSlinesUsingAdj(KS_acq, R1, R2, S1, S2, KS_lines, anim=False):
    """
    Motion-Compensated Reconstruction (adjoint method) from k-space lines.
    KS_acq: acquired k-space (H x N), R1,R2: motion model (H x W x 2), S1,S2: surrogate arrays (length N).
    KS_lines: list of column indices for each line.
    anim: if True, return frames of reconstruction process.
    Returns: (I_rec, anim_frames) - reconstructed image and list of frames (if anim=True).
    """

    #--- Dimensions ---
    D = KS_acq.shape[0]
    H = KS_acq.shape[1]
    W = R1.shape[2] if R1 is not None else H

    num_lines = KS_acq.shape[2]
    I_rec = np.zeros((D, H, W), dtype=np.complex128)
    
    
    anim_frames = []
    if anim:
        fig, axes = plt.subplots(3, 4, figsize=(10,8))
        plt.ion()
        plt.show(block=False)
        
        
    for n in range(num_lines):
        F_acq = np.zeros((D, H, W), dtype=np.complex128)
        col_idx = KS_lines[n]
        
        F_acq[:, :, col_idx] = KS_acq[:, :, n]
        # F_acq[:, :, col_idx] = KS_acq[:,n][:,None]
        count = KS_lines.count(col_idx)
        if count > 1:
            F_acq /= count
            
        F_acq_ifft = np.fft.ifftshift(F_acq)
        I_acq = np.fft.ifftn(F_acq_ifft)
        I_def, weights = deformImWithModelAdjoint(I_acq, R1, R2, float(S1[n]), float(S2[n]))
        I_def = I_def / weights
        mask = np.logical_or(np.isnan(I_def), np.isinf(I_def))
        I_def[mask] = 0
        I_rec += I_def
        if anim:
            z_mid = D // 2
            axes = axes.flatten()
            axes[0].clear(); axes[0].imshow(np.real(F_acq[z_mid]), cmap='gray'); axes[0].set_title("Re(F_acq)"); axes[0].axis('off')
            axes[4].clear(); axes[4].imshow(np.imag(F_acq[z_mid]), cmap='gray'); axes[4].set_title("Im(F_acq)"); axes[4].axis('off')
            axes[8].clear(); axes[8].imshow(np.log(np.abs(F_acq[z_mid]) + 1e-6), cmap='gray'); axes[8].set_title("log|F_acq|"); axes[8].axis('off')
            axes[1].clear(); axes[1].imshow(np.real(I_acq[z_mid]), cmap='gray'); axes[1].set_title("Re(I_acq)"); axes[1].axis('off')
            axes[5].clear(); axes[5].imshow(np.imag(I_acq[z_mid]), cmap='gray'); axes[5].set_title("Im(I_acq)"); axes[5].axis('off')
            axes[9].clear(); axes[9].imshow(np.abs(I_acq[z_mid]), cmap='gray'); axes[9].set_title("|I_acq|"); axes[9].axis('off')
            axes[2].clear(); axes[2].imshow(np.real(I_def[z_mid]), cmap='gray'); axes[2].set_title("Re(I_def)"); axes[2].axis('off')
            axes[6].clear(); axes[6].imshow(np.imag(I_def[z_mid]), cmap='gray'); axes[6].set_title("Im(I_def)"); axes[6].axis('off')
            axes[10].clear(); axes[10].imshow(np.abs(I_def[z_mid]), cmap='gray'); axes[10].set_title("|I_def|"); axes[10].axis('off')
            axes[3].clear(); axes[3].imshow(np.real(I_rec[z_mid]), cmap='gray'); axes[3].set_title("Re(I_rec)"); axes[3].axis('off')
            axes[7].clear(); axes[7].imshow(np.imag(I_rec[z_mid]), cmap='gray'); axes[7].set_title("Im(I_rec)"); axes[7].axis('off')
            axes[11].clear(); axes[11].imshow(np.abs(I_rec[z_mid]), cmap='gray'); axes[11].set_title("|I_rec|"); axes[11].axis('off')
            plt.tight_layout()

            # draw the updated canvas
            fig.canvas.draw()
            fig.canvas.flush_events()

            # grab the RGBA buffer, convert to H×W×4 uint8
            buf = fig.canvas.buffer_rgba()
            h, w = buf.shape[:2]
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, buf.shape[1], 4)
         
            
            # drop the alpha channel → H×W×3 RGB
            frame_img = arr[..., :3].copy()
            anim_frames.append(frame_img)
            plt.close(fig)
        
    return I_rec, anim_frames
