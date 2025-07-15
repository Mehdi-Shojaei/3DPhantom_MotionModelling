import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from deformImWithModel import deformImWithModel

def simAcquireAllKSlines(I_ref, R1, R2, S1, S2, KS_lines, noise=0, anim=False, save=False):
    """
    Simulate acquiring k-space lines from a moving 2D object.
    I_ref: 2D reference image (numpy or cupy array).
    R1, R2: motion model parameter fields (H x W x 2 arrays).
    S1, S2: surrogate signal arrays (length = number of lines).
    KS_lines: list of column indices (0-based) for each acquired line.
    noise: percent noise level in k-space.
    anim: if True, collect frames and save an animated GIF.
    Returns:
      KS_acq: (H x N) complex array of acquired k-space lines
      frames: list of RGB uint8 arrays (each H x W x 3)
    """
    framess = []
    D, H, W = I_ref.shape[:3]
    KS_idx = [int(x) for x in np.array(KS_lines).flatten()]
    N = len(KS_idx)

    # ensure surrogates are long enough
    s1 = np.array(S1, float).flatten()
    s2 = np.array(S2, float).flatten()
    while len(s1) < N:
        s1 = np.concatenate((s1, s1))
        s2 = np.concatenate((s2, s2))
    s1 = s1[:N]; s2 = s2[:N]

    # prepare outputs
    KS_acq = np.zeros((D, H, N), dtype=np.complex128)
    F_acq  = np.zeros((D, H, W), dtype=np.complex128)
    frames = []

    if anim:
        plt.ioff()
        # plt.ion()
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        ax1, ax2, ax3, ax4 = axes.flatten()
        plt.tight_layout()
    
    for n in tqdm(range(N)):
        # 1) deform
        I_def, _, _ ,_= deformImWithModel(I_ref, R1, R2,
                                        float(s1[n]), float(s2[n]))
        # 2) FFT + center
        F_def = np.fft.fftn(I_def, axes=(0,1,2))
        F_def = np.fft.fftshift(F_def, axes=(0,1,2))
        
        # 3) add noise
        if noise:
            nr = np.random.randn(*F_def.shape)
            ni = np.random.randn(*F_def.shape)
            std_r = (noise/100.) * float(np.max(np.abs(np.real(F_def))))
            std_i = (noise/100.) * float(np.max(np.abs(np.imag(F_def))))
            F_def += std_r*nr + 1j*std_i*ni
        # 4) acquire the k-space plane at ky (i.e. fixed x-index through Z,Y)
        ky = KS_idx[n]
        
        # pick the slice F_def[:,:,ky] which is shape (D, H)
        KS_acq[:, :, n] = F_def[:, :, ky]
        # KS_acq[:, :, n] = F_def[:,n][:,None]
        # 5) accumulate for partial recon
        cnt = KS_idx.count(ky)
        F_acq[:, :, ky] += KS_acq[:, :, n] / cnt

        # 6) collect frame
        if anim:
            
            os.makedirs('frames', exist_ok=True)

            
            zmid = D // 2

            # panel 1: deformed image at zmid
            ax1.clear()
            ax1.imshow(np.abs(I_def[zmid]), cmap='gray')
            ax1.set_title(f"Deformed Image (z={zmid})")
            ax1.axis('off')

            # panel 2: k-space magnitude (log) at zmid
            ax2.clear()
            ax2.imshow(np.log(np.abs(F_def[zmid]) + 1e-6), cmap='gray')
            ax2.axvline(ky-0.5, color='r'); ax2.axvline(ky+0.5, color='r')
            ax2.set_title("K-space (log|F|)")
            ax2.axis('off')

            # panel 3: acquired lines so far at zmid
            ax3.clear()
            ax3.imshow(np.log(np.abs(KS_acq[zmid]) + 1e-6), aspect='auto', cmap='gray')
            ax3.set_title("Acquired Lines")
            ax3.axis('off')

            # panel 4: partial recon at zmid
            ax4.clear()
            Fp = np.fft.ifftshift(F_acq, axes=(0,1, 2))
            recon = np.fft.ifftn(Fp)
            ax4.imshow(np.abs(recon[zmid]), cmap='gray')
            ax4.set_title("Partial Recon")
            ax4.axis('off')

            
            framess.append(fig)
            
            if save:
                os.makedirs('frames', exist_ok=True)
                fig.savefig(f'frames/frame_{n:04d}.png', dpi=150)
            
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            buf = fig.canvas.buffer_rgba()
            h, w = buf.shape[:2]
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, buf.shape[1], 4)
            
            frame_rgb = arr[..., :3].copy()
            frames.append(frame_rgb)
            
            # frames.append(arr[..., :3])
            # plt.close(fig)
    return KS_acq, frames
