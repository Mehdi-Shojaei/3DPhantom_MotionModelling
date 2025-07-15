import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simAcquireAllKSlines import simAcquireAllKSlines
from MCRFromKSlinesUsingAdj import MCRFromKSlinesUsingAdj
from calcKSlinesCFDerivWRTModel import calcKSlinesCFDerivWRTModel
from save_anim import save_animation
from scipy.ndimage import gaussian_filter

def fitModelAndNonIterMCRToKSlines(KS_acq, S1, S2, KS_lines,
                                   anim=True,
                                   lims_R1_x=None, lims_R1_y=None, lims_R1_z = None,
                                   lims_R2_x=None, lims_R2_y=None, lims_R2_z = None,
                                   I_size=None, num_lev=3, sigma=5,
                                   max_iter=100,
                                   step_sizes_R=None, C_thresh=0.01):
    """
    Fit motion model to k-space lines & do non-iterative motion-compensated recon.
    Returns: I_rec, R1_fit, R2_fit, anim_frames
    """
    
    #--- Prepare surrogate arrays ---
    S1_arr = np.array(S1, dtype=float).flatten()
    S2_arr = np.array(S2, dtype=float).flatten()
    N = KS_acq.shape[2]
    while S1_arr.size < N:
        S1_arr = np.concatenate((S1_arr, S1_arr))
        S2_arr = np.concatenate((S2_arr, S2_arr))
    S1_arr = S1_arr[:N]; S2_arr = S2_arr[:N]

    #--- Dimensions ---
    D = KS_acq.shape[0]
    H = KS_acq.shape[1]
    W = I_size[2] if I_size is not None else H

    #--- Defaults ---
    if step_sizes_R is None:
        step_sizes_R = [2.0/x for x in (2,4,8,16,32,64,128,256)]

    #--- Initialization ---
    R1_fit = np.zeros((D, H, W, 3), dtype=float)
    R2_fit = np.zeros((D, H, W, 3), dtype=float)

    anim_frames = []
    if anim:
        plt.ioff()
        fig, axes = plt.subplots(2, 5, figsize=(10,6))
        gs = axes[0,0].get_gridspec()
        for ax in axes[:, :2].flatten():
            ax.remove()
        ax_img  = fig.add_subplot(gs[:, :2])
        ax_R1x  = fig.add_subplot(gs[0,2])
        ax_R1y  = fig.add_subplot(gs[0,3])
        ax_R1z  = fig.add_subplot(gs[0,4])
        ax_R2x  = fig.add_subplot(gs[1,2])
        ax_R2y  = fig.add_subplot(gs[1,3])
        ax_R2z  = fig.add_subplot(gs[1,4])
        plt.tight_layout()

    # Multi-resolution loop
    for lev in range(1, num_lev+1):
        print(f"\n=== Level {lev} ===============================")
        if lev == 1:
            I_rec, _ = MCRFromKSlinesUsingAdj(KS_acq, R1_fit, R2_fit,
                                              S1_arr, S2_arr, KS_lines,
                                              anim=True)
            sigma_level = sigma * 2**(num_lev-1)
        else:
            sigma_level = sigma / 2**(lev-2)

        # save_animation(I_rec_anim,f'I_rec_test2_level_{lev}/anim.mp4')
        # Simulate with current fit and compute initial loss
        KS_sim, anim1 = simAcquireAllKSlines(I_rec, R1_fit, R2_fit,
                                         S1_arr, S2_arr, KS_lines,
                                         noise=0, anim=True)
        
        save_animation(anim1,f'I_rec_test2_level_{lev}/anim.mp4')
        
        diff_KS = KS_acq - KS_sim
        C = float(np.sum(np.real(diff_KS)**2 + np.imag(diff_KS)**2))
        print(f"[Level {lev}] initial loss C = {C:.6e}")

        # Display initial (unchanged)
        if anim:
            mid_z = R1_fit.shape[0] // 2
            ax_img.clear(); ax_img.imshow(np.dstack([np.abs(I_rec[mid_z, :, :] / I_rec.max())]*3)); ax_img.axis('off')
            ax_R1x.clear(); ax_R1x.imshow(R1_fit[mid_z,:,:,0], vmin=lims_R1_x[0], vmax=lims_R1_x[1]); ax_R1x.axis('off')
            ax_R1y.clear(); ax_R1y.imshow(R1_fit[mid_z,:,:,1], vmin=lims_R1_y[0], vmax=lims_R1_y[1]); ax_R1y.axis('off')
            ax_R1z.clear(); ax_R1z.imshow(R1_fit[mid_z,:,:,2], vmin=lims_R1_z[0], vmax=lims_R1_z[1]); ax_R1z.axis('off')
            ax_R2x.clear(); ax_R2x.imshow(R2_fit[mid_z,:,:,0], vmin=lims_R2_x[0], vmax=lims_R2_x[1]); ax_R2x.axis('off')
            ax_R2y.clear(); ax_R2y.imshow(R2_fit[mid_z,:,:,1], vmin=lims_R2_y[0], vmax=lims_R2_y[1]); ax_R2y.axis('off')
            ax_R2z.clear(); ax_R2z.imshow(R2_fit[mid_z,:,:,2], vmin=lims_R2_z[0], vmax=lims_R2_z[1]); ax_R2z.axis('off')
            fig.savefig(f'I_rec_test2_level_{lev}/Rs_I.png')
            
            fig.suptitle(f"Level {lev} - Initial Fit")
            fig.canvas.draw()
            buf, (w, h) = fig.canvas.print_to_buffer()
            img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            img_rgb  = img_rgba[:, :, :3]
            anim_frames.append(img_rgb)
            
            plt.close(fig)

        iter_count = 0
        # Iterative refinement
        while True:
            print(f"\n[Level {lev}] Iteration {iter_count}")
            print(f"  -> current loss C = {C:.6e}")
            iter_count += 1
            if iter_count > max_iter:
                print("  !! Reached max iterations")
                break
            C_prev = C

            # Gradients
            dC_dR1, dC_dR2 = calcKSlinesCFDerivWRTModel(diff_KS, I_rec,
                                                        R1_fit, R2_fit,
                                                        S1_arr, S2_arr,
                                                        KS_lines)
            # Smooth
            if sigma_level > 0:
                for c in (0,1,2):
                    dC_dR1[:,:,:,c] = gaussian_filter(dC_dR1[:,:,:,c], sigma_level)
                    dC_dR2[:,:,:,c] = gaussian_filter(dC_dR2[:,:,:,c], sigma_level)
            # Normalize
            maxg = max(np.max(np.abs(dC_dR1)), np.max(np.abs(dC_dR2)))
            if maxg != 0:
                dC_dR1 /= maxg; dC_dR2 /= maxg

            improved = False
            # Line search
            for step in step_sizes_R:
                print(f"   Testing step size {step}")
                while True:
                    R1_new = R1_fit + step * dC_dR1
                    R2_new = R2_fit + step * dC_dR2
                    KS_sim_new, anim2 = simAcquireAllKSlines(I_rec, R1_new, R2_new,
                                                         S1_arr, S2_arr, KS_lines,
                                                         noise=0, anim=True)
                    
                    save_animation(anim2,f'I_rec_test2_level_{lev}/anim_{step}.mp4')
                    
                    diff_new = KS_acq - KS_sim_new
                    C_new = float(np.sum(np.real(diff_new)**2 + np.imag(diff_new)**2))
                    if C_new < C:
                        C = C_new
                        R1_fit = R1_new
                        R2_fit = R2_new
                        diff_KS = diff_new
                        improved = True
                        print(f"     -> improved! new loss = {C:.6e}")
                    else:
                        break

            # Check convergence
            if not improved:
                print("\033[1;31mNo improvement in this iteration\033[0m")
            elif (C_prev - C) < (C_prev * C_thresh):
                print(f"   Improvement below threshold: ΔC = {C_prev-C:.6e}")
                print("   Converged for this level")
                break

            # Update image reconstruction
            print("  Reconstructing image with updated motion...")
            I_rec, _ = MCRFromKSlinesUsingAdj(KS_acq, R1_fit, R2_fit,
                                              S1_arr, S2_arr, KS_lines,
                                              anim=False)
            KS_sim, _ = simAcquireAllKSlines(I_rec, R1_fit, R2_fit,
                                             S1_arr, S2_arr, KS_lines,
                                             noise=0, anim=False)
            diff_KS = KS_acq - KS_sim
            C = float(np.sum(np.real(diff_KS)**2 + np.imag(diff_KS)**2))
            print(f"  -> post-recon loss = {C:.6e}")

        print(f"[Level {lev}] Final loss = {C:.6e}")

    return (I_rec, R1_fit, R2_fit, anim_frames)
