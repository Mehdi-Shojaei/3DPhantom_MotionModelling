import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Set path to ffmpeg
mpl.rcParams["animation.ffmpeg_path"] = r"D:\Software\ffmpeg\ffmpeg\bin\ffmpeg.exe"

def save_animation(anim_mov_acq, output_path='anim.mp4', fps=10, dpi=300):
    """
    Save an animation from a 3D numpy array as an MP4 using matplotlib and ffmpeg.
    
    Parameters:
        anim_mov_acq (np.ndarray): 3D array (frames, height, width) representing the animation.
        output_path (str): Path to save the animation MP4.
        fps (int): Frames per second.
        dpi (int): Dots per inch for video quality.
        save (bool): Whether to save the animation or not.
    """

    fig, ax = plt.subplots()
    im = ax.imshow(anim_mov_acq[0], cmap='gray')
    ax.axis('off')
    plt.tight_layout()

    def update(frame_index):
        im.set_data(anim_mov_acq[frame_index])
        return (im,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(anim_mov_acq),
        blit=True,
        interval=1000/fps
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved MP4 to {output_path}")
