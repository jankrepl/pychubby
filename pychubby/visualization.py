"""Collection of tools for visualization."""

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


def create_animation(df, img, include_backwards=True, fps=24, n_seconds=2, figsize=(8, 8), repeat=True):
    """Create animation from a displacement field.

    Parameters
    ----------
    df : DisplacementField
        Instance of the ``DisplacementField`` representing the coordinate transformation.

    img : np.ndarray
        Image.

    include_backwards : bool
        If True, then animation also played backwards after its played forwards.

    fps : int
        Frames per second.

    n_seconds : int
        Number of seconds to play the animation forwards.

    figsize : tuple
        Size of the figure.

    repeat : bool
        If True, then animation always replayed at the end.

    Returns
    -------
    ani : matplotlib.animation.ArtistAnimation
        Animation showing the transformation.

    Notes
    -----
    To enable animation viewing in a jupyter notebook write:
    ```
    from matplotlib import rc
    rc('animation', html='html5')
    ```

    """
    n_frames = int(n_seconds * fps)
    interval = (1 / fps) * 1000
    frames = []

    fig = plt.figure(figsize=figsize)
    plt.axis('off')

    for i in range(n_frames + 1):
        df_new = df * (i / n_frames)
        warped_img = df_new.warp(img)
        frames.append([plt.imshow(warped_img, cmap='gray')])

    if include_backwards:
        frames += frames[::-1]

    ani = ArtistAnimation(fig, frames, interval=interval, repeat=repeat)

    return ani
