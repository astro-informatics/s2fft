import pyvista as pv
import numpy as np
from s2fft.sampling import s2_samples as samples


def _cell_bounds(points: np.ndarray, bound_position: float = 0.5) -> np.ndarray:
    if points.ndim != 1:
        raise ValueError("Only 1D points are allowed.")
    diffs = np.diff(points)
    delta = diffs[0] * bound_position
    bounds = np.concatenate([[points[0] - delta], points + delta])
    return bounds


def plot_sphere(
    f: np.ndarray,
    L: int,
    sampling: str = "mw",
    cmap: str = "inferno",
    isnotebook: bool = False,
) -> None:
    """Plots signal (f) on the sphere using PyVista.

    Args:
        f (np.ndarray): Signal on the sphere.

        L (int): Harmonic band-limit.

        sampling (str, optional): Sampling scheme.  Supported sampling schemes include
            {"mw", "mwss", "dh"}. Defaults to "mw".

        cmap (str, optional): Colour map to plot data with. Defaults to "inferno".

        isnotebook (bool, optional): Whether to render as an embedded interative image
            in a Jupyter notebook environment. Defaults to False.

    Note:
        TODO: Add support for healpix sampling.
    """

    phis = samples.phis_equiang(L, sampling)
    thetas = samples.thetas(L, sampling)
    xx_bounds = _cell_bounds(np.degrees(phis))
    yy_bounds = _cell_bounds(np.degrees(thetas))
    grid_scalar = pv.grid_from_sph_coords(xx_bounds, yy_bounds, [1])
    grid_scalar.cell_data["example"] = np.array(f).swapaxes(-2, -1).ravel("C")

    # Make a plot
    pv.set_plot_theme("dark")
    p = pv.Plotter(notebook=isnotebook)
    p.add_mesh(grid_scalar, opacity=1.0, cmap=cmap)
    p.view_isometric()
    p.enable_anti_aliasing()
    p.remove_blurring()
    p.show(jupyter_backend="client") if isnotebook else p.show()
