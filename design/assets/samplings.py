# Node layouts of the samplings discussed in the design doc, drawn on an orthographic globe.
# Run: uv run --with numpy --with matplotlib python samplings.py
# Writes one PNG per sampling family next to this script.
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#2E2E2E"
KW = "#C2560A"
KW2 = "#521463"
BLUE = "#3B6FB6"

VIEW_LAT, VIEW_LON = np.radians(25.0), np.radians(-20.0)


def ortho(theta, phi, lat0=VIEW_LAT, lon0=VIEW_LON):
    """Orthographic projection of colatitude/longitude; returns x, y and a front-side mask."""
    lat = np.pi / 2 - theta
    x = np.cos(lat) * np.sin(phi - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(phi - lon0)
    front = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(phi - lon0) > 0
    return x, y, front


def rings(thetas, nphis, offsets=None):
    """Flatten ring grids: thetas[t] with nphis[t] equally spaced longitudes."""
    th, ph = [], []
    for t, (theta, n) in enumerate(zip(thetas, nphis)):
        off = 0.0 if offsets is None else offsets[t]
        th.append(np.full(n, theta))
        ph.append(2 * np.pi * (np.arange(n) + off) / n)
    return np.concatenate(th), np.concatenate(ph)


def gauss_thetas(nlat):
    mu, _ = np.polynomial.legendre.leggauss(nlat)
    return np.arccos(mu)[::-1]


def healpix(nside):
    """Ring latitudes, ring sizes and half-pixel offsets of HEALPix."""
    th, n, off = [], [], []
    for i in range(1, 4 * nside):
        if i < nside:
            z = 1 - i**2 / (3 * nside**2)
            npix, o = 4 * i, 0.5
        elif i <= 3 * nside:
            z = 4 / 3 - 2 * i / (3 * nside)
            npix, o = 4 * nside, 0.5 if (i - nside) % 2 == 0 else 0.0
        else:
            j = 4 * nside - i
            z = -(1 - j**2 / (3 * nside**2))
            npix, o = 4 * j, 0.5
        th.append(np.arccos(z))
        n.append(npix)
        off.append(o)
    return th, n, off


def schmidt(theta, phi, c, lat0, lon0):
    """Schmidt stretch towards (lat0, lon0): tan(theta_E/2) = tan(theta_c/2)/c, then rotate pole to target."""
    th = 2 * np.arctan(np.tan(theta / 2) / c)
    v = np.stack([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    t = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    e = np.cross([0.0, 0.0, 1.0], t)
    e /= np.linalg.norm(e)
    n = np.cross(t, e)
    u = np.stack([e, n, t], axis=1) @ v
    return np.arccos(np.clip(u[2], -1, 1)), np.arctan2(u[1], u[0])


def draw(ax, theta, phi, color, title, size=4):
    circle = plt.Circle((0, 0), 1, facecolor="#faf7f0", edgecolor=INK, lw=1.0)
    ax.add_patch(circle)
    x, y, front = ortho(theta, phi)
    ax.scatter(x[front], y[front], s=size, color=color, lw=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=INK, fontsize=11)


def save(panels, name):
    """Write {name}.png (panels side by side, for DESIGN.md) and {name}_v.png (stacked, for the slides)."""
    n = len(panels)
    for suffix, shape, size in (("", (1, n), (3.2 * n, 3.4)), ("_v", (n, 1), (3.4, 3.6 * n))):
        fig, axes = plt.subplots(*shape, figsize=size)
        axes = np.atleast_1d(axes)
        for ax, (theta, phi, color, title) in zip(axes, panels):
            draw(ax, theta, phi, color, title)
        fig.tight_layout()
        fig.savefig(f"{name}{suffix}.png", dpi=200, transparent=True)
        plt.close(fig)


L = 24

# Equiangular: MW rings at (2t+1)pi/(2L-1), 2L-1 points; MWSS rings at t pi/L, 2L points.
mw = rings((2 * np.arange(L) + 1) * np.pi / (2 * L - 1), [2 * L - 1] * L)
mwss = rings(np.arange(L + 1) * np.pi / L, [2 * L] * (L + 1))
save([(*mw, BLUE, "MW"), (*mwss, BLUE, "MWSS")], "equiangular")

# Gaussian family.
N = 12
gl = rings(gauss_thetas(L), [2 * L - 1] * L)
fn = rings(gauss_thetas(2 * N), [4 * N] * (2 * N))
# Octahedral O_N: ring i from the pole has 4i + 16 points.
o_n = [4 * i + 16 for i in range(1, N + 1)]
on = rings(gauss_thetas(2 * N), o_n + o_n[::-1])
# Reduced N_N: illustrative only, ring sizes follow 4N sin(theta) (the ECMWF table is not reproduced).
th_n = gauss_thetas(2 * N)
nn = rings(th_n, [max(8, int(np.ceil(4 * N * np.sin(t) / 4)) * 4) for t in th_n])
save([(*gl, KW, "GL (2L-1)"), (*fn, KW, "Regular F$_N$ (4N)")], "gaussian_regular")
save([(*on, KW, "Octahedral O$_N$"), (*nn, KW, "Reduced N$_N$ (illustrative)")], "gaussian_reduced")

# HEALPix.
hp = rings(*healpix(8))
save([(*hp, KW2, "HEALPix, nside 8")], "healpix")

# Stretched GL towards London with c = 2.4.
st = schmidt(*gl, 2.4, np.radians(51.5), np.radians(-0.13))
save([(*gl, KW, "GL, c = 1"), (*st, KW, "GL, c = 2.4 (London)")], "stretched")
