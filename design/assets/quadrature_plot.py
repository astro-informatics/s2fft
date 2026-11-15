# Ring weights against colatitude for GL, DH, CC and HEALPix, from quadrature_weights.npz.
# Run: uv run --no-project --with numpy --with matplotlib python quadrature_plot.py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("quadrature_weights.npz")
style = {
    "gl": ("#C2560A", "o", "GL, L = 32"),
    "dh": ("#3B6FB6", "s", "DH, L = 32"),
    "cc": ("#2E2E2E", "^", "CC, L = 32"),
    "healpix": ("#521463", "D", "HEALPix, nside = 16"),
}
fig, ax = plt.subplots(figsize=(5.2, 5.6))
for s, (c, m, lab) in style.items():
    th, w = d[f"{s}_theta"], d[f"{s}_ring"]
    ax.plot(np.degrees(th), w / w.mean(), m, ms=3.5, color=c, label=lab, lw=0)
ax.set_xlabel(r"colatitude $\theta$ (deg)")
ax.set_ylabel("ring weight / mean ring weight")
ax.set_xlim(0, 180)
ax.legend(frameon=False, fontsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("quadrature_weights.png", dpi=200, transparent=True)
