import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

np.random.seed(0)


N = 20000
samples = np.random.randn(N, 2)


d = 2
r_typ = np.sqrt(d - 1.0)  # = 1.0
ring_halfwidth = 0.3
r_inner = r_typ - ring_halfwidth
r_outer = r_typ + ring_halfwidth


fig, ax = plt.subplots(figsize=(4, 4), dpi=200)


ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.15, color="#4C78A8", linewidths=0)


# grid = np.linspace(-3.5, 3.5, 200)
# X, Y = np.meshgrid(grid, grid)
# Z = np.exp(-0.5 * (X + Y)) / (2 * np.pi)

grid = np.linspace(-3.5, 3.5, 200)
X, Y = np.meshgrid(grid, grid)
R2 = np.square(X) + np.square(Y)
Z = np.exp(-0.5 * R2) / (2 * np.pi)

ax.contour(X, Y, Z, levels=6, colors="k", alpha=0.3)


ring = Wedge(
    center=(0, 0),
    r=r_outer,
    theta1=0,
    theta2=360,
    width=r_outer - r_inner,
    facecolor="#F58518",
    edgecolor="#F58518",
    alpha=0.25,
)
ax.add_patch(ring)


ax.plot(0, 0, marker="*", color="k", ms=10)


ax.annotate(
    "Mode (highest density)",
    xy=(0, 0),
    xytext=(-2.8, 2.8),
    arrowprops=dict(arrowstyle="->", lw=1.2),
)
ax.annotate(
    "Typical set (most mass)",
    xy=(r_typ, 0),
    xytext=(1.6, 2.4),
    arrowprops=dict(arrowstyle="->", lw=1.2),
)


ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel("q1")
ax.set_ylabel("q2")
ax.set_title("2D Gaussian: mode vs typical set")
plt.tight_layout()


plt.savefig("mode_vs_typical_set.pdf", dpi=600, bbox_inches="tight")
# plt.savefig('mode_vs_typical_set.svg', bbox_inches='tight')
# plt.show()
