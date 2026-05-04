import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# PARAMETERS
# =========================

# Segment endpoints
A = np.array([-0.85, 0.0, 1.0])
B = np.array([0.85, 0.0, 1.0])
z_seg = A[2]

# Spread parameters (independent axes)
ax_x, bx_x = 0.2, 0.0
ax_y, bx_y = 0.2, 0.0

# Z asymmetry
az_pos, bz_pos = 0.3, 0.0   # above segment
az_neg, bz_neg = 0.15, 0.0   # below segment

# =========================
# HARD BOUNDARIES
# =========================
x_min, x_max = -1.1, 1.1
y_min, y_max = -0.25, 0.25
z_min, z_max = 0.7, 1.4

# =========================
# PLOT BOUNDARIES
# =========================
x_plot_min, x_plot_max = -1.5, 1.5
y_plot_min, y_plot_max = -1.0, 1.0
z_plot_min, z_plot_max = 0.0, 2.0

# =========================
# GRID
# =========================
x, y, z = np.mgrid[x_plot_min:x_plot_max:80j, y_plot_min:y_plot_max:80j, z_plot_min:z_plot_max:80j]

# =========================
# DISTANCE TO SEGMENT
# =========================
AB = B - A
AB2 = np.dot(AB, AB)

P = np.stack([x - A[0], y - A[1], z - A[2]], axis=-1)

t = (P[...,0]*AB[0] + P[...,1]*AB[1] + P[...,2]*AB[2]) / AB2
t = np.clip(t, 0, 1)

closest = np.stack([
    A[0] + t * AB[0],
    A[1] + t * AB[1],
    A[2] + t * AB[2]
], axis=-1)

diff = np.stack([x, y, z], axis=-1) - closest

dx = diff[...,0]
dy = diff[...,1]
dz = diff[...,2]

# =========================
# HEIGHT-DEPENDENT SPREAD
# =========================
z_rel = z - z_seg

sigma_x = ax_x * (1 + bx_x * np.abs(z_rel))
sigma_y = ax_y * (1 + bx_y * np.abs(z_rel))

# asymmetric z spread
sigma_z = np.where(
    z_rel >= 0,
    az_pos * (1 + bz_pos * z_rel),
    az_neg * (1 + bz_neg * np.abs(z_rel))
)

# =========================
# PROBABILITY
# =========================
density = np.exp(
    -(dx**2 / sigma_x**2 +
      dy**2 / sigma_y**2 +
      dz**2 / sigma_z**2)
)

# axis-aligned box
mask_x = (x >= x_min) & (x <= x_max)
mask_y = (y >= y_min) & (y <= y_max)
mask_z = (z >= z_min) & (z <= z_max)

# combine
mask_box = mask_x & mask_y & mask_z

density *= mask_box

# =========================
# VISUALIZATION
# =========================
fig = go.Figure(data=go.Isosurface(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=density.flatten(),
    isomin=0.2,
    isomax=1.0,
    surface_count=3,
))

fig.update_layout(title="Anisotropic segment-based distribution with asymmetric Z spread")
fig.show()

# =========================
# CROSS SECTIONS
# =========================
ix = x.shape[0] // 2
iy = x.shape[1] // 2
iz = z.shape[2] // 2

fig2, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].imshow(density[:, :, iz].T, origin='lower', extent=[x_plot_min,x_plot_max,y_plot_min,y_plot_max])
axs[0].set_title("XY slice")

axs[1].imshow(density[:, :, iz+10].T, origin='lower', extent=[x_plot_min,x_plot_max,y_plot_min,y_plot_max])
axs[1].set_title("Higher Z slice")

axs[2].imshow(density[ix, :, :].T, origin='lower', extent=[y_plot_min,y_plot_max,z_plot_min,z_plot_max])
axs[2].set_title("YZ slice")

axs[3].imshow(density[:, iy, :].T, origin='lower', extent=[x_plot_min,x_plot_max,z_plot_min,z_plot_max])
axs[3].set_title("ZX slice")

plt.tight_layout()
plt.show()