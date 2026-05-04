import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------
# PARAMETERS
# -------------------------
z_seg = 0.9
z_cut = 0.9

width = 0.16  # y extent (-w/2, w/2)

ax_x, bx_x = 0.15, 0.8   # x spread. The higher the more spread
ax_y, bx_y = 0.1, 0.4   # y spread
sigma_z = 1.0 #z spread

# -------------------------
# GRID
# -------------------------
x, y, z = np.mgrid[-2:2:80j, -2:2:80j, 0:3:60j]

# -------------------------
# PLANE PATCH BOUNDS
# -------------------------
x_min, x_max = -0.9, 0.9
y_min, y_max = -width/2, width/2

# -------------------------
# CLAMP POINTS TO RECTANGLE (in XY plane)
# -------------------------
x_clamped = np.clip(x, x_min, x_max)
y_clamped = np.clip(y, y_min, y_max)

# closest point on the rectangular plane patch
dx = x - x_clamped
dy = y - y_clamped

# vertical distance from plane
dz = z - z_seg

# distance to patch (3D)
dist = np.sqrt(dx**2 + dy**2 + dz**2)

# -------------------------
# HEIGHT-DEPENDENT SPREAD
# (only above the plane patch)
# -------------------------
z_rel = np.maximum(0, z - z_seg)
sigma_x = ax_x + bx_x * z_rel
sigma_y = ax_y + bx_y * z_rel

# -------------------------
# DISTRIBUTION
# -------------------------
dist_z = dz**2  # vertical component

density = np.exp(
    -(dx**2 / sigma_x**2 +
      dy**2 / sigma_y**2 +
      dist_z / (sigma_z**2))
)

# -------------------------
# HARD CUTOFF BELOW z_cut
# -------------------------
density *= (z >= z_cut)

# -------------------------
# 3D VISUALIZATION
# -------------------------
fig = go.Figure(data=go.Isosurface(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=density.flatten(),
    isomin=0.2,
    isomax=1.0,
    surface_count=3,
))

fig.update_layout(title="Probability over rectangular plane patch")
#fig.show()

# -------------------------
# CROSS-SECTIONS
# -------------------------
ix = x.shape[0] // 2
iy = y.shape[1] // 2
iz = z.shape[2] // 2

fig2, axs = plt.subplots(1, 4, figsize=(15, 4))

axs[0].imshow(density[:, :, int(iz*1.0)].T, origin='lower', extent=[-2,2,-2,2])
axs[0].set_title("XY slice")

axs[1].imshow(density[ix, :, :].T, origin='lower', extent=[-2,2,0,3])
axs[1].set_title("YZ slice")

axs[2].imshow(density[:, iy, :].T, origin='lower', extent=[-2,2,0,3])
axs[2].set_title("YZ slice")

axs[3].imshow(density[:, :, int(iz*1.6)].T, origin='lower', extent=[-2,2,-2,2])
axs[3].set_title("Higher Z slice")

plt.tight_layout()
plt.show()