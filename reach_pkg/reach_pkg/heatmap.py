import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf

option=2 #0:Linear, 1: RBF, 2: Nadaraya Watson
# ============================================================
# LOAD FILE
# ============================================================

filename = "/home/rosdev/ros2_ws/src/reach_pkg/data/history_width_1_8.txt"

with open(filename, "r") as f:
    text = f.read()

# ============================================================
# PARSE DATA
# ============================================================

pattern = (
    r"y=([-\d.eE]+)\s+"
    r"z=([-\d.eE]+)\s+"
    r"roll=([-\d.eE]+)\s+"
    r"score=([-\d.eE]+)"
)

pattern = (
    r"y=([-\d.eE]+)\s+"
    r"z=([-\d.eE]+)\s+"
    r"roll=([-\d.eE]+)\s+"
    r"score=([-\d.eE]+)\s+" \
    r"score_prune=([-\d.eE]+)\s+" \
    r"score_scan=([-\d.eE]+)\s+" \
    r"score_grasp=([-\d.eE]+)"
)

matches = re.findall(pattern, text)

data = np.array(matches, dtype=float)

y = data[:, 0]
z = data[:, 1]
roll = data[:, 2]
#score = data[:, 3]
score = data[:, 3]

print(f"Loaded {len(score)} samples")

# ============================================================
# NADARAYA-WATSON REGRESSION
# ============================================================

def nw_kernel_regression(z_train,
                         roll_train,
                         score_train,
                         ZI,
                         RI,
                         bandwidth=0.12):
    """
    Gaussian-kernel Nadaraya-Watson estimator.
    """

    pred = np.zeros_like(ZI)

    for i in range(ZI.shape[0]):
        for j in range(ZI.shape[1]):

            dz = z_train - ZI[i, j]
            dr = roll_train - RI[i, j]

            d2 = dz**2 + dr**2

            w = np.exp(-0.5 * d2 / bandwidth**2)

            pred[i, j] = np.sum(w * score_train) / (np.sum(w) + 1e-12)

    return pred


# ============================================================
# 3D SCATTER HEATMAP
# ============================================================

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# sc = ax.scatter(
#     y,
#     z,
#     roll,
#     c=score,
#     cmap="viridis",
#     s=60,
#     vmin=score[int(len(score)*0.7)],#0.15,
#     vmax=score[0]
# )

# ax.set_xlabel("y")
# ax.set_ylabel("z")
# ax.set_zlabel("roll")
# ax.set_title("3D Score Heatmap")

# plt.colorbar(sc, label="score")

# plt.show()

# ============================================================
# Z-ROLL PROJECTIONS AT DIFFERENT Y
# ============================================================

#levels = [0.2, 0.5, 0.8]
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
y_levels = np.quantile(y, levels)

fig, axs = plt.subplots(1, len(levels), figsize=(18, 5))

for ax, y_target in zip(axs, y_levels):

    tol = 0.05 * (y.max() - y.min())

    mask = np.abs(y - y_target) < tol

    if np.sum(mask) < 4:
        ax.set_title(f"y≈{y_target:.3f}\nNot enough points")
        continue

    z_slice = z[mask]
    roll_slice = roll[mask]
    score_slice = score[mask]

    zi = np.linspace(z.min(), z.max(), 100)
    ri = np.linspace(roll.min(), roll.max(), 100)

    ZI, RI = np.meshgrid(zi, ri)

    if option==0:
        ## LINEAR ###

        SI = griddata(
            (z_slice, roll_slice),
            score_slice,
            (ZI, RI),
            method="linear"
        )

        im = ax.imshow(
            SI,
            origin="lower",
            extent=[
                zi.min(), zi.max(),
                ri.min(), ri.max()
            ],
            aspect="auto",
            vmin=score[int(len(score)*0.7)],#0.15,
            #vmax=score[0]
        )

    elif option==1:
        ## RBF ###
        rbf = Rbf(
            z_slice,
            roll_slice,
            score_slice,
            function='multiquadric'
        )

        SI = rbf(ZI, RI)

        im = ax.imshow(
            SI,
            origin='lower',
            extent=[zi.min(), zi.max(),
                    ri.min(), ri.max()],
            aspect='auto',
            cmap='viridis',
            vmin=score[int(len(score)*0.7)],#0.15,
            vmax=score[0]
        )

    elif option==2:
        ### Nadaraya-Watson ###
        # Kernel smoothing
        SI = nw_kernel_regression(
            z_slice,
            roll_slice,
            score_slice,
            ZI,
            RI,
            bandwidth=0.15
        )

        im = ax.imshow(
            SI,
            origin="lower",
            extent=[
                zi.min(),
                zi.max(),
                ri.min(),
                ri.max()
            ],
            aspect="auto",
            cmap='viridis',
            vmin=score[int(len(score)*0.7)],#0.15,
            vmax=score[0]*0.9
        )

    ax.scatter(z_slice, roll_slice, s=10)

    ax.set_xlabel("z")
    ax.set_ylabel("roll")
    ax.set_title(f"y ≈ {y_target:.3f}")

fig.colorbar(im, ax=axs, label="score")

plt.tight_layout()
plt.show()