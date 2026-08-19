import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
N = 1000

# angle limits (deg)
lim = 75

# spherical distribution parameters
mu = 45       # radius of high probability shell
sigma = 20   # thickness of shell

# =========================
# REJECTION SAMPLING
# =========================
samples = []

while len(samples) < N:
    # uniform proposal in cube
    x = np.random.uniform(-lim, lim)
    y = np.random.uniform(-lim, lim)
    z = np.random.uniform(-lim, lim)

    r = np.sqrt(x**2 + y**2 + z**2)

    # shell Gaussian probability
    p = np.exp(-((r - mu)**2) / (2 * sigma**2))

    # accept/reject
    if np.random.rand() < p:
        samples.append([x, y, z])

samples = np.array(samples)

roll, pitch, yaw = samples[:,0], samples[:,1], samples[:,2]

# =========================
# 3D VISUALIZATION
# =========================
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

ax.scatter(roll, pitch, yaw, s=8, alpha=0.4)

ax.set_xlabel("Roll")
ax.set_ylabel("Pitch")
ax.set_zlabel("Yaw")
ax.set_title("Hollow spherical distribution in RPY space")

plt.show()

# =========================
# 2D PROJECTIONS
# =========================
fig2, axs = plt.subplots(1, 3, figsize=(15,4))

axs[0].scatter(roll, pitch, s=8, alpha=0.4)
axs[0].set_title("Roll vs Pitch")

axs[1].scatter(pitch, yaw, s=8, alpha=0.4)
axs[1].set_title("Pitch vs Yaw")

axs[2].scatter(roll, yaw, s=8, alpha=0.4)
axs[2].set_title("Roll vs Yaw")

plt.tight_layout()
plt.show()