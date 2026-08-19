import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import Rbf


# ============================================================
# OPTIONS
# ============================================================

option = 2  # 0: Linear, 1: RBF, 2: Nadaraya-Watson


# ============================================================
# LOAD FILE
# ============================================================

#filename = "/home/rosdev/ros2_ws/src/reach_pkg/data/history_ur5_length_0_8.txt"
filename = "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/scores/score_ur5e_straight_1.txt" #Upside down

with open(filename, "r") as f:
    text = f.read()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

# Get filename without extension
file_name = os.path.splitext(os.path.basename(filename))[0]

# Example:
# /.../data/history_ur5_length_1_6_Lshape/
#                              └── score.jpg
output_dir = os.path.join(
    os.path.dirname(filename),
    file_name
)

os.makedirs(output_dir, exist_ok=True)

print(f"Saving figures to: {output_dir}")


# ============================================================
# PARSE DATA
# ============================================================

# Find all key=value pairs.
#
# This will handle an arbitrary number of columns:
#
# y=-0.97
# z=0.27
# roll=0.58
# score=0
# score_prune=0
# ...
#
# It does not care how many fields exist.

pair_pattern = r"(\w+)=([-\d.eE]+)"

matches = re.findall(pair_pattern, text)


# ============================================================
# PARSE LINE BY LINE
# ============================================================
#
# We need to keep each line as one sample.
# Otherwise re.findall() would flatten everything into one array.

data = []

with open(filename, "r") as f:

    for line in f:

        pairs = dict(re.findall(pair_pattern, line))

        # Ignore lines that don't contain the required coordinates
        if not all(k in pairs for k in ["y", "z", "roll"]):
            continue

        # Convert values to float
        sample = {
            key: float(value)
            for key, value in pairs.items()
        }

        data.append(sample)


if len(data) == 0:
    raise RuntimeError("No valid samples found in the file.")


print(f"Loaded {len(data)} samples")


# ============================================================
# GET AVAILABLE COLUMNS
# ============================================================

# Get all keys that occur in the file
all_columns = set()

for sample in data:
    all_columns.update(sample.keys())


# We want these three to be the independent variables
required_axes = ["y", "z", "roll"]

# Everything else will be plotted
score_columns = [
    column
    for column in all_columns
    if column not in required_axes and column != "Iteration"
]

# Preserve the order in which the columns first appeared
ordered_columns = []

with open(filename, "r") as f:

    for line in f:

        pairs = re.findall(pair_pattern, line)

        for key, value in pairs:

            if key not in ordered_columns:
                ordered_columns.append(key)


score_columns = [
    column
    for column in ordered_columns
    if column not in required_axes and column != "Iteration"
]


print("\nColumns found:")
for column in score_columns:
    print(f"  {column}")


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

y = np.array([sample["y"] for sample in data])
z = np.array([sample["z"] for sample in data])
roll = np.array([sample["roll"] for sample in data])


# ============================================================
# NADARAYA-WATSON REGRESSION
# ============================================================

def nw_kernel_regression(
    z_train,
    roll_train,
    score_train,
    ZI,
    RI,
    bandwidth=0.12
):
    """
    Gaussian-kernel Nadaraya-Watson estimator.
    """

    pred = np.zeros_like(ZI)

    for i in range(ZI.shape[0]):

        for j in range(ZI.shape[1]):

            dz = z_train - ZI[i, j]
            dr = roll_train - RI[i, j]

            d2 = dz**2 + dr**2

            w = np.exp(
                -0.5 * d2 / bandwidth**2
            )

            pred[i, j] = (
                np.sum(w * score_train)
                / (np.sum(w) + 1e-12)
            )

    return pred


# ============================================================
# Y LEVELS
# ============================================================

levels = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9
]

y_levels = np.quantile(y, levels)


# ============================================================
# GENERATE ONE FIGURE PER COLUMN
# ============================================================

for score_name in score_columns:

    print(f"\nGenerating: {score_name}")


    # --------------------------------------------------------
    # Check that every sample has this value
    # --------------------------------------------------------

    valid_mask = np.array([
        score_name in sample
        for sample in data
    ])

    if np.sum(valid_mask) < 4:

        print(
            f"  Skipping {score_name}: "
            f"not enough data"
        )

        continue


    # Extract the score
    score = np.array([
        sample[score_name]
        if score_name in sample
        else np.nan
        for sample in data
    ])

    # Remove NaN values
    valid = (
        valid_mask
        & np.isfinite(score)
        & np.isfinite(y)
        & np.isfinite(z)
        & np.isfinite(roll)
    )

    y_current = y[valid]
    z_current = z[valid]
    roll_current = roll[valid]
    score_current = score[valid]


    if len(score_current) < 4:

        print(
            f"  Skipping {score_name}: "
            f"not enough valid samples"
        )

        continue


    # --------------------------------------------------------
    # Color limits
    # --------------------------------------------------------

    # Same idea as your original code:
    # lower limit = value at 70% of sorted-by-file samples
    #
    # However, using percentiles is generally safer.

    # ========================================================
    # COLOR LIMITS
    # ========================================================

    vmin = np.quantile(score_current, 0.15)
    vmax = np.max(score_current) * 0.98

    # If the requested range is invalid, fall back to
    # the actual data range.
    if vmin > vmax:

        print(
            f"  Warning: vmin ({vmin:.6g}) > "
            f"vmax ({vmax:.6g}) for {score_name}. "
            "Using actual data range."
        )

        vmin = np.min(score_current)
        vmax = np.max(score_current)


    # Avoid zero-width color scale
    if np.isclose(vmin, vmax):

        if np.isclose(vmin, 0.0):
            vmin = 0.0
            vmax = 1e-12
        else:
            margin = abs(vmin) * 0.01
            vmin -= margin
            vmax += margin


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axs = plt.subplots(
        1,
        len(levels),
        figsize=(18, 5)
    )


    # --------------------------------------------------------
    # Generate each Y slice
    # --------------------------------------------------------

    for ax, y_target in zip(
        axs,
        y_levels
    ):

        # Same tolerance as your original code
        tol = 0.05 * (
            y_current.max()
            - y_current.min()
        )

        mask = (
            np.abs(y_current - y_target)
            < tol
        )


        if np.sum(mask) < 4:

            ax.set_title(
                f"y ≈ {y_target:.3f}\n"
                "Not enough points"
            )

            continue


        z_slice = z_current[mask]
        roll_slice = roll_current[mask]
        score_slice = score_current[mask]


        # ----------------------------------------------------
        # Interpolation grid
        # ----------------------------------------------------

        zi = np.linspace(
            z_current.min(),
            z_current.max(),
            100
        )

        ri = np.linspace(
            roll_current.min(),
            roll_current.max(),
            100
        )

        ZI, RI = np.meshgrid(
            zi,
            ri
        )


        # ----------------------------------------------------
        # LINEAR
        # ----------------------------------------------------

        if option == 0:

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
                    zi.min(),
                    zi.max(),
                    ri.min(),
                    ri.max()
                ],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ----------------------------------------------------
        # RBF
        # ----------------------------------------------------

        elif option == 1:

            rbf = Rbf(
                z_slice,
                roll_slice,
                score_slice,
                function="multiquadric"
            )

            SI = rbf(ZI, RI)

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
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ----------------------------------------------------
        # NADARAYA-WATSON
        # ----------------------------------------------------

        elif option == 2:

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
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ----------------------------------------------------
        # Scatter original samples
        # ----------------------------------------------------

        ax.scatter(
            z_slice,
            roll_slice,
            s=10
        )

        ax.set_xlabel("z")
        ax.set_ylabel("roll")

        ax.set_title(
            f"y ≈ {y_target:.3f}"
        )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"{score_name} — Z-Roll projections"
    )


    # ========================================================
    # COLORBAR
    # ========================================================

    fig.colorbar(
        im,
        ax=axs,
        label=score_name
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = os.path.join(
        output_dir,
        f"{score_name}.jpg"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"  Saved: {output_file}")


print("\nDone.")