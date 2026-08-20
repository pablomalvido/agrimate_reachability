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

# filename = "/home/rosdev/ros2_ws/src/reach_pkg/data/history_ur5_length_0_8.txt"
filename = "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/scores/score_ur5e_straight_1.txt"

with open(filename, "r") as f:
    text = f.read()


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

file_name = os.path.splitext(
    os.path.basename(filename)
)[0]

output_dir = os.path.join(
    os.path.dirname(filename),
    file_name+"_RY"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

print(
    f"Saving figures to: {output_dir}"
)


# ============================================================
# PARSE DATA
# ============================================================

# Find all key=value pairs.
#
# This supports an arbitrary number of columns:
#
# y=-0.97
# z=0.27
# roll=0.58
# score=0
# score_prune=0
# ...
#
pair_pattern = r"(\w+)=([-\d.eE]+)"


# ============================================================
# PARSE DATA LINE BY LINE
# ============================================================

data = []

with open(filename, "r") as f:

    for line in f:

        pairs = dict(
            re.findall(
                pair_pattern,
                line
            )
        )

        # Ignore lines without the required coordinates
        if not all(
            k in pairs
            for k in ["y", "z", "roll"]
        ):
            continue

        sample = {
            key: float(value)
            for key, value in pairs.items()
        }

        data.append(sample)


if len(data) == 0:
    raise RuntimeError(
        "No valid samples found in the file."
    )


print(
    f"Loaded {len(data)} samples"
)


# ============================================================
# GET AVAILABLE COLUMNS
# ============================================================

all_columns = set()

for sample in data:
    all_columns.update(
        sample.keys()
    )


# The three spatial/orientation variables
required_axes = [
    "y",
    "z",
    "roll"
]


# Everything else will be plotted
score_columns = [
    column
    for column in all_columns
    if column not in required_axes
    and column != "Iteration"
]


# ============================================================
# PRESERVE ORIGINAL COLUMN ORDER
# ============================================================

ordered_columns = []

with open(filename, "r") as f:

    for line in f:

        pairs = re.findall(
            pair_pattern,
            line
        )

        for key, value in pairs:

            if key not in ordered_columns:
                ordered_columns.append(
                    key
                )


score_columns = [
    column
    for column in ordered_columns
    if column not in required_axes
    and column != "Iteration"
]


print("\nColumns found:")

for column in score_columns:
    print(
        f"  {column}"
    )


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

y = np.array([
    sample["y"]
    for sample in data
])

z = np.array([
    sample["z"]
    for sample in data
])

roll = np.array([
    sample["roll"]
    for sample in data
])


# ============================================================
# NADARAYA-WATSON REGRESSION
# ============================================================

def nw_kernel_regression(
    x_train,
    roll_train,
    score_train,
    XI,
    RI,
    bandwidth=0.12
):
    """
    Gaussian-kernel Nadaraya-Watson estimator.

    x_train:
        First spatial variable, here Y.

    roll_train:
        Roll values.

    score_train:
        Score values.

    XI, RI:
        Interpolation grid.
    """

    pred = np.zeros_like(
        XI
    )

    for i in range(
        XI.shape[0]
    ):

        for j in range(
            XI.shape[1]
        ):

            dx = (
                x_train
                - XI[i, j]
            )

            dr = (
                roll_train
                - RI[i, j]
            )

            d2 = (
                dx**2
                + dr**2
            )

            w = np.exp(
                -0.5
                * d2
                / bandwidth**2
            )

            pred[i, j] = (
                np.sum(
                    w * score_train
                )
                /
                (
                    np.sum(w)
                    + 1e-12
                )
            )

    return pred


# ============================================================
# Z LEVELS
# ============================================================

# levels = [
#     0.1,
#     0.3,
#     0.5,
#     0.7,
#     0.9
# ]

levels = [
    0.0,
    #0.1,
    0.2,
    #0.3,
    0.4,
    #0.5,
    0.6,
    #0.7,
    0.8,
    #0.9,
    1.0
]

# Different Z values for the five projections
z_levels = np.quantile(
    z,
    levels
)


# ============================================================
# GENERATE ONE FIGURE PER SCORE COLUMN
# ============================================================

for score_name in score_columns:

    if not (score_name in ['score']):
        continue

    print(
        f"\nGenerating: {score_name}"
    )


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


    # --------------------------------------------------------
    # Extract score
    # --------------------------------------------------------

    score = np.array([
        sample[score_name]
        if score_name in sample
        else np.nan
        for sample in data
    ])


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

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


    # ========================================================
    # COLOR LIMITS
    # ========================================================

    vmin = np.quantile(
        score_current,
        0.15
    )

    vmax = (
        np.max(score_current)
        * 0.98
    )


    # If the requested range is invalid,
    # fall back to actual data range.
    if vmin > vmax:

        print(
            f"  Warning: vmin ({vmin:.6g}) > "
            f"vmax ({vmax:.6g}) for "
            f"{score_name}. "
            f"Using actual data range."
        )

        vmin = np.min(
            score_current
        )

        vmax = np.max(
            score_current
        )


    # Avoid zero-width color scale
    if np.isclose(
        vmin,
        vmax
    ):

        if np.isclose(
            vmin,
            0.0
        ):

            vmin = 0.0
            vmax = 1e-12

        else:

            margin = (
                abs(vmin)
                * 0.01
            )

            vmin -= margin
            vmax += margin


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axs = plt.subplots(
        1,
        len(levels),
        figsize=(18, 5)
    )


    # ========================================================
    # GENERATE EACH Z SLICE
    # ========================================================

    for ax, z_target in zip(
        axs,
        z_levels
    ):

        # ----------------------------------------------------
        # Tolerance around target Z
        # ----------------------------------------------------

        tol = 0.05 * (
            z_current.max()
            - z_current.min()
        )


        # Select samples close to this Z
        mask = (
            np.abs(
                z_current
                - z_target
            )
            < tol
        )


        if np.sum(mask) < 4:

            ax.set_title(
                f"z ≈ {z_target:.3f}\n"
                "Not enough points"
            )

            continue


        # ----------------------------------------------------
        # Data for this Z slice
        # ----------------------------------------------------

        y_slice = y_current[
            mask
        ]

        roll_slice = roll_current[
            mask
        ]

        score_slice = score_current[
            mask
        ]


        # ----------------------------------------------------
        # Y-Roll interpolation grid
        # ----------------------------------------------------

        yi = np.linspace(
            y_current.min(),
            y_current.max(),
            100
        )

        ri = np.linspace(
            roll_current.min(),
            roll_current.max(),
            100
        )


        YI, RI = np.meshgrid(
            yi,
            ri
        )


        # ====================================================
        # LINEAR
        # ====================================================

        if option == 0:

            SI = griddata(
                (
                    y_slice,
                    roll_slice
                ),
                score_slice,
                (
                    YI,
                    RI
                ),
                method="linear"
            )


            im = ax.imshow(
                SI,
                origin="lower",
                extent=[
                    yi.min(),
                    yi.max(),
                    ri.min(),
                    ri.max()
                ],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ====================================================
        # RBF
        # ====================================================

        elif option == 1:

            rbf = Rbf(
                y_slice,
                roll_slice,
                score_slice,
                function="multiquadric"
            )


            SI = rbf(
                YI,
                RI
            )


            im = ax.imshow(
                SI,
                origin="lower",
                extent=[
                    yi.min(),
                    yi.max(),
                    ri.min(),
                    ri.max()
                ],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ====================================================
        # NADARAYA-WATSON
        # ====================================================

        elif option == 2:

            SI = nw_kernel_regression(
                y_slice,
                roll_slice,
                score_slice,
                YI,
                RI,
                bandwidth=0.05
            )


            im = ax.imshow(
                SI,
                origin="lower",
                extent=[
                    yi.min(),
                    yi.max(),
                    ri.min(),
                    ri.max()
                ],
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax
            )


        # ====================================================
        # ORIGINAL SAMPLES
        # ====================================================

        ax.scatter(
            y_slice,
            roll_slice,
            s=10
        )


        ax.set_xlabel(
            "y"
        )

        ax.set_ylabel(
            "roll"
        )


        ax.set_title(
            f"z ≈ {z_target:.3f}"
        )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"{score_name} — Y-Roll projections"
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


    plt.close(
        fig
    )


    print(
        f"  Saved: {output_file}"
    )


# ============================================================
# DONE
# ============================================================

print("\nDone.")