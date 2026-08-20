import re
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# OPTIONS
# ============================================================

# 3D Nadaraya-Watson interpolation
option = 2

# ------------------------------------------------------------
# Number of interpolation points in each dimension
# ------------------------------------------------------------

n_y = 60
n_z = 60
n_roll = 60

# ------------------------------------------------------------
# Bandwidths
#
# These are in the ORIGINAL units of the variables.
# ------------------------------------------------------------

bandwidth_y = 0.05
bandwidth_z = 0.05
bandwidth_roll = 0.05

# ------------------------------------------------------------
# Y values at which Roll-Z slices will be displayed
#
# These are normalized positions:
#
# 0.0 -> minimum Y
# 0.5 -> middle Y
# 1.0 -> maximum Y
# ------------------------------------------------------------

levels = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]

levels = [
    0.0,
    #0.2,
    0.25,
    #0.5,
    0.75,
    #0.8,
    1.0
]


# ============================================================
# LOAD FILE
# ============================================================

filename = (
    "/home/rosdev/ros2_ws/src/"
    "moveit_cpp_demo/data/scores/"
    "score_ur5e_straight_2.txt"
)

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
    file_name + "_RZ_3D_NW"
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

pair_pattern = r"(\w+)=([-\d.eE]+)"

data = []

with open(filename, "r") as f:

    for line in f:

        pairs = dict(
            re.findall(
                pair_pattern,
                line
            )
        )

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

required_axes = [
    "y",
    "z",
    "roll"
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
                ordered_columns.append(key)


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
# 3D NADARAYA-WATSON REGRESSION
# ============================================================

def nw_kernel_regression_3d(
    y_train,
    z_train,
    roll_train,
    score_train,
    YI,
    ZI,
    RI,
    bandwidth_y,
    bandwidth_z,
    bandwidth_roll
):
    """
    3D Gaussian-kernel Nadaraya-Watson estimator.

    Estimates:

        score = f(Y, Z, Roll)

    using all three dimensions simultaneously.
    """

    # --------------------------------------------------------
    # Flatten interpolation grid
    # --------------------------------------------------------

    Y_flat = YI.ravel()
    Z_flat = ZI.ravel()
    R_flat = RI.ravel()

    prediction = np.zeros(
        len(Y_flat)
    )

    # --------------------------------------------------------
    # Process grid points in chunks
    # --------------------------------------------------------

    chunk_size = 1000

    for start in range(
        0,
        len(Y_flat),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(Y_flat)
        )

        Y_chunk = Y_flat[
            start:end
        ]

        Z_chunk = Z_flat[
            start:end
        ]

        R_chunk = R_flat[
            start:end
        ]

        # ----------------------------------------------------
        # Differences
        # ----------------------------------------------------

        dy = (
            Y_chunk[:, None]
            - y_train[None, :]
        )

        dz = (
            Z_chunk[:, None]
            - z_train[None, :]
        )

        dr = (
            R_chunk[:, None]
            - roll_train[None, :]
        )

        # ----------------------------------------------------
        # Squared normalized distance
        # ----------------------------------------------------

        d2 = (
            (dy / bandwidth_y) ** 2
            +
            (dz / bandwidth_z) ** 2
            +
            (dr / bandwidth_roll) ** 2
        )

        # ----------------------------------------------------
        # Gaussian kernel
        # ----------------------------------------------------

        weights = np.exp(
            -0.5 * d2
        )

        # ----------------------------------------------------
        # Nadaraya-Watson estimate
        # ----------------------------------------------------

        numerator = (
            weights
            @ score_train
        )

        denominator = (
            np.sum(
                weights,
                axis=1
            )
            + 1e-12
        )

        prediction[
            start:end
        ] = (
            numerator
            / denominator
        )

    return prediction.reshape(
        YI.shape
    )


# ============================================================
# GENERATE ONE FIGURE PER SCORE COLUMN
# ============================================================

for score_name in score_columns:

    # --------------------------------------------------------
    # Only plot the desired score
    # --------------------------------------------------------

    if score_name not in ["score"]:
        continue

    print(
        f"\nGenerating: {score_name}"
    )


    # ========================================================
    # CHECK THAT EVERY SAMPLE HAS THIS VALUE
    # ========================================================

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


    # ========================================================
    # EXTRACT SCORE
    # ========================================================

    score = np.array([
        sample[score_name]
        if score_name in sample
        else np.nan
        for sample in data
    ])


    # ========================================================
    # REMOVE INVALID VALUES
    # ========================================================

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
    # PRINT DATA RANGE
    # ========================================================

    print("\nData ranges:")

    print(
        f"  Y:    "
        f"{y_current.min():.4f} -> "
        f"{y_current.max():.4f}"
    )

    print(
        f"  Z:    "
        f"{z_current.min():.4f} -> "
        f"{z_current.max():.4f}"
    )

    print(
        f"  Roll:  "
        f"{roll_current.min():.4f} -> "
        f"{roll_current.max():.4f}"
    )

    print(
        f"  Score: "
        f"{score_current.min():.4f} -> "
        f"{score_current.max():.4f}"
    )


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


    if vmin > vmax:

        print(
            f"  Warning: vmin ({vmin:.6g}) > "
            f"vmax ({vmax:.6g}). "
            f"Using actual data range."
        )

        vmin = np.min(
            score_current
        )

        vmax = np.max(
            score_current
        )


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
    # CREATE 3D INTERPOLATION GRID
    # ========================================================

    print(
        "\nCreating 3D interpolation grid..."
    )

    yi = np.linspace(
        y_current.min(),
        y_current.max(),
        n_y
    )

    zi = np.linspace(
        z_current.min(),
        z_current.max(),
        n_z
    )

    ri = np.linspace(
        roll_current.min(),
        roll_current.max(),
        n_roll
    )


    YI, ZI, RI = np.meshgrid(
        yi,
        zi,
        ri,
        indexing="ij"
    )


    print(
        f"3D grid: "
        f"{n_y} x {n_z} x {n_roll}"
    )

    print(
        f"Total grid points: "
        f"{YI.size}"
    )


    # ========================================================
    # PERFORM ONE 3D NW INTERPOLATION
    # ========================================================

    print(
        "\nPerforming 3D "
        "Nadaraya-Watson interpolation..."
    )

    SI_3D = nw_kernel_regression_3d(
        y_current,
        z_current,
        roll_current,
        score_current,
        YI,
        ZI,
        RI,
        bandwidth_y,
        bandwidth_z,
        bandwidth_roll
    )


    print(
        "3D interpolation complete."
    )


    # ========================================================
    # CONVERT REQUESTED Y LEVELS
    # ========================================================
    #
    # levels:
    #
    # 0.0 -> minimum Y
    # 0.5 -> middle Y
    # 1.0 -> maximum Y
    #
    # ========================================================

    y_levels = (
        y_current.min()
        +
        np.array(levels)
        *
        (
            y_current.max()
            - y_current.min()
        )
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axs = plt.subplots(
        1,
        len(y_levels),
        figsize=(18, 5)
    )


    if len(y_levels) == 1:

        axs = [axs]


    # ========================================================
    # GENERATE ROLL-Z SLICES
    # ========================================================

    im = None


    for ax, y_target in zip(
        axs,
        y_levels
    ):

        # ----------------------------------------------------
        # Find nearest interpolated Y grid value
        # ----------------------------------------------------

        iy = np.argmin(
            np.abs(
                yi - y_target
            )
        )

        y_actual = yi[iy]


        # ----------------------------------------------------
        # Extract Z-Roll slice
        #
        # SI_3D dimensions:
        #
        #   [Y, Z, Roll]
        #
        # Fix Y and retain:
        #
        #   [Z, Roll]
        # ----------------------------------------------------

        SI = SI_3D[
            iy,
            :,
            :
        ]


        # ----------------------------------------------------
        # Plot
        #
        # SI has dimensions:
        #
        #   [Z, Roll]
        #
        # Transpose so:
        #
        #   x-axis = Z
        #   y-axis = Roll
        # ----------------------------------------------------

        im = ax.imshow(
            SI.T,
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
        # Original samples near this Y
        #
        # These are ONLY visualized.
        #
        # They are NOT used for interpolation.
        # ----------------------------------------------------

        visualization_tol = (
            0.5
            *
            (
                yi[1]
                - yi[0]
            )
        )


        mask = (
            np.abs(
                y_current
                - y_actual
            )
            < visualization_tol
        )


        ax.scatter(
            z_current[mask],
            roll_current[mask],
            s=10
        )


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "z"
        )

        ax.set_ylabel(
            "roll"
        )

        ax.set_title(
            f"y = {y_actual:.3f}"
        )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"{score_name} — "
        f"3D NW interpolation — "
        f"Roll-Z slices"
    )


    # ========================================================
    # COLORBAR
    # ========================================================

    if im is not None:

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
        f"{score_name}_RollZ.jpg"
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
        f"\nSaved: {output_file}"
    )


# ============================================================
# DONE
# ============================================================

print(
    "\nDone."
)