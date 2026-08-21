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
#
# Adjust these depending on the density of your data.
# ------------------------------------------------------------

bandwidth_y = 0.05
bandwidth_z = 0.05
bandwidth_roll = 0.05

# ------------------------------------------------------------
# Z values at which Y-Roll slices will be displayed
# ------------------------------------------------------------

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
    "score_ur5e_Lshape_5.txt"
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
    file_name + "_RY_3D_NW"
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

all_columns = set()

for sample in data:
    all_columns.update(
        sample.keys()
    )


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

    Parameters
    ----------
    y_train:
        Training Y coordinates.

    z_train:
        Training Z coordinates.

    roll_train:
        Training Roll coordinates.

    score_train:
        Score values.

    YI, ZI, RI:
        3D interpolation grid.

    bandwidth_y:
        Kernel bandwidth in Y.

    bandwidth_z:
        Kernel bandwidth in Z.

    bandwidth_roll:
        Kernel bandwidth in Roll.
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
    # Process grid points in chunks.
    #
    # This avoids creating one enormous
    # (grid_points x samples) matrix.
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
        # Differences between interpolation points
        # and training points
        #
        # Shape:
        #
        #   chunk_size x n_samples
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


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # indexing="ij" gives:
    #
    # YI.shape = (n_y, n_z, n_roll)
    # ZI.shape = (n_y, n_z, n_roll)
    # RI.shape = (n_y, n_z, n_roll)
    # --------------------------------------------------------

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
    # CONVERT REQUESTED Z LEVELS
    # ========================================================
    #
    # The levels are interpreted as normalized positions
    # between min(Z) and max(Z).
    #
    # For example:
    #
    # level = 0.0 -> minimum Z
    # level = 0.5 -> middle Z
    # level = 1.0 -> maximum Z
    #
    # ========================================================

    z_levels = (
        z_current.min()
        +
        np.array(levels)
        *
        (
            z_current.max()
            - z_current.min()
        )
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axs = plt.subplots(
        1,
        len(z_levels),
        figsize=(
            18,
            5
        )
    )


    # If there is only one level,
    # matplotlib does not return an array.
    if len(z_levels) == 1:

        axs = [axs]


    # ========================================================
    # GENERATE Y-ROLL SLICES
    # ========================================================

    im = None


    for ax, z_target in zip(
        axs,
        z_levels
    ):

        # ----------------------------------------------------
        # Find nearest interpolated Z grid value
        # ----------------------------------------------------

        iz = np.argmin(
            np.abs(
                zi - z_target
            )
        )

        z_actual = zi[iz]


        # ----------------------------------------------------
        # Extract Y-Roll slice
        #
        # SI_3D dimensions:
        #
        #   [Y, Z, Roll]
        #
        # ----------------------------------------------------

        SI = SI_3D[
            :,
            iz,
            :
        ]


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        im = ax.imshow(
            SI.T,
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


        # ----------------------------------------------------
        # Original samples near this Z
        #
        # These are ONLY visualized.
        #
        # They are NOT used for interpolation.
        # ----------------------------------------------------

        visualization_tol = (
            0.5
            *
            (
                zi[1]
                - zi[0]
            )
        )


        mask = (
            np.abs(
                z_current
                - z_actual
            )
            < visualization_tol
        )


        ax.scatter(
            y_current[mask],
            roll_current[mask],
            s=10
        )


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "y"
        )

        ax.set_ylabel(
            "roll"
        )

        ax.set_title(
            f"z = {z_actual:.3f}"
        )


    # ========================================================
    # FIGURE TITLE
    # ========================================================

    fig.suptitle(
        f"{score_name} — "
        f"3D NW interpolation — "
        f"Y-Roll slices"
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
        f"{score_name}_YRoll.jpg"
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