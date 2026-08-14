#!/usr/bin/env python3

import numpy as np

from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.ndimage import maximum_filter


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/"
    "raw_configuration_table.txt"
)

OUTPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/"
    "final_configuration_table_global2.txt"
)


# ============================================================
# TABLE COLUMNS
# ============================================================
#
# 0   iy
# 1   iz
# 2   iroll
# 3   Y
# 4   Z
# 5   Roll
# 6   candidate
# 7   quality
# 8   planning_ratio
# 9   q0
# 10  q1
# 11  q2
# 12  q3
# 13  q4
# 14  q5
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

# The joints that are actually optimized.
#
# These refer to the six-joint vector:
#
# [q0, q1, q2, q3, q4, q5]
#
VARIABLE_JOINTS = [1, 2, 3]


# ------------------------------------------------------------
# Quality weighting
# ------------------------------------------------------------

# Smaller value -> stronger preference for high-quality
# candidates.
#
QUALITY_TEMPERATURE = 0.5


# ------------------------------------------------------------
# Spatial kernel bandwidths
# ------------------------------------------------------------
#
# These are in the actual units of Y/Z/Roll.
#
# The placement domain being optimized here is:
#
#       Z, Roll
#
# because Y is constant in your table.
#

SPATIAL_SIGMA_Z = 0.10
SPATIAL_SIGMA_ROLL = 0.20


# ------------------------------------------------------------
# Configuration-space kernel
# ------------------------------------------------------------
#
# Gaussian bandwidth in normalized joint space.
#
# All variable joints are normalized to [0,1].
#

JOINT_SIGMA = 0.08


# ------------------------------------------------------------
# Mode detection
# ------------------------------------------------------------
#
# We construct a KDE in the complete joint space and search
# for density maxima.
#
# MODE_BANDWIDTH controls how much nearby configurations are
# considered part of the same density structure.
#

MODE_BANDWIDTH = 0.10

# Minimum density relative to the global maximum for a point
# to be considered a meaningful mode.
#
MODE_MIN_RELATIVE_DENSITY = 0.05

# Two detected modes closer than this in normalized joint
# space are merged.
#

MODE_MERGE_DISTANCE = 0.12


# ------------------------------------------------------------
# Global surrogate optimization
# ------------------------------------------------------------

OPTIMIZER_MAXITER = 300

OPTIMIZER_FTOL = 1e-9

OPTIMIZER_GTOL = 1e-6


# ------------------------------------------------------------
# Search bounds
# ------------------------------------------------------------

# The optimization is performed in normalized joint space:
#
#       0 <= q <= 1
#
# corresponding to the observed range of each joint.
#


# ============================================================
# LOAD TABLE
# ============================================================

def load_table(filename):

    data = np.loadtxt(
        filename,
        comments="#"
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 15:

        raise ValueError(
            f"Expected 15 columns, "
            f"but found {data.shape[1]}"
        )

    return data


# ============================================================
# QUALITY WEIGHTS
# ============================================================

def compute_quality_weights(data):

    """
    Convert quality into normalized weights separately
    at every spatial location.
    """

    weights = np.zeros(
        len(data)
    )

    # Group by placement indices.
    locations = {}

    for i, row in enumerate(data):

        key = (
            int(round(row[1])),   # iz
            int(round(row[2]))    # iroll
        )

        locations.setdefault(
            key,
            []
        ).append(i)

    for indices in locations.values():

        indices = np.asarray(
            indices,
            dtype=int
        )

        quality = data[
            indices,
            7
        ]

        x = quality / QUALITY_TEMPERATURE

        x -= np.max(x)

        w = np.exp(x)

        w_sum = np.sum(w)

        if w_sum > 0:

            w /= w_sum

        weights[
            indices
        ] = w

    return weights


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_joints(q):

    q_min = np.min(
        q,
        axis=0
    )

    q_max = np.max(
        q,
        axis=0
    )

    ranges = q_max - q_min

    ranges[
        ranges < 1e-12
    ] = 1.0

    q_norm = (
        q - q_min
    ) / ranges

    return (
        q_norm,
        q_min,
        q_max
    )


def denormalize_joints(
    q_norm,
    q_min,
    q_max
):

    return (
        q_norm
        *
        (q_max - q_min)
        +
        q_min
    )


# ============================================================
# MULTIVARIATE JOINT DISTANCE
# ============================================================

def squared_joint_distance(
    q,
    candidates
):

    """
    IMPORTANT:

    q and candidates are complete joint vectors.

    The distance is:

        ||q - q_candidate||^2

    in the full N-dimensional configuration space.

    It is NOT three independent estimators.
    """

    diff = (
        candidates
        -
        q
    )

    return np.sum(
        diff * diff,
        axis=1
    )


# ============================================================
# JOINT KDE
# ============================================================

def joint_log_kde(
    q,
    candidate_q,
    candidate_weights,
    sigma
):

    """
    Weighted multivariate Gaussian KDE.

    q:
        (3,)

    candidate_q:
        (N,3)

    candidate_weights:
        (N,)
    """

    if len(candidate_q) == 0:

        return -np.inf

    d2 = squared_joint_distance(
        q,
        candidate_q
    )

    log_kernel = (
        -0.5
        *
        d2
        /
        (sigma * sigma)
    )

    log_weights = np.log(
        np.maximum(
            candidate_weights,
            1e-300
        )
    )

    return logsumexp(
        log_weights
        +
        log_kernel
    )


# ============================================================
# MODE DETECTION
# ============================================================

def compute_joint_kde_grid(
    q_norm,
    weights,
    bandwidth,
    grid_size=35
):

    """
    Evaluate a KDE on a regular 3-D grid in normalized
    configuration space.

    This is used only for detecting the modes.

    The final surrogate itself is continuous and is not
    restricted to this grid.
    """

    axes = [
        np.linspace(
            0.0,
            1.0,
            grid_size
        )
        for _ in range(3)
    ]

    X, Y, Z = np.meshgrid(
        axes[0],
        axes[1],
        axes[2],
        indexing="ij"
    )

    grid_points = np.column_stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ]
    )

    density = np.zeros(
        len(grid_points)
    )

    # Process in chunks to avoid enormous memory usage.
    chunk_size = 2000

    for start in range(
        0,
        len(grid_points),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(grid_points)
        )

        chunk = grid_points[
            start:end
        ]

        diff = (
            chunk[:, None, :]
            -
            q_norm[None, :, :]
        )

        d2 = np.sum(
            diff * diff,
            axis=2
        )

        kernel = np.exp(
            -0.5
            *
            d2
            /
            (bandwidth * bandwidth)
        )

        density[
            start:end
        ] = kernel @ weights

    density = density.reshape(
        X.shape
    )

    return (
        axes,
        density
    )


def detect_joint_modes(
    q_norm,
    weights
):

    """
    Find density maxima of the complete 3-D joint KDE.

    Returns a list of mode centers in normalized joint space.
    """

    print()
    print("================================================")
    print("DETECTING N-D CONFIGURATION MODES")
    print("================================================")

    axes, density = compute_joint_kde_grid(
        q_norm,
        weights,
        MODE_BANDWIDTH
    )

    maximum = np.max(
        density
    )

    threshold = (
        maximum
        *
        MODE_MIN_RELATIVE_DENSITY
    )

    # Local maxima in the 3-D KDE.
    local_max = (
        density
        ==
        maximum_filter(
            density,
            size=3
        )
    )

    local_max &= (
        density >= threshold
    )

    indices = np.argwhere(
        local_max
    )

    modes = []

    for index in indices:

        center = np.array(
            [
                axes[0][index[0]],
                axes[1][index[1]],
                axes[2][index[2]]
            ]
        )

        value = density[
            tuple(index)
        ]

        modes.append(
            (
                value,
                center
            )
        )

    # Sort by density.
    modes.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # Merge very close modes.
    # --------------------------------------------------------

    final_modes = []

    for value, center in modes:

        too_close = False

        for _, existing in final_modes:

            distance = np.linalg.norm(
                center
                -
                existing
            )

            if distance < MODE_MERGE_DISTANCE:

                too_close = True
                break

        if not too_close:

            final_modes.append(
                (
                    value,
                    center
                )
            )

    print()

    print(
        f"Detected {len(final_modes)} "
        f"joint-space modes."
    )

    for i, (density_value, center) in enumerate(
        final_modes
    ):

        print(
            f"Mode {i}: "
            f"density={density_value:.6e}, "
            f"center={center}"
        )

    return [
        center
        for _, center in final_modes
    ]


# ============================================================
# ASSIGN CANDIDATES TO MODES
# ============================================================

def assign_candidates_to_modes(
    q_norm,
    modes
):

    """
    Assign every candidate to the nearest N-D mode.

    Importantly, the distance is computed using the COMPLETE
    joint vector.
    """

    if len(modes) == 0:

        raise RuntimeError(
            "No configuration modes detected."
        )

    mode_centers = np.asarray(
        modes
    )

    labels = np.zeros(
        len(q_norm),
        dtype=int
    )

    for i, q in enumerate(
        q_norm
    ):

        distances = np.linalg.norm(
            mode_centers
            -
            q,
            axis=1
        )

        labels[i] = np.argmin(
            distances
        )

    return labels


# ============================================================
# BUILD MODE DATA
# ============================================================

def build_mode_data(
    data,
    q_norm,
    weights,
    labels,
    mode
):

    """
    Organize candidates belonging to a particular mode
    by spatial location.
    """

    mode_data = {}

    indices = np.where(
        labels == mode
    )[0]

    for index in indices:

        row = data[
            index
        ]

        key = (
            int(round(row[1])),
            int(round(row[2]))
        )

        mode_data.setdefault(
            key,
            []
        ).append(
            index
        )

    # Build arrays for faster evaluation.
    prepared = {}

    for location, indices in mode_data.items():

        indices = np.asarray(
            indices,
            dtype=int
        )

        prepared[
            location
        ] = {
            "indices": indices,
            "q": q_norm[
                indices
            ],
            "weights": weights[
                indices
            ],
            "Z": data[
                indices,
                4
            ],
            "Roll": data[
                indices,
                5
            ]
        }

    return prepared


# ============================================================
# JOINT SPATIAL-CONFIGURATION SURROGATE
# ============================================================

def log_surrogate(
    Z,
    Roll,
    q,
    candidate_Z,
    candidate_Roll,
    candidate_q,
    candidate_weights
):

    """
    Evaluate:

        S(Z, Roll, q1, q2, q3)

    using a joint spatial + configuration KDE.

    The complete kernel is:

        K_Z
        *
        K_Roll
        *
        K_q

    where K_q is a multivariate Gaussian over the complete
    joint vector.
    """

    if len(candidate_q) == 0:

        return -np.inf

    # --------------------------------------------------------
    # Spatial distance
    # --------------------------------------------------------

    dz = (
        Z
        -
        candidate_Z
    )

    droll = (
        Roll
        -
        candidate_Roll
    )

    spatial_term = (
        -0.5
        *
        (
            dz * dz
            /
            (
                SPATIAL_SIGMA_Z
                *
                SPATIAL_SIGMA_Z
            )
            +
            droll * droll
            /
            (
                SPATIAL_SIGMA_ROLL
                *
                SPATIAL_SIGMA_ROLL
            )
        )
    )

    # --------------------------------------------------------
    # Complete N-D configuration distance
    # --------------------------------------------------------

    d2 = squared_joint_distance(
        q,
        candidate_q
    )

    joint_term = (
        -0.5
        *
        d2
        /
        (
            JOINT_SIGMA
            *
            JOINT_SIGMA
        )
    )

    # --------------------------------------------------------
    # Weighted KDE
    # --------------------------------------------------------

    log_weights = np.log(
        np.maximum(
            candidate_weights,
            1e-300
        )
    )

    return logsumexp(
        log_weights
        +
        spatial_term
        +
        joint_term
    )


# ============================================================
# OPTIMIZE SURROGATE AT ONE LOCATION
# ============================================================

def optimize_surrogate_at_location(
    Z,
    Roll,
    mode_data,
    initial_q
):

    """
    Find:

        argmax_q S(Z, Roll, q)

    """

    def objective(q):

        return -log_surrogate(
            Z,
            Roll,
            q,
            mode_data["Z"],
            mode_data["Roll"],
            mode_data["q"],
            mode_data["weights"]
        )

    result = minimize(
        objective,
        initial_q,
        method="L-BFGS-B",
        bounds=[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0)
        ],
        options={
            "maxiter": OPTIMIZER_MAXITER,
            "ftol": OPTIMIZER_FTOL,
            "gtol": OPTIMIZER_GTOL
        }
    )

    q_opt = result.x

    strength = -result.fun

    return (
        q_opt,
        strength,
        result
    )


# ============================================================
# OPTIMIZE COMPLETE MODE
# ============================================================

def optimize_mode(
    mode,
    mode_data,
    data,
    q_min,
    q_max
):

    """
    Optimize one surrogate mode over every grid location.

    NOTE:

    This is now interpolation of ONE continuous surrogate.

    The locations are not optimized against each other through
    a separate graph penalty.

    Their relationship is encoded by the spatial kernels of
    the surrogate itself.
    """

    print()
    print("================================================")
    print(f"OPTIMIZING SURROGATE MODE {mode}")
    print("================================================")

    # All grid locations in the original table.
    locations = sorted(
        {
            (
                int(round(row[1])),
                int(round(row[2]))
            )
            for row in data
        }
    )

    print(
        f"Evaluating {len(locations)} grid locations."
    )

    # --------------------------------------------------------
    # Candidate data for this mode
    # --------------------------------------------------------

    indices = []

    for location_data in mode_data.values():

        indices.extend(
            location_data["indices"]
        )

    indices = np.asarray(
        indices,
        dtype=int
    )

    # candidate_q = data[
    #     indices,
    #     9 + np.array(VARIABLE_JOINTS)
    # ]

    candidate_q = data[
        indices
    ][:, 9 + np.array(VARIABLE_JOINTS)]

    # Normalize with the same ranges as the complete data.
    variable_min = q_min
    variable_max = q_max

    ranges = (
        variable_max
        -
        variable_min
    )

    ranges[
        ranges < 1e-12
    ] = 1.0

    candidate_q = (
        candidate_q
        -
        variable_min
    ) / ranges

    candidate_Z = data[
        indices,
        4
    ]

    candidate_Roll = data[
        indices,
        5
    ]

    candidate_weights = np.ones(
        len(indices)
    )

    # Use quality weighting.
    #
    # The original weights are reconstructed from quality.
    #
    quality = data[
        indices,
        7
    ]

    quality_x = (
        quality
        /
        QUALITY_TEMPERATURE
    )

    quality_x -= np.max(
        quality_x
    )

    candidate_weights = np.exp(
        quality_x
    )

    candidate_weights /= np.sum(
        candidate_weights
    )

    surrogate_data = {
        "q": candidate_q,
        "Z": candidate_Z,
        "Roll": candidate_Roll,
        "weights": candidate_weights
    }

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    best_index = np.argmax(
        candidate_weights
    )

    initial_q = candidate_q[
        best_index
    ]

    results = {}

    # --------------------------------------------------------
    # Evaluate all spatial locations
    # --------------------------------------------------------

    for counter, location in enumerate(
        locations
    ):

        iz, iroll = location

        # Recover actual placement coordinates.
        #
        # Find one row at this grid point.
        #

        matching = np.where(
            (
                data[:, 1].astype(int)
                ==
                iz
            )
            &
            (
                data[:, 2].astype(int)
                ==
                iroll
            )
        )[0]

        if len(matching) == 0:
            continue

        reference_row = data[
            matching[0]
        ]

        Z = reference_row[4]
        Roll = reference_row[5]

        q_opt, strength, result = \
            optimize_surrogate_at_location(
                Z,
                Roll,
                surrogate_data,
                initial_q
            )

        results[
            location
        ] = {
            "q_norm": q_opt,
            "strength": strength,
            "result": result
        }

        # Warm start next location with previous optimum.
        initial_q = q_opt

        if (
            counter % 10 == 0
            or
            counter == len(locations) - 1
        ):

            print(
                f"  {counter + 1}/"
                f"{len(locations)}"
            )

    return {
        "locations": locations,
        "results": results,
        "mode_data": mode_data
    }


# ============================================================
# SELECT FINAL CONFIGURATION
# ============================================================

def select_final_configurations(
    data,
    mode_results,
    q_min,
    q_max
):

    """
    At every spatial location choose the mode with the
    strongest surrogate value.
    """

    locations = sorted(
        {
            (
                int(round(row[1])),
                int(round(row[2]))
            )
            for row in data
        }
    )

    final_rows = []

    for location in locations:

        candidates = []

        for mode, result in mode_results.items():

            if location not in result[
                "results"
            ]:
                continue

            entry = result[
                "results"
            ][location]

            candidates.append(
                (
                    entry["strength"],
                    mode,
                    entry["q_norm"]
                )
            )

        if len(candidates) == 0:

            continue

        # Strongest surrogate.
        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        strength, mode, q_norm = \
            candidates[0]

        # ----------------------------------------------------
        # Find original row at this location
        # ----------------------------------------------------

        matching = np.where(
            (
                data[:, 1].astype(int)
                ==
                location[0]
            )
            &
            (
                data[:, 2].astype(int)
                ==
                location[1]
            )
        )[0]

        if len(matching) == 0:
            continue

        # Use highest-quality original candidate as template.
        #
        template_index = matching[
            np.argmax(
                data[
                    matching,
                    7
                ]
            )
        ]

        row = data[
            template_index
        ].copy()

        # ----------------------------------------------------
        # Optimized variable joints
        # ----------------------------------------------------

        variable_min = q_min
        variable_max = q_max

        q_opt = denormalize_joints(
            q_norm,
            variable_min,
            variable_max
        )

        for local_index, joint_index in enumerate(
            VARIABLE_JOINTS
        ):

            # q0 starts at column 9.
            column = 9 + joint_index

            row[
                column
            ] = q_opt[
                local_index
            ]

        # ----------------------------------------------------
        # Store mode / surrogate information
        # ----------------------------------------------------

        row[6] = mode

        row[7] = strength

        #
        # planning_ratio remains unchanged.
        #

        final_rows.append(
            row
        )

    return np.asarray(
        final_rows
    )


# ============================================================
# SAVE
# ============================================================

def save_table(
    filename,
    data
):

    if data.ndim != 2:

        raise ValueError(
            f"Expected 2-D data, "
            f"got {data.shape}"
        )

    if data.shape[1] != 15:

        raise ValueError(
            f"Expected 15 columns, "
            f"got {data.shape[1]}"
        )

    header = (
        "iy iz iroll Y Z Roll candidate quality "
        "planning_ratio q0 q1 q2 q3 q4 q5"
    )

    fmt = [
        "%d",       # iy
        "%d",       # iz
        "%d",       # iroll
        "%.8f",     # Y
        "%.8f",     # Z
        "%.8f",     # Roll
        "%d",       # candidate / mode
        "%.8f",     # quality / surrogate
        "%.8f",     # planning_ratio
        "%.8f",     # q0
        "%.8f",     # q1
        "%.8f",     # q2
        "%.8f",     # q3
        "%.8f",     # q4
        "%.8f",     # q5
    ]

    np.savetxt(
        filename,
        data,
        fmt=fmt,
        header=header,
        comments=""
    )

    print()
    print(
        f"Saved {len(data)} rows to:"
    )
    print(filename)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("================================================")
    print("JOINT SPATIAL-CONFIGURATION SURROGATE")
    print("================================================")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_table(
        INPUT_FILE
    )

    print(
        f"Loaded {len(data)} candidates."
    )

    print(
        f"Input shape: {data.shape}"
    )

    # --------------------------------------------------------
    # Extract optimized joints
    # --------------------------------------------------------

    #
    # q0 is column 9
    # q1 is column 10
    # q2 is column 11
    # q3 is column 12
    # q4 is column 13
    # q5 is column 14
    #

    all_joints = data[
        :,
        9:15
    ]

    variable_q = all_joints[
        :,
        VARIABLE_JOINTS
    ]

    print(
        "Optimized joints:",
        VARIABLE_JOINTS
    )

    print(
        "Variable joint dimension:",
        variable_q.shape[1]
    )

    # --------------------------------------------------------
    # Normalize configuration space
    # --------------------------------------------------------

    q_norm, q_min, q_max = \
        normalize_joints(
            variable_q
        )

    # --------------------------------------------------------
    # Quality weights
    # --------------------------------------------------------

    quality_weights = \
        compute_quality_weights(
            data
        )

    # --------------------------------------------------------
    # Detect modes in complete N-D configuration space
    # --------------------------------------------------------

    modes = detect_joint_modes(
        q_norm,
        quality_weights
    )

    # --------------------------------------------------------
    # Assign candidates to modes
    # --------------------------------------------------------

    labels = assign_candidates_to_modes(
        q_norm,
        modes
    )

    print()

    for mode in range(
        len(modes)
    ):

        count = np.sum(
            labels == mode
        )

        print(
            f"Mode {mode}: "
            f"{count} candidates"
        )

    # --------------------------------------------------------
    # Build mode-specific candidate sets
    # --------------------------------------------------------

    mode_data_all = {}

    for mode in range(
        len(modes)
    ):

        mode_data = build_mode_data(
            data,
            q_norm,
            quality_weights,
            labels,
            mode
        )

        mode_data_all[
            mode
        ] = mode_data

    # --------------------------------------------------------
    # Optimize each mode independently
    # --------------------------------------------------------

    mode_results = {}

    for mode in range(
        len(modes)
    ):

        if len(
            mode_data_all[mode]
        ) == 0:

            continue

        result = optimize_mode(
            mode,
            mode_data_all[mode],
            data,
            q_min,
            q_max
        )

        mode_results[
            mode
        ] = result

    # --------------------------------------------------------
    # Select strongest mode at each location
    # --------------------------------------------------------

    print()
    print("================================================")
    print("SELECTING FINAL CONFIGURATIONS")
    print("================================================")

    final_data = \
        select_final_configurations(
            data,
            mode_results,
            q_min,
            q_max
        )

    print(
        "Final data shape:",
        final_data.shape
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_table(
        OUTPUT_FILE,
        final_data
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()