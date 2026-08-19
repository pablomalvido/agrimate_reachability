#!/usr/bin/env python3

import numpy as np

from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.cluster import DBSCAN


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/initial_config_optimization/"
    "raw_configuration_table_upsidedown_straight.txt"
)

OUTPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/initial_config_optimization/"
    "final_configuration_table_upsidedown_straight.txt"
)


# ============================================================
# TABLE FORMAT
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

# Joints that vary and therefore participate in the
# 3-dimensional heatmap.
#
# These are indices inside [q0,q1,q2,q3,q4,q5].
#
VARIABLE_JOINTS = [1, 2, 3]


# ------------------------------------------------------------
# Spatial kernel
# ------------------------------------------------------------
#
# A 3x3 kernel:
#
#       x x x
#       x C x
#       x x x
#
KERNEL_RADIUS = 1


# Spatial weighting.
#
# 1.0 means:
#
#   center and neighbors have equal importance
#
# Smaller values make the center more important.
#
# This is a Gaussian spatial kernel:
#
#     exp(-distance^2 / (2*sigma^2))
#
SPATIAL_SIGMA = 1.0


# ------------------------------------------------------------
# Joint-space Gaussian
# ------------------------------------------------------------
#
# The Gaussian is evaluated in the COMPLETE 3-D joint space.
#
# q = [q1,q2,q3]
#
# Smaller -> sharper peaks
# Larger  -> more diffusion
#
JOINT_SIGMA = 0.10


# ------------------------------------------------------------
# Quality -> heat
# ------------------------------------------------------------
#
# The quality is converted into a positive weight.
#
# If quality is already a suitable positive score, this can
# simply be quality.
#
# If larger quality means better configuration, this is
# appropriate.
#
QUALITY_POWER = 2.0


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

OPTIMIZER_MAXITER = 200
OPTIMIZER_FTOL = 1e-10
OPTIMIZER_GTOL = 1e-7


# ------------------------------------------------------------
# Cluster detection
# ------------------------------------------------------------
#
# Clustering is performed in normalized q1/q2/q3 space.
#

CLUSTER_EPS = 0.25
CLUSTER_MIN_SAMPLES = 2


# ------------------------------------------------------------
# Cluster reassignment
# ------------------------------------------------------------

MAX_CLUSTER_REFINEMENT_ITERATIONS = 10

CLUSTER_CONVERGENCE_TOL = 1e-8


# ============================================================
# LOAD
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
            f"got {data.shape[1]}"
        )

    return data


# ============================================================
# EXTRACT VARIABLE JOINTS
# ============================================================

def extract_variable_joints(data):

    #
    # q0...q5 start at column 9.
    #

    joint_columns = [
        9 + j
        for j in VARIABLE_JOINTS
    ]

    return data[
        :,
        joint_columns
    ]


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
# BUILD GRID
# ============================================================

def build_grid(data):

    """
    Build:

        (iz, iroll) -> list of candidate row indices
    """

    grid = {}

    for i, row in enumerate(data):

        iz = int(round(row[1]))
        iroll = int(round(row[2]))

        key = (
            iz,
            iroll
        )

        grid.setdefault(
            key,
            []
        ).append(i)

    return grid


# ============================================================
# FIND AVAILABLE SPATIAL NEIGHBORS
# ============================================================

def get_neighbors(
    iz,
    iroll,
    grid
):

    neighbors = []

    for dz in range(
        -KERNEL_RADIUS,
        KERNEL_RADIUS + 1
    ):

        for dr in range(
            -KERNEL_RADIUS,
            KERNEL_RADIUS + 1
        ):

            neighbor = (
                iz + dz,
                iroll + dr
            )

            if neighbor in grid:

                neighbors.append(
                    (
                        neighbor,
                        dz,
                        dr
                    )
                )

    return neighbors


# ============================================================
# SPATIAL WEIGHT
# ============================================================

def spatial_weight(
    dz,
    dr
):

    distance_squared = (
        dz * dz
        +
        dr * dr
    )

    return np.exp(
        -0.5
        *
        distance_squared
        /
        (
            SPATIAL_SIGMA
            *
            SPATIAL_SIGMA
        )
    )


# ============================================================
# QUALITY -> HEAT
# ============================================================

def quality_to_weight(
    quality
):

    #
    # Make sure there are no negative weights.
    #

    quality = max(
        float(quality),
        0.0
    )

    return (
        quality
        **
        QUALITY_POWER
    )


# ============================================================
# PREPARE LOCAL HEAT SOURCES
# ============================================================

def collect_local_candidates(
    center,
    grid,
    q_norm,
    data,
    allowed_rows=None
):

    """
    Collect candidates from the 3x3 spatial neighborhood.

    Returns:

        candidate_q
        candidate_weights

    Each candidate remains a COMPLETE 3-D point.
    """

    iz, iroll = center

    local_q = []
    local_weights = []

    neighbors = get_neighbors(
        iz,
        iroll,
        grid
    )

    allowed_set = None

    if allowed_rows is not None:

        allowed_set = set(
            allowed_rows
        )

    for (
        location,
        dz,
        dr
    ) in neighbors:

        spatial_w = spatial_weight(
            dz,
            dr
        )

        row_indices = grid[
            location
        ]

        for row_index in row_indices:

            if (
                allowed_set is not None
                and
                row_index not in allowed_set
            ):
                continue

            q = q_norm[
                row_index
            ]

            quality = data[
                row_index,
                7
            ]

            quality_w = quality_to_weight(
                quality
            )

            total_weight = (
                spatial_w
                *
                quality_w
            )

            local_q.append(
                q
            )

            local_weights.append(
                total_weight
            )

    if len(local_q) == 0:

        return (
            np.empty((0, 3)),
            np.empty(0)
        )

    return (
        np.asarray(local_q),
        np.asarray(local_weights)
    )


# ============================================================
# 3-D HEATMAP / KDE
# ============================================================

def log_heat(
    q,
    candidate_q,
    candidate_weights
):

    """
    Evaluate the 3-D heat at q.

    Every candidate is one complete 3-D Gaussian source.

        candidate_q[k] = [q1,q2,q3]

    """

    if len(candidate_q) == 0:

        return -np.inf

    diff = (
        candidate_q
        -
        q
    )

    squared_distance = np.sum(
        diff * diff,
        axis=1
    )

    kernel = (
        -0.5
        *
        squared_distance
        /
        (
            JOINT_SIGMA
            *
            JOINT_SIGMA
        )
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
        kernel
    )


# ============================================================
# FIND PEAK OF 3-D HEATMAP
# ============================================================

def find_heat_peak(
    candidate_q,
    candidate_weights,
    initial=None
):

    """
    Find:

        argmax_q Heat(q)

    in normalized 3-D joint space.
    """

    if len(candidate_q) == 0:

        return None, -np.inf

    # --------------------------------------------------------
    # Multiple starting points are useful because a KDE may
    # have multiple peaks.
    # --------------------------------------------------------

    starts = []

    if initial is not None:

        starts.append(
            initial
        )

    # Start from strongest candidates.
    strongest = np.argsort(
        candidate_weights
    )[
        -min(5, len(candidate_q)):
    ]

    for index in strongest:

        starts.append(
            candidate_q[index]
        )

    best_q = None
    best_value = -np.inf

    def objective(q):

        return -log_heat(
            q,
            candidate_q,
            candidate_weights
        )

    for start in starts:

        result = minimize(
            objective,
            np.clip(
                start,
                0.0,
                1.0
            ),
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

        value = -result.fun

        if value > best_value:

            best_value = value
            best_q = result.x

    return (
        best_q,
        best_value
    )


# ============================================================
# FIRST PASS: CONVOLUTION OVER ENTIRE TABLE
# ============================================================

def run_convolution(
    data,
    grid,
    q_norm,
    allowed_rows_by_cell=None,
    initial_values=None
):

    """
    Apply the 3x3 spatial kernel to every grid cell.

    The output is one optimized 3-D joint vector per cell.
    """

    results = {}

    locations = sorted(
        grid.keys()
    )

    print()
    print(
        f"Processing {len(locations)} grid cells..."
    )

    for counter, location in enumerate(
        locations
    ):

        allowed_rows = None

        if allowed_rows_by_cell is not None:

            allowed_rows = \
                allowed_rows_by_cell.get(
                    location,
                    None
                )

        initial = None

        if initial_values is not None:

            initial = initial_values.get(
                location,
                None
            )

        candidate_q, candidate_weights = \
            collect_local_candidates(
                location,
                grid,
                q_norm,
                data,
                allowed_rows
            )

        q_peak, heat_peak = \
            find_heat_peak(
                candidate_q,
                candidate_weights,
                initial
            )

        if q_peak is None:
            continue

        results[
            location
        ] = {
            "q_norm": q_peak,
            "heat": heat_peak
        }

        if (
            counter % 10 == 0
            or
            counter == len(locations) - 1
        ):

            print(
                f"  {counter + 1}/"
                f"{len(locations)}"
            )

    return results


# ============================================================
# CLUSTER FINAL GRID CONFIGURATIONS
# ============================================================

def cluster_grid_results(
    results
):

    locations = sorted(
        results.keys()
    )

    values = np.asarray(
        [
            results[x]["q_norm"]
            for x in locations
        ]
    )

    clustering = DBSCAN(
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_SAMPLES
    ).fit(
        values
    )

    labels = clustering.labels_

    # --------------------------------------------------------
    # DBSCAN uses -1 for noise.
    #
    # For our purpose, each isolated/noise point should still
    # belong to the closest real cluster if clusters exist.
    # --------------------------------------------------------

    real_clusters = sorted(
        set(labels)
        -
        {-1}
    )

    if len(real_clusters) == 0:

        # Everything is noise.
        labels[:] = 0

        real_clusters = [0]

    elif np.any(labels == -1):

        centroids = {}

        for cluster in real_clusters:

            mask = (
                labels == cluster
            )

            centroids[
                cluster
            ] = np.mean(
                values[mask],
                axis=0
            )

        for i in np.where(
            labels == -1
        )[0]:

            distances = {
                cluster:
                np.linalg.norm(
                    values[i]
                    -
                    centroid
                )
                for cluster, centroid
                in centroids.items()
            }

            labels[i] = min(
                distances,
                key=distances.get
            )

    return (
        locations,
        labels
    )


# ============================================================
# FIND CLUSTER CENTROIDS
# ============================================================

def compute_cluster_centroids(
    locations,
    labels,
    results
):

    centroids = {}

    for cluster in sorted(
        set(labels)
    ):

        values = []

        for location, label in zip(
            locations,
            labels
        ):

            if label == cluster:

                values.append(
                    results[
                        location
                    ]["q_norm"]
                )

        if len(values) > 0:

            centroids[
                cluster
            ] = np.mean(
                values,
                axis=0
            )

    return centroids


# ============================================================
# FIND CLUSTER BOUNDARY CELLS
# ============================================================

def find_cluster_boundaries(
    grid,
    locations,
    labels
):

    label_by_location = {
        location: label
        for location, label
        in zip(
            locations,
            labels
        )
    }

    boundaries = []

    for location in locations:

        label = label_by_location[
            location
        ]

        iz, iroll = location

        neighboring_labels = set()

        # 4-connected spatial neighborhood.
        #
        # We use 4-connected neighbors to identify actual
        # boundaries between regions.
        #

        for dz, dr in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]:

            neighbor = (
                iz + dz,
                iroll + dr
            )

            if neighbor not in label_by_location:
                continue

            neighbor_label = \
                label_by_location[
                    neighbor
                ]

            if neighbor_label != label:

                neighboring_labels.add(
                    neighbor_label
                )

        if len(
            neighboring_labels
        ) > 0:

            boundaries.append(
                (
                    location,
                    label,
                    neighboring_labels
                )
            )

    return boundaries


# ============================================================
# REASSIGN CLUSTER BOUNDARY
# ============================================================

def determine_best_cluster_for_boundary(
    location,
    original_q,
    neighboring_clusters,
    cluster_centroids
):

    """
    Determine which configuration cluster the ORIGINAL
    configuration of the cell belongs to.

    This is deliberately based on the ORIGINAL value rather
    than the already-smoothed value.
    """

    best_cluster = None
    best_distance = np.inf

    for cluster in neighboring_clusters:

        if cluster not in cluster_centroids:
            continue

        distance = np.linalg.norm(
            original_q
            -
            cluster_centroids[
                cluster
            ]
        )

        if distance < best_distance:

            best_distance = distance
            best_cluster = cluster

    return best_cluster


# ============================================================
# BUILD ALLOWED ROWS FOR CLUSTER
# ============================================================

def build_allowed_rows(
    data,
    grid,
    cluster_labels,
    locations,
    selected_boundary_clusters
):

    """
    For every cell that has been assigned to a particular
    cluster, restrict its local convolution to candidates
    belonging to that cluster.

    Non-boundary cells are left unrestricted.
    """

    label_by_location = {
        location: label
        for location, label
        in zip(
            locations,
            cluster_labels
        )
    }

    allowed_rows_by_cell = {}

    for location in locations:

        label = label_by_location[
            location
        ]

        if location in selected_boundary_clusters:

            label = selected_boundary_clusters[
                location
            ]

        allowed = []

        # ----------------------------------------------------
        # The cluster restriction applies to all candidates
        # in the local neighborhood.
        # ----------------------------------------------------

        neighbors = get_neighbors(
            location[0],
            location[1],
            grid
        )

        for (
            neighbor,
            dz,
            dr
        ) in neighbors:

            neighbor_label = \
                label_by_location[
                    neighbor
                ]

            if neighbor in selected_boundary_clusters:

                neighbor_label = \
                    selected_boundary_clusters[
                        neighbor
                    ]

            if neighbor_label != label:
                continue

            allowed.extend(
                grid[
                    neighbor
                ]
            )

        allowed_rows_by_cell[
            location
        ] = allowed

    return allowed_rows_by_cell


# ============================================================
# CLUSTER REFINEMENT
# ============================================================

def refine_clusters(
    data,
    grid,
    q_norm,
    original_best_q,
    results
):

    """
    Iteratively resolve boundaries between configuration
    clusters.

    """

    previous_labels = None

    current_results = results

    for iteration in range(
        MAX_CLUSTER_REFINEMENT_ITERATIONS
    ):

        print()
        print(
            "------------------------------------------------"
        )
        print(
            f"CLUSTER REFINEMENT ITERATION "
            f"{iteration + 1}"
        )
        print(
            "------------------------------------------------"
        )

        locations, labels = \
            cluster_grid_results(
                current_results
            )

        unique_clusters = sorted(
            set(labels)
        )

        print(
            f"Detected "
            f"{len(unique_clusters)} "
            f"spatial configuration clusters."
        )

        if len(unique_clusters) <= 1:

            print(
                "Only one cluster remains."
            )

            return (
                current_results,
                locations,
                labels
            )

        # ----------------------------------------------------
        # Cluster centroids in 3-D joint space
        # ----------------------------------------------------

        centroids = \
            compute_cluster_centroids(
                locations,
                labels,
                current_results
            )

        for cluster, centroid in \
            centroids.items():

            print(
                f"Cluster {cluster}: "
                f"centroid = {centroid}"
            )

        # ----------------------------------------------------
        # Find boundaries
        # ----------------------------------------------------

        boundaries = \
            find_cluster_boundaries(
                grid,
                locations,
                labels
            )

        print(
            f"Boundary cells: "
            f"{len(boundaries)}"
        )

        selected_boundary_clusters = {}

        # ----------------------------------------------------
        # Compare ORIGINAL configuration of each boundary
        # cell to neighboring cluster centroids.
        # ----------------------------------------------------

        for (
            location,
            current_cluster,
            neighboring_clusters
        ) in boundaries:

            original_q = \
                original_best_q[
                    location
                ]

            selected = \
                determine_best_cluster_for_boundary(
                    location,
                    original_q,
                    neighboring_clusters,
                    centroids
                )

            if selected is not None:

                selected_boundary_clusters[
                    location
                ] = selected

        # ----------------------------------------------------
        # Check whether the assignments changed.
        # ----------------------------------------------------

        if previous_labels is not None:

            same = np.array_equal(
                labels,
                previous_labels
            )

            if same:

                print(
                    "Cluster assignment "
                    "converged."
                )

                return (
                    current_results,
                    locations,
                    labels
                )

        previous_labels = labels.copy()

        # ----------------------------------------------------
        # Re-run convolution with cluster restrictions.
        # ----------------------------------------------------

        allowed_rows = build_allowed_rows(
            data,
            grid,
            labels,
            locations,
            selected_boundary_clusters
        )

        initial_values = {
            location:
            current_results[
                location
            ]["q_norm"]
            for location in locations
        }

        current_results = \
            run_convolution(
                data,
                grid,
                q_norm,
                allowed_rows_by_cell=allowed_rows,
                initial_values=initial_values
            )

    print(
        "Maximum cluster refinement "
        "iterations reached."
    )

    locations, labels = \
        cluster_grid_results(
            current_results
        )

    return (
        current_results,
        locations,
        labels
    )


# ============================================================
# ORIGINAL BEST CONFIGURATION
# ============================================================

def get_original_best_configurations(
    data,
    grid,
    q_norm
):

    """
    For every spatial cell, keep the highest-quality original
    candidate.

    This is used only for deciding which configuration branch
    a boundary cell originally belonged to.
    """

    result = {}

    for location, indices in grid.items():

        quality = data[
            indices,
            7
        ]

        best_local = np.argmax(
            quality
        )

        index = indices[
            best_local
        ]

        result[
            location
        ] = q_norm[
            index
        ]

    return result


# ============================================================
# CREATE FINAL TABLE
# ============================================================

def create_final_table(
    data,
    grid,
    results,
    q_min,
    q_max,
    locations,
    labels
):
    """
    Create the final table.

    q1/q2/q3 are replaced by the smoothed configuration.

    The cluster assigned to each grid location is appended as
    the last column.
    """

    # Map:
    #
    #   (iz, iroll) -> cluster
    #
    cluster_by_location = {
        location: int(cluster)
        for location, cluster in zip(
            locations,
            labels
        )
    }

    final_rows = []

    for location in sorted(grid.keys()):

        indices = grid[location]

        # ----------------------------------------------------
        # Use the original highest-quality candidate as the
        # template for the remaining columns.
        # ----------------------------------------------------

        qualities = data[
            indices,
            7
        ]

        best_index = indices[
            np.argmax(qualities)
        ]

        row = data[
            best_index
        ].copy()

        # ----------------------------------------------------
        # Smoothed q1/q2/q3
        # ----------------------------------------------------

        q_norm_opt = results[
            location
        ]["q_norm"]

        q_opt = denormalize_joints(
            q_norm_opt,
            q_min,
            q_max
        )

        for local_index, joint_index in enumerate(
            VARIABLE_JOINTS
        ):

            column = 9 + joint_index

            row[column] = q_opt[
                local_index
            ]

        # ----------------------------------------------------
        # Candidate ID is no longer an original candidate.
        # ----------------------------------------------------

        row[6] = 0

        # ----------------------------------------------------
        # Append cluster number.
        # ----------------------------------------------------

        cluster = cluster_by_location[
            location
        ]

        row = np.append(
            row,
            cluster
        )

        final_rows.append(row)

    return np.asarray(final_rows)


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

    if data.shape[1] != 16:
        raise ValueError(
            f"Expected 16 columns, "
            f"got {data.shape[1]}"
        )

    header = (
        "# iy iz iroll Y Z Roll candidate quality "
        "planning_ratio q0 q1 q2 q3 q4 q5 cluster"
    )

    fmt = [
        "%d",
        "%d",
        "%d",
        "%.8f",
        "%.8f",
        "%.8f",
        "%d",
        "%.8f",
        "%.8f",
        "%.8f",
        "%.8f",
        "%.8f",
        "%.8f",
        "%.8f",
        "%.8f",
        "%d",
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
    print(
        "================================================"
    )
    print(
        "3-D JOINT-SPACE SPATIAL CONVOLUTION"
    )
    print(
        "================================================"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_table(
        INPUT_FILE
    )

    print(
        f"Loaded {len(data)} candidates."
    )

    # --------------------------------------------------------
    # Extract q1,q2,q3
    # --------------------------------------------------------

    variable_q = \
        extract_variable_joints(
            data
        )

    print(
        "Variable joint dimension:",
        variable_q.shape[1]
    )

    print(
        "Variable joints:",
        VARIABLE_JOINTS
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    q_norm, q_min, q_max = \
        normalize_joints(
            variable_q
        )

    # --------------------------------------------------------
    # Build spatial grid
    # --------------------------------------------------------

    grid = build_grid(
        data
    )

    print(
        f"Grid locations: {len(grid)}"
    )

    # --------------------------------------------------------
    # Save original best configuration at every cell.
    #
    # This is NEVER overwritten and is used later for
    # cluster-boundary decisions.
    # --------------------------------------------------------

    original_best_q = \
        get_original_best_configurations(
            data,
            grid,
            q_norm
        )

    # --------------------------------------------------------
    # FIRST CONVOLUTION
    # --------------------------------------------------------

    print()
    print(
        "================================================"
    )
    print(
        "FIRST 3x3 CONVOLUTION"
    )
    print(
        "================================================"
    )

    results = run_convolution(
        data,
        grid,
        q_norm
    )

    # --------------------------------------------------------
    # CLUSTER / BOUNDARY REFINEMENT
    # --------------------------------------------------------

    print()
    print(
        "================================================"
    )
    print(
        "CLUSTER ANALYSIS"
    )
    print(
        "================================================"
    )

    results, locations, labels = \
        refine_clusters(
            data,
            grid,
            q_norm,
            original_best_q,
            results
        )

    # --------------------------------------------------------
    # Final cluster statistics
    # --------------------------------------------------------

    print()
    print(
        "================================================"
    )
    print(
        "FINAL CLUSTERS"
    )
    print(
        "================================================"
    )

    for cluster in sorted(
        set(labels)
    ):

        count = np.sum(
            labels == cluster
        )

        print(
            f"Cluster {cluster}: "
            f"{count} grid cells"
        )

    # --------------------------------------------------------
    # Create output table
    # --------------------------------------------------------

    final_data = \
        create_final_table(
            data,
            grid,
            results,
            q_min,
            q_max,
            locations,
            labels
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_table(
        OUTPUT_FILE,
        final_data
    )

    print()
    print(
        "================================================"
    )
    print(
        "DONE"
    )
    print(
        "================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()