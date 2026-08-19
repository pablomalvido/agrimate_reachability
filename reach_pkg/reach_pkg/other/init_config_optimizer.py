#!/usr/bin/env python3

import os
import numpy as np

from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.cluster import DBSCAN


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/"
    "data/raw_configuration_table.txt"
)

OUTPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/"
    "data/final_configuration_table2.txt"
)


# ============================================================
# USER PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Variable joints
#
# In your example only q1, q2 and q3 vary.
#
# IMPORTANT:
# These are treated as ONE 3-dimensional vector:
#
#       q = [q1, q2, q3]
#
# They are NOT processed independently.
# ------------------------------------------------------------
VARIABLE_JOINTS = [1, 2, 3]


# ------------------------------------------------------------
# QUALITY WEIGHTING
#
# Higher quality -> larger contribution to KDE.
#
# Smaller temperature:
#     strongly favors the best candidates
#
# Larger temperature:
#     allows lower-quality candidates to contribute more
# ------------------------------------------------------------
QUALITY_TEMPERATURE = 0.5


# ------------------------------------------------------------
# CLUSTERING
#
# Clustering is performed in normalized N-D joint space.
#
# eps is therefore expressed in normalized joint-space units.
#
# Example:
#
#   eps = 0.25
#
# means two configurations closer than approximately 25% of
# the normalized joint-space scale can belong to the same
# configuration cluster.
# ------------------------------------------------------------
DBSCAN_EPS = 0.25

DBSCAN_MIN_SAMPLES = 5


# ------------------------------------------------------------
# JOINT-SPACE KDE BANDWIDTH
#
# This determines how broad each candidate's Gaussian is.
#
# Smaller:
#   preserves separate configurations more strongly
#
# Larger:
#   merges nearby configurations
# ------------------------------------------------------------
JOINT_SIGMA = 0.08


# ------------------------------------------------------------
# PLACEMENT-SPACE SMOOTHING
#
# Expressed in GRID CELL units.
#
# sigma = 1.0 means neighboring grid cells contribute
# significantly.
#
# Increase this if you want a smoother configuration field.
# ------------------------------------------------------------
PLACEMENT_SIGMA = 1.5


# ------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------

# Number of random optimization starts for each cluster/mode.
N_RANDOM_STARTS = 5


# ------------------------------------------------------------
# Joint bounds
#
# None -> automatically inferred from the raw data.
#
# You can explicitly specify them if desired:
#
# JOINT_BOUNDS = [
#     (-np.pi, np.pi),
#     (-np.pi, np.pi),
#     (-np.pi, np.pi),
# ]
# ------------------------------------------------------------
JOINT_BOUNDS = None


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

    # Columns:
    #iy iz iroll Y Z Roll candidate quality q0 q1 q2 q3 q4 q5
    #0 0 0 -0.75 -0.05 -0.8 0 8.06573 0.7 0 -1.04102 1.45241 0.928734 0 0

    #
    # 0  iy
    # 1  iz
    # 2  iroll
    # 3  Y
    # 4  Z
    # 5  Roll
    # 6  candidate
    # 7  quality
    # 8 planning_ratio
    # 9  q0
    # 10 q1
    # 11 q2
    # 12 q3
    # 13 q4
    # 14 q5

    return data


data = load_table(INPUT_FILE)

print("============================================================")
print("Configuration smoothing")
print("============================================================")
print(f"Input file : {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print()


# ============================================================
# EXTRACT DATA
# ============================================================

iy = data[:, 0].astype(int)
iz = data[:, 1].astype(int)
iroll = data[:, 2].astype(int)

Y = data[:, 3]
Z = data[:, 4]
Roll = data[:, 5]

candidate = data[:, 6].astype(int)
quality = data[:, 7]
planning_ratio = data[:, 8]

all_joints = data[:, 9:15]

variable_joints = all_joints[:, VARIABLE_JOINTS]

n_variable = len(VARIABLE_JOINTS)

print(f"Number of rows       : {len(data)}")
print(f"Variable joints      : {VARIABLE_JOINTS}")
print(f"Configuration dim.   : {n_variable}")
print()


# ============================================================
# IDENTIFY PLACEMENT GRID
# ============================================================

iz_values = np.sort(np.unique(iz))
iroll_values = np.sort(np.unique(iroll))

nz = len(iz_values)
nr = len(iroll_values)

iz_to_grid = {
    value: index
    for index, value in enumerate(iz_values)
}

iroll_to_grid = {
    value: index
    for index, value in enumerate(iroll_values)
}


print(f"Placement grid       : {nz} x {nr}")
print(f"Number of placements : {nz * nr}")
print()


# ============================================================
# CREATE PLACEMENT ARRAYS
# ============================================================

placement_Z = np.zeros((nz, nr))
placement_Roll = np.zeros((nz, nr))
placement_Y = np.zeros((nz, nr))


# Store the candidate indices belonging to each placement.

placement_rows = [
    [
        []
        for _ in range(nr)
    ]
    for _ in range(nz)
]


for row in range(len(data)):

    gz = iz_to_grid[iz[row]]
    gr = iroll_to_grid[iroll[row]]

    placement_rows[gz][gr].append(row)

    placement_Z[gz, gr] = Z[row]
    placement_Roll[gz, gr] = Roll[row]
    placement_Y[gz, gr] = Y[row]


# ============================================================
# NORMALIZE JOINT SPACE
# ============================================================

# The clustering and KDE operate in normalized joint space.
#
# This prevents a joint with a larger numerical range from
# dominating the distance calculations.

joint_min = np.min(
    variable_joints,
    axis=0
)

joint_max = np.max(
    variable_joints,
    axis=0
)

joint_range = joint_max - joint_min

# Avoid division by zero in case a supposedly variable joint
# actually has no variation.
joint_range[joint_range < 1e-12] = 1.0


def normalize_joint(q):

    return (
        (q - joint_min)
        / joint_range
    )


def denormalize_joint(q_normalized):

    return (
        q_normalized * joint_range
        + joint_min
    )


normalized_joints = normalize_joint(
    variable_joints
)


print("Joint-space ranges:")
for i, joint_id in enumerate(VARIABLE_JOINTS):

    print(
        f"  q{joint_id}: "
        f"[{joint_min[i]:.6f}, "
        f"{joint_max[i]:.6f}]"
    )

print()


# ============================================================
# JOINT BOUNDS
# ============================================================

if JOINT_BOUNDS is None:

    # Give the optimizer a small amount of freedom outside
    # the observed data.

    margin = 0.05 * joint_range

    joint_bounds = [
        (
            joint_min[i] - margin[i],
            joint_max[i] + margin[i]
        )
        for i in range(n_variable)
    ]

else:

    joint_bounds = JOINT_BOUNDS


normalized_bounds = [
    (
        (low - joint_min[i]) / joint_range[i],
        (high - joint_min[i]) / joint_range[i]
    )
    for i, (low, high) in enumerate(joint_bounds)
]


# ============================================================
# QUALITY WEIGHTS
# ============================================================

# At each placement, convert the candidate qualities to
# normalized weights.
#
# IMPORTANT:
#
# The weights belong to COMPLETE candidate vectors.
#
# We never calculate independent weights for q1, q2, q3.

candidate_weights = np.zeros(
    len(data)
)


for gz in range(nz):

    for gr in range(nr):

        rows = placement_rows[gz][gr]

        if len(rows) == 0:
            continue

        local_quality = quality[rows]

        local_weights = softmax(
            local_quality / QUALITY_TEMPERATURE
        )

        candidate_weights[rows] = local_weights


# ============================================================
# STEP 1:
# GLOBAL CONFIGURATION-SPACE CLUSTERING
# ============================================================

print("============================================================")
print("Clustering configuration space")
print("============================================================")

clusterer = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES
)

cluster_labels = clusterer.fit_predict(
    normalized_joints
)

unique_labels = np.unique(
    cluster_labels
)

# DBSCAN uses -1 for noise.

real_labels = [
    label
    for label in unique_labels
    if label != -1
]

print(
    f"Detected configuration clusters: "
    f"{len(real_labels)}"
)

print(
    f"Noise candidates: "
    f"{np.sum(cluster_labels == -1)}"
)

for label in real_labels:

    count = np.sum(
        cluster_labels == label
    )

    print(
        f"  Cluster {label}: "
        f"{count} candidates"
    )

print()


# ============================================================
# HANDLE DBSCAN NOISE
# ============================================================

# Noise points should not simply disappear.
#
# We assign each noise candidate to the closest existing
# cluster if it is reasonably close.
#
# Otherwise it gets its own cluster.
#
# This keeps unusual but potentially valid configurations.

if len(real_labels) > 0:

    cluster_centers = {}

    for label in real_labels:

        cluster_centers[label] = np.mean(
            normalized_joints[
                cluster_labels == label
            ],
            axis=0
        )


    next_label = (
        max(real_labels) + 1
    )

    for row in range(len(data)):

        if cluster_labels[row] != -1:
            continue

        q = normalized_joints[row]

        distances = {
            label: np.linalg.norm(
                q - center
            )
            for label, center
            in cluster_centers.items()
        }

        nearest_label = min(
            distances,
            key=distances.get
        )

        nearest_distance = distances[
            nearest_label
        ]

        if nearest_distance <= 2.0 * DBSCAN_EPS:

            cluster_labels[row] = (
                nearest_label
            )

        else:

            cluster_labels[row] = next_label

            cluster_centers[next_label] = q

            next_label += 1


else:

    # No clusters found by DBSCAN.
    # Treat every candidate as its own cluster.

    cluster_labels = np.arange(
        len(data)
    )


cluster_ids = np.sort(
    np.unique(cluster_labels)
)

print(
    f"Final number of configuration modes: "
    f"{len(cluster_ids)}"
)

for label in cluster_ids:

    count = np.sum(
        cluster_labels == label
    )

    print(
        f"  Mode {label}: "
        f"{count} candidates"
    )

print()


# ============================================================
# SPATIAL KERNEL
# ============================================================

def spatial_weight(
    target_z,
    target_roll,
    source_z,
    source_roll
):

    dz = target_z - source_z
    dr = target_roll - source_roll

    distance_squared = (
        dz * dz +
        dr * dr
    )

    return np.exp(
        -0.5
        * distance_squared
        / PLACEMENT_SIGMA**2
    )


# ============================================================
# N-D GAUSSIAN KERNEL
# ============================================================

def joint_kernel(
    q,
    samples,
    sigma
):

    """
    Multivariate isotropic Gaussian kernel.

    q:
        shape (D,)

    samples:
        shape (N,D)

    The COMPLETE D-dimensional vector is used.
    """

    diff = samples - q

    squared_distance = np.sum(
        diff * diff,
        axis=1
    )

    return np.exp(
        -0.5
        * squared_distance
        / sigma**2
    )


# ============================================================
# PREPARE CLUSTER DATA
# ============================================================

cluster_data = {}

for label in cluster_ids:

    rows = np.where(
        cluster_labels == label
    )[0]

    cluster_data[label] = {
        "rows": rows,
        "q": normalized_joints[rows],
        "quality_weights": candidate_weights[rows],
        "grid_z": iz[rows],
        "grid_roll": iroll[rows],
    }


# ============================================================
# SURROGATE FOR ONE CONFIGURATION MODE
# ============================================================

def evaluate_mode_surrogate(
    q,
    target_gz,
    target_gr,
    label
):

    """
    Evaluate the smoothed KDE for one configuration mode.

    The estimator is:

        S_k(q | x)
        =
        sum_i
            K_x(x, x_i)
            sum_j
                w_ij
                K_q(q, q_ij)

    where j is restricted to configuration mode k.

    q is the COMPLETE N-D configuration vector.
    """

    mode = cluster_data[label]

    rows = mode["rows"]

    if len(rows) == 0:
        return 0.0

    target_z = target_gz
    target_roll = target_gr

    value = 0.0

    for local_index, row in enumerate(rows):

        source_gz = iz_to_grid[
            iz[row]
        ]

        source_gr = iroll_to_grid[
            iroll[row]
        ]

        # Spatial smoothing.
        ws = spatial_weight(
            target_z,
            target_roll,
            source_gz,
            source_gr
        )

        # Candidate quality weight.
        wq = candidate_weights[row]

        # N-dimensional configuration-space Gaussian.
        wk = joint_kernel(
            q,
            normalized_joints[row:row + 1],
            JOINT_SIGMA
        )[0]

        value += (
            ws
            * wq
            * wk
        )

    return value


# ============================================================
# FIND MODE OF ONE CONFIGURATION CLUSTER
# ============================================================

def optimize_mode(
    target_gz,
    target_gr,
    label
):

    """
    Find the maximum of the smoothed N-D KDE for one mode.

        q* = argmax S_k(q | x)
    """

    mode = cluster_data[label]

    rows = mode["rows"]

    if len(rows) == 0:
        return None, -np.inf


    # --------------------------------------------------------
    # Only optimize modes that have meaningful support near
    # the target placement.
    # --------------------------------------------------------

    starts = []

    spatial_support = []

    for row in rows:

        source_gz = iz_to_grid[
            iz[row]
        ]

        source_gr = iroll_to_grid[
            iroll[row]
        ]

        ws = spatial_weight(
            target_gz,
            target_gr,
            source_gz,
            source_gr
        )

        spatial_support.append(
            ws
        )

    spatial_support = np.asarray(
        spatial_support
    )

    # If this mode has essentially no support around the
    # target location, don't optimize it.

    if np.max(spatial_support) < 1e-4:

        return None, -np.inf


    # --------------------------------------------------------
    # Good starting points:
    # candidate configurations near this placement.
    # --------------------------------------------------------

    order = np.argsort(
        -spatial_support
    )

    for index in order[:10]:

        starts.append(
            normalized_joints[
                rows[index]
            ].copy()
        )


    # --------------------------------------------------------
    # Add random starts.
    # --------------------------------------------------------

    for _ in range(
        N_RANDOM_STARTS
    ):

        starts.append(
            np.array([
                np.random.uniform(
                    low,
                    high
                )
                for low, high
                in normalized_bounds
            ])
        )


    # --------------------------------------------------------
    # Optimize negative surrogate because scipy minimizes.
    # --------------------------------------------------------

    def objective(q):

        return -evaluate_mode_surrogate(
            q,
            target_gz,
            target_gr,
            label
        )


    best_q = None
    best_value = -np.inf


    for q0 in starts:

        result = minimize(
            objective,
            q0,
            method="L-BFGS-B",
            bounds=normalized_bounds,
            options={
                "maxiter": 300,
                "ftol": 1e-10,
                "gtol": 1e-8
            }
        )

        if not np.all(
            np.isfinite(result.x)
        ):
            continue

        value = -result.fun

        if value > best_value:

            best_value = value
            best_q = result.x.copy()


    if best_q is None:

        return None, -np.inf


    return (
        denormalize_joint(best_q),
        best_value
    )


# ============================================================
# OPTIMIZE EVERY PLACEMENT
# ============================================================

optimized_variable_joints = np.zeros(
    (nz, nr, n_variable)
)

optimized_score = np.zeros(
    (nz, nr)
)

optimized_mode = np.zeros(
    (nz, nr),
    dtype=int
)


print("============================================================")
print("Optimizing configuration field")
print("============================================================")
print()


for gz in range(nz):

    for gr in range(nr):

        best_q = None
        best_score = -np.inf
        best_label = -1


        # ----------------------------------------------------
        # Optimize every configuration mode independently.
        # ----------------------------------------------------

        for label in cluster_ids:

            q_opt, score = optimize_mode(
                gz,
                gr,
                label
            )

            if q_opt is None:
                continue

            if score > best_score:

                best_score = score
                best_q = q_opt
                best_label = label


        if best_q is None:

            print(
                f"WARNING: no solution at "
                f"iz={iz_values[gz]}, "
                f"iroll={iroll_values[gr]}"
            )

            continue


        optimized_variable_joints[
            gz, gr
        ] = best_q

        optimized_score[
            gz, gr
        ] = best_score

        optimized_mode[
            gz, gr
        ] = best_label


        print(
            f"iz={iz_values[gz]:3d} "
            f"iroll={iroll_values[gr]:3d} "
            f"mode={best_label:3d} "
            f"q={best_q} "
            f"score={best_score:.6f}"
        )


# ============================================================
# RECONSTRUCT COMPLETE 6-DOF JOINT VECTOR
# ============================================================

# Start with the first candidate's joint values at every
# placement. This preserves the joints that do not vary.

optimized_all_joints = np.zeros(
    (nz, nr, 6)
)


for gz in range(nz):

    for gr in range(nr):

        rows = placement_rows[gz][gr]

        if len(rows) == 0:
            continue

        # Use the first local candidate as the source for
        # non-variable joints.
        reference_row = rows[0]

        optimized_all_joints[
            gz, gr
        ] = all_joints[reference_row]


        # Replace only the optimized variable joints.
        optimized_all_joints[
            gz, gr, VARIABLE_JOINTS
        ] = optimized_variable_joints[
            gz, gr
        ]


# ============================================================
# WRITE OUTPUT TABLE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w"
) as f:

    f.write(
        "# iy iz iroll Y Z Roll candidate "
        "quality q0 q1 q2 q3 q4 q5 mode\n"
    )

    for gz in range(nz):

        for gr in range(nr):

            q = optimized_all_joints[
                gz, gr
            ]

            f.write(
                f"{int(iy[placement_rows[gz][gr][0]])} "
                f"{iz_values[gz]} "
                f"{iroll_values[gr]} "
                f"{placement_Y[gz, gr]:.8f} "
                f"{placement_Z[gz, gr]:.8f} "
                f"{placement_Roll[gz, gr]:.8f} "
                f"-1 "
                f"{optimized_score[gz, gr]:.8f} "
                f"{q[0]:.8f} "
                f"{q[1]:.8f} "
                f"{q[2]:.8f} "
                f"{q[3]:.8f} "
                f"{q[4]:.8f} "
                f"{q[5]:.8f} "
                f"{optimized_mode[gz, gr]}\n"
            )


print()
print("============================================================")
print("DONE")
print("============================================================")
print()
print(f"Saved optimized table to:")
print(OUTPUT_FILE)