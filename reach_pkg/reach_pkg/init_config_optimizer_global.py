import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.cluster import DBSCAN


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/"
    "raw_configuration_table.txt"
)

OUTPUT_FILE = (
    "/home/rosdev/ros2_ws/src/moveit_cpp_demo/data/"
    "final_configuration_table_global.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Joints that are actually changing / being optimized.
# The other joints are copied from the selected original candidate.
VARIABLE_JOINTS = [1, 2, 3]

# ------------------------------------------------------------
# Quality weighting
# ------------------------------------------------------------

# Smaller -> strongly favors high-quality candidates.
# Larger -> makes candidates more equally important.
QUALITY_TEMPERATURE = 0.5


# ------------------------------------------------------------
# Configuration-space KDE
# ------------------------------------------------------------

# Gaussian width in normalized joint space.
#
# Smaller:
#   only very similar configurations influence each other.
#
# Larger:
#   broader configuration families.
#
JOINT_SIGMA = 0.08


# ------------------------------------------------------------
# Spatial smoothness
# ------------------------------------------------------------

# Weight of the smoothness term.
#
# Larger:
#   smoother configuration field.
#
# Smaller:
#   follows the quality surrogate more closely.
#
SMOOTHNESS_LAMBDA = 5.0


# ------------------------------------------------------------
# Mode clustering
# ------------------------------------------------------------

# DBSCAN operates on COMPLETE n-D joint vectors.
#
# Since the joint vectors are normalized to [0,1],
# this is measured in normalized configuration space.
#
DBSCAN_EPS = 0.20
DBSCAN_MIN_SAMPLES = 5


# ------------------------------------------------------------
# Global optimizer
# ------------------------------------------------------------

OPTIMIZER_MAXITER = 500
OPTIMIZER_FTOL = 1e-8
OPTIMIZER_GTOL = 1e-6


# ============================================================
# DATA LOADING
# ============================================================

def load_table(filename):
    """
    Load the configuration table.

    Expected columns:

        iy iz iroll Y Z Roll candidate quality planning_ratio q0 q1 q2 q3 q4 q5
    """

    data = np.loadtxt(filename, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return data


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def softmax_quality(quality, temperature):
    """
    Convert quality values into normalized positive weights.
    """

    x = quality / temperature

    # Numerical stability
    x = x - np.max(x)

    w = np.exp(x)

    return w / np.sum(w)


def normalize_configurations(q):
    """
    Normalize every joint independently to [0,1].

    q:
        shape = (N, n_joints)
    """

    q_min = np.min(q, axis=0)
    q_max = np.max(q, axis=0)

    ranges = q_max - q_min

    # Avoid division by zero for constant joints.
    ranges[ranges < 1e-12] = 1.0

    q_norm = (q - q_min) / ranges

    return q_norm, q_min, q_max


def denormalize_configuration(q_norm, q_min, q_max):
    """
    Convert normalized joint vector back to original units.
    """

    return q_norm * (q_max - q_min) + q_min


# ============================================================
# SPATIAL GRID
# ============================================================

def get_placement_key(row):
    """
    Grid identity.

    We use iz and iroll because iy is constant.
    """

    iz = int(round(row[1]))
    iroll = int(round(row[2]))

    return iz, iroll


def build_placement_data(data):
    """
    Organize rows by grid location.
    """

    placements = {}

    for row in data:
        key = get_placement_key(row)

        if key not in placements:
            placements[key] = []

        placements[key].append(row)

    return placements


# ============================================================
# GLOBAL MODE IDENTIFICATION
# ============================================================

def identify_modes(data, q_norm):
    """
    Cluster COMPLETE joint vectors globally.

    Returns:
        labels for every row in the input table.
    """

    print()
    print("================================================")
    print("IDENTIFYING CONFIGURATION MODES")
    print("================================================")

    clusterer = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    labels = clusterer.fit_predict(q_norm)

    unique_labels = sorted(set(labels))

    print()

    for label in unique_labels:

        count = np.sum(labels == label)

        if label == -1:
            print(
                f"Noise candidates: {count}"
            )
        else:
            print(
                f"Mode {label}: {count} candidates"
            )

    print()

    return labels


# ============================================================
# MODE / PLACEMENT DATA
# ============================================================

def build_mode_data(data, q_norm, labels, mode):
    """
    Extract candidates belonging to one mode and organize
    them by placement.
    """

    indices = np.where(labels == mode)[0]

    mode_data = {}

    for idx in indices:

        key = get_placement_key(data[idx])

        if key not in mode_data:
            mode_data[key] = []

        mode_data[key].append(idx)

    return mode_data


# ============================================================
# LOCAL SURROGATE
# ============================================================

def compute_local_log_surrogate(
    q,
    candidate_q,
    candidate_weights,
    sigma
):
    """
    Log of a weighted N-D Gaussian KDE.

    q:
        shape (n_joints,)

    candidate_q:
        shape (N, n_joints)

    candidate_weights:
        shape (N,)
    """

    if len(candidate_q) == 0:
        return -np.inf

    diff = candidate_q - q

    squared_distance = np.sum(diff * diff, axis=1)

    log_kernel = (
        -0.5 * squared_distance / (sigma * sigma)
    )

    log_weights = np.log(
        np.maximum(candidate_weights, 1e-300)
    )

    return logsumexp(
        log_weights + log_kernel
    )


# ============================================================
# GLOBAL MODE OPTIMIZATION
# ============================================================

def optimize_mode(
    mode,
    mode_data,
    data,
    q_norm_all,
    quality_weights_all,
    q_min,
    q_max
):
    """
    Globally optimize one configuration mode.

    IMPORTANT:

    Every placement in this mode has one COMPLETE
    n-dimensional optimization variable.

    The optimizer therefore solves:

        q_1, q_2, ..., q_N

    simultaneously.
    """

    print()
    print("================================================")
    print(f"OPTIMIZING MODE {mode}")
    print("================================================")

    locations = sorted(mode_data.keys())

    n_locations = len(locations)
    n_joints = len(VARIABLE_JOINTS)

    print(
        f"Locations in mode: {n_locations}"
    )

    print(
        f"Optimization variables: "
        f"{n_locations} x {n_joints} = "
        f"{n_locations * n_joints}"
    )

    # --------------------------------------------------------
    # Map placement -> optimization variable index
    # --------------------------------------------------------

    location_to_index = {
        location: i
        for i, location in enumerate(locations)
    }

    # --------------------------------------------------------
    # Candidate information for every location
    # --------------------------------------------------------

    local_candidates = {}

    for location in locations:

        indices = mode_data[location]

        candidate_q = q_norm_all[indices]

        candidate_weights = quality_weights_all[indices]

        # Renormalize weights within this mode/location.
        weight_sum = np.sum(candidate_weights)

        if weight_sum > 0:
            candidate_weights = (
                candidate_weights / weight_sum
            )

        local_candidates[location] = (
            candidate_q,
            candidate_weights,
            indices
        )

    # --------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------

    #
    # For every location choose the highest-quality candidate
    # belonging to this mode.
    #

    x0 = np.zeros(
        n_locations * n_joints
    )

    for location, i in location_to_index.items():

        candidate_q, candidate_weights, _ = \
            local_candidates[location]

        best = np.argmax(candidate_weights)

        x0[
            i * n_joints:
            (i + 1) * n_joints
        ] = candidate_q[best]

    # --------------------------------------------------------
    # Build neighboring edges
    # --------------------------------------------------------

    edges = []

    for location, i in location_to_index.items():

        iz, iroll = location

        # 4-connected grid
        neighbors = [
            (iz + 1, iroll),
            (iz - 1, iroll),
            (iz, iroll + 1),
            (iz, iroll - 1),
        ]

        for neighbor in neighbors:

            if neighbor not in location_to_index:
                continue

            j = location_to_index[neighbor]

            # Avoid adding the same edge twice.
            if i < j:
                edges.append((i, j))

    print(
        f"Smoothness edges: {len(edges)}"
    )

    # --------------------------------------------------------
    # Objective function
    # --------------------------------------------------------

    evaluation_counter = [0]

    def objective(x):

        evaluation_counter[0] += 1

        # Reshape:
        #
        # x =
        # [
        #   q_location_0,
        #   q_location_1,
        #   ...
        # ]
        #
        Q = x.reshape(
            n_locations,
            n_joints
        )

        # ====================================================
        # DATA / SURROGATE TERM
        # ====================================================

        data_term = 0.0

        for location, i in location_to_index.items():

            candidate_q, candidate_weights, _ = \
                local_candidates[location]

            q = Q[i]

            log_p = compute_local_log_surrogate(
                q,
                candidate_q,
                candidate_weights,
                JOINT_SIGMA
            )

            data_term += log_p

        # ====================================================
        # SMOOTHNESS TERM
        # ====================================================

        smoothness = 0.0

        for i, j in edges:

            difference = Q[i] - Q[j]

            smoothness += np.sum(
                difference * difference
            )

        # We MAXIMIZE:
        #
        #     data_term - lambda * smoothness
        #
        # scipy minimizes, therefore return negative.

        objective_value = (
            data_term
            -
            SMOOTHNESS_LAMBDA * smoothness
        )

        return -objective_value

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    #
    # Since configurations are normalized:
    #
    #       0 <= q <= 1
    #

    bounds = [
        (0.0, 1.0)
        for _ in range(
            n_locations * n_joints
        )
    ]

    # --------------------------------------------------------
    # Run global optimization
    # --------------------------------------------------------

    print()
    print("Starting global optimization...")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": OPTIMIZER_MAXITER,
            "ftol": OPTIMIZER_FTOL,
            "gtol": OPTIMIZER_GTOL,
            "maxls": 50,
        }
    )

    print()
    print("Optimization finished.")

    print(
        f"Success: {result.success}"
    )

    print(
        f"Message: {result.message}"
    )

    print(
        f"Iterations: {result.nit}"
    )

    print(
        f"Function evaluations: "
        f"{result.nfev}"
    )

    # --------------------------------------------------------
    # Extract optimized configurations
    # --------------------------------------------------------

    Q_opt_norm = result.x.reshape(
        n_locations,
        n_joints
    )

    # Convert back to actual joint units.
    Q_opt = np.zeros_like(Q_opt_norm)

    variable_min = q_min[VARIABLE_JOINTS]
    variable_max = q_max[VARIABLE_JOINTS]

    Q_opt = (
        Q_opt_norm
        *
        (variable_max - variable_min)
        +
        variable_min
    )

    # --------------------------------------------------------
    # Calculate final mode strengths
    # --------------------------------------------------------

    strengths = {}

    for location, i in location_to_index.items():

        candidate_q, candidate_weights, _ = \
            local_candidates[location]

        log_p = compute_local_log_surrogate(
            Q_opt_norm[i],
            candidate_q,
            candidate_weights,
            JOINT_SIGMA
        )

        strengths[location] = log_p

    return {
        "locations": locations,
        "location_to_index": location_to_index,
        "Q_norm": Q_opt_norm,
        "Q": Q_opt,
        "strengths": strengths,
        "result": result,
        "mode_data": mode_data,
    }


# ============================================================
# FINAL MODE SELECTION
# ============================================================

def select_final_configurations(
    data,
    mode_results,
    q_min,
    q_max
):
    """
    Select the strongest optimized mode at every placement.
    """

    placements = build_placement_data(data)

    final_rows = []

    for location in sorted(placements.keys()):

        available_modes = []

        for mode, result in mode_results.items():

            if location not in result["location_to_index"]:
                continue

            i = result["location_to_index"][location]

            strength = result["strengths"][location]

            q = result["Q"][i]

            available_modes.append(
                (
                    strength,
                    mode,
                    q
                )
            )

        if len(available_modes) == 0:
            continue

        # Highest surrogate strength
        available_modes.sort(
            key=lambda x: x[0],
            reverse=True
        )

        best_strength, best_mode, best_q = \
            available_modes[0]

        # ----------------------------------------------------
        # Construct output row
        # ----------------------------------------------------

        original_rows = placements[location]

        # Use the original row with the best quality
        # as the template.
        template = max(
            original_rows,
            key=lambda row: row[7]
        ).copy()

        # Replace optimized variable joints.
        for local_index, joint_index in enumerate(
            VARIABLE_JOINTS
        ):
            template[9 + joint_index] = \
                best_q[local_index]

        #
        # Store the mode in the candidate column.
        #
        # This is useful for seeing which branch was selected.
        #
        template[6] = best_mode

        #
        # Store surrogate strength in the quality column.
        #
        template[7] = best_strength

        final_rows.append(template)

    return np.array(final_rows)


# ============================================================
# SAVE
# ============================================================

def save_table(filename, data):
    """
    Save final table.
    """

    header = (
        "iy iz iroll Y Z Roll candidate quality planning_ratio"
        "q0 q1 q2 q3 q4 q5"
    )

    np.savetxt(
        filename,
        data,
        fmt=[
            "%d",   # iy
            "%d",   # iz
            "%d",   # iroll
            "%.8f", # Y
            "%.8f", # Z
            "%.8f", # Roll
            "%d",   # candidate / mode
            "%.8f", # quality / surrogate
            "%.8f", # planning_ratio
            "%.8f", # q0
            "%.8f", # q1
            "%.8f", # q2
            "%.8f", # q3
            "%.8f", # q4
            "%.8f", # q5
        ],
        header=header
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("================================================")
    print("GLOBAL CONFIGURATION FIELD OPTIMIZER")
    print("================================================")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_table(INPUT_FILE)

    print(
        f"Loaded {len(data)} candidate rows."
    )

    # --------------------------------------------------------
    # Extract all joint configurations
    # --------------------------------------------------------

    all_joints = data[:, 9:15]


    variable_q = all_joints[:, VARIABLE_JOINTS]

    print(
        f"Optimized joints: {VARIABLE_JOINTS}"
    )

    print(
        f"Configuration dimensionality: "
        f"{len(VARIABLE_JOINTS)}"
    )

    # --------------------------------------------------------
    # Normalize configuration space
    # --------------------------------------------------------

    q_norm, q_min_var, q_max_var = \
        normalize_configurations(variable_q)

    #
    # Put min/max into arrays corresponding to all 6 joints.
    #

    q_min = np.min(all_joints, axis=0)
    q_max = np.max(all_joints, axis=0)

    # But for the optimized variables use the values
    # calculated above.
    q_min[VARIABLE_JOINTS] = q_min_var
    q_max[VARIABLE_JOINTS] = q_max_var

    # --------------------------------------------------------
    # Quality weights
    # --------------------------------------------------------

    #
    # Compute quality weights separately at each placement.
    #

    placements = build_placement_data(data)

    quality_weights = np.zeros(
        len(data)
    )

    for location, rows in placements.items():

        indices = []

        for i, row in enumerate(data):

            if get_placement_key(row) == location:
                indices.append(i)

        indices = np.array(
            indices,
            dtype=int
        )

        quality = data[indices, 7]

        quality_weights[indices] = \
            softmax_quality(
                quality,
                QUALITY_TEMPERATURE
            )

    # --------------------------------------------------------
    # Identify global configuration modes
    # --------------------------------------------------------

    labels = identify_modes(
        data,
        q_norm
    )

    # Ignore DBSCAN noise
    modes = sorted(
        label
        for label in set(labels)
        if label != -1
    )

    print(
        f"Detected {len(modes)} configuration modes."
    )

    # --------------------------------------------------------
    # Optimize every mode independently
    # --------------------------------------------------------

    mode_results = {}

    for mode in modes:

        mode_data = build_mode_data(
            data,
            q_norm,
            labels,
            mode
        )

        #
        # A mode needs to exist at more than one location
        # to have meaningful spatial smoothness.
        #

        if len(mode_data) == 0:
            continue

        result = optimize_mode(
            mode,
            mode_data,
            data,
            q_norm,
            quality_weights,
            q_min,
            q_max
        )

        mode_results[mode] = result

    # --------------------------------------------------------
    # Select final configuration at every grid point
    # --------------------------------------------------------

    print()
    print("================================================")
    print("SELECTING FINAL CONFIGURATIONS")
    print("================================================")

    final_data = select_final_configurations(
        data,
        mode_results,
        q_min,
        q_max
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
        f"Saved {len(final_data)} optimized "
        f"grid points."
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()