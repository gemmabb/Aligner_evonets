import numpy as np
from collections import OrderedDict, defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#our imports
from aligner.montecarlo import helper as helper_montecarlo
from aligner.poissonAlignment import helper as helper_poisson

from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def latent_assignment_seqthreshold(ensemble):
    """
    latent_assignment_seqthreshold:
    Generates the latent assignment vector. For each permutation of the ensemble, it creates a vector with the most frequently assigned label for each node position across snapshots. It performs the assignation sequentially, by changing the threshold of the frequency. The resulting array contains all label nodes.

    @param ensemble: ensemble of permutations (sampled at equilibrium)

    @return: latent assignment vector (one for each perm in the ensemble)
    """
    T = len(ensemble[0])
    N = len(ensemble[0][0])
    latent_assignments = []  # stores one latent vector per sampled alignment
    
    for alignment in ensemble:
        latent = [None] * N
        used_labels = set()
    
        # Sequential assignment by decreasing threshold
        for threshold in range(T, 0, -1):
            for n in range(N):
                if latent[n] is not None:
                    continue  # already assigned
                
                this_column = [alignment[t][n] for t in range(T)]
                values, counts = np.unique(this_column, return_counts=True)
    
                # candidate labels with enough frequency and not used
                candidates = [val for val, count in zip(values, counts)
                              if count >= threshold and val not in used_labels]
    
                if candidates:
                    # choose the most frequent available label
                    chosen_value = candidates[np.argmax([counts[np.where(values == val)[0][0]] for val in candidates])]
                    latent[n] = chosen_value
                    used_labels.add(chosen_value)
    
        # Fill unassigned nodes with remaining unused labels
        missing_labels = [val for val in range(N) if val not in used_labels]
        for i in range(N):
            if latent[i] is None:
                latent[i] = missing_labels.pop(0)
                used_labels.add(latent[i])
    
        # Save final latent assignment for this sample
        latent_assignments.append(latent)
    
    latent_assignments = np.array(latent_assignments)
    return latent_assignments
    

def get_list_assigned_labels(ensemble, latent_assignment_vector):
    """
    get_list_assigned_labels:
    Collects the latent assigned vectors for each pair of (n,t) across samples.

    @param ensemble: ensemble of permutations (sampled at equilibrium)
    @param latent_assignment_vector: latent assignment vector (one for each sample)

    @return tn_latent_assignment: for each (t,n,): collection of latent assignments across samples
    """
    tn_latent_assignment = []
    S,T,N = ensemble.shape

    for t in range(T):
        dict_per_node = defaultdict(list)
        for n in range(N):
            assigned_labels = []
            for sample, out_labels in zip(ensemble, latent_assignment_vector):
                pos_n = [i for i, x in enumerate(sample[t]) if x == n]
                if len(pos_n) == 1:
                    assigned = out_labels[pos_n[0]]
                    assigned_labels.append(assigned)
            dict_per_node[n] = assigned_labels
        tn_latent_assignment.append(dict_per_node)
    return tn_latent_assignment


def get_binary_mapping(tn_latent_assignment, T, N, S):
    """
    Builds the binary mapping matrix (correct/incorrect). For each pair of (t,n) we look at the assigned label across samples. If latent==n in the majority of the samples (>= S/2), we consider a correct mapping.

    @param tn_latent_assignment: for each (t,n,): collection of latent assignments across samples
    @param T: #snapshots
    @param N: #nodes
    @param S: #samples in the ensemble

    @return binary_mapping (correct/incorrect)
    """
    binary_mapping = np.zeros((T,N))
    for t in range(T):
        for node, labels in tn_latent_assignment[t].items():
            outlabels_forthisnode, freq_outlabels_forthisnode = np.unique(labels, return_counts=True)
            if node in outlabels_forthisnode:
                freq_correct = freq_outlabels_forthisnode[np.where(outlabels_forthisnode==node)[0]]
                if freq_correct >= S/2:
                    binary_mapping[t,node] = 1.
                    
    return binary_mapping

def infer_node_mapping(ensemble):
    """
    From the ensemble of sampled alignments, infer the node mapping (average of ensemble) and compare to the ground truth.

    @param ensemble: ensemble of permutations

    @return binary_node_mapping (correct/incorrect wrt ground truth)
    """
    S, T, N = ensemble.shape
    lat_assignment_vector = latent_assignment_seqthreshold(ensemble)
    tn_assigned_labels = get_list_assigned_labels(ensemble, lat_assignment_vector)
    binary_mapping = get_binary_mapping(tn_assigned_labels, T, N, S)

    return binary_mapping

def mark_empty_nodes(ensemble_node_mapping, A):
    """
    mark_empty_nodes:
    Modifies the binary node mapping so as to highlight nodes that are non-alignable (i.e., nodes that are disconnected in a given snapshot).

    @param ensemble_node_mapping: binary node mapping wrt ground truth
    @param A: adj matrices of the snapshots that we have aligned

    @returns notempty_ensemble_node_mapping (with 1 if correct, 0 if incorrect, and 0.5 if non-alignable)
    """
    T, N = ensemble_node_mapping.shape
    empty_nodes = np.zeros((T, N))
    
    for t in range(T):
        for n in range(N):
            num_neighbours = np.sum(A[t][n,:]>0) + np.sum(A[t][:,n]>0)
            if num_neighbours == 0:
                empty_nodes[t][n] = 0.5
    
    notempty_ensemble_node_mapping = np.absolute(ensemble_node_mapping-empty_nodes)

    return notempty_ensemble_node_mapping

def order_node_mapping(ensemble_node_mapping, A, groups=None):
    """
    mark_empty_nodes:
    Modifies the binary node mapping so as the order of the nodes is based on the order of the nodes in descending strength in the densest snapshot.
    @param ensemble_node_mapping: binary node mapping wrt ground truth
    @param A: adj matrices of the snapshots that we have aligned
    @param groups: node membership (same order as they appear in A) to preserve group structure (if applicable)

    @returns ordered_ensemble_node_mapping
    """
    T, N = ensemble_node_mapping.shape

    #compute permutations ordered by strength (inside montecarlo pckg)
    if groups is not None:
        ordered_P = helper_montecarlo.initial_permutation_groups(A, groups)
    else:
        ordered_P = helper_montecarlo.initial_permutation(A)

    #take densest network and use that permutation:
    t_densest = np.argmax(helper_poisson.get_alpha(A, helper_poisson.get_ST(A)))
    ordered_ensemble_node_mapping = np.array([ensemble_node_mapping[t][ordered_P[t_densest]] for t in range(T)]) 

    return ordered_ensemble_node_mapping

def get_alignment_groundState(groundState):
    """
    get_alignment_groundState:
    Computes the binary node mapping corresponding to the given ground state permutation. 
    This matrix indicates which nodes are correctly aligned across snapshots according to 
    the latent assignment sequence derived from the ground state.

    @param groundState: array (T x N)
        Ground state permutation array, where each row corresponds to the node order at a 
        specific snapshot (T = number of snapshots, N = number of nodes).

    @returns binary_mapping_gs: array (T x N)
        Binary mapping matrix, where binary_mapping_gs[t, n] = 1 if node n at snapshot t 
        is aligned with its latent position, and 0 otherwise.
    """
    T, N = groundState.shape
    latent = helper_analysis.latent_assignment_seqthreshold([groundState])[0]  # not ensemble but single permutation
    binary_mapping_gs = np.zeros((T, N))
    for t in range(T):
        for n in range(N):
            ix_n = np.where(latent == n)[0]  # order nodes by latent structure
            binary_mapping_gs[t][n] = 1 if groundState[t][ix_n] == latent[ix_n] else 0
    return binary_mapping_gs

def compute_accuracy(binary_node_mapping):
    """
    compute_accuracy:
    Computes the accuracy of the alignment, mapping wrt ground truth. We discard non-alignable nodes for the calculation.

    @param binary_node_mapping: matrix with 0 (inc), 1 (corr), 0.5 (non-alignable)

    @return accuracy
    """
    T, N = binary_node_mapping.shape
    non_alignable = np.sum(binary_node_mapping == 0.5)
    correct = np.sum(binary_node_mapping == 1)
    accuracy = correct/(T*N - non_alignable)

    return accuracy

def infer_blueprint(A, ensemble):
    """
    get_sampled_L:
    Computes the average latent interaction matrix (L-matrix) inferred from a given ensemble of sampled alignments.
    Each alignment in the ensemble represents a possible configuration of node mappings across snapshots.
    The latent node assignment is obtained via the sequential threshold procedure implemented in
    `helper_analysis.latent_assignment_seqthreshold`.

    Steps performed:
    1. For each alignment in the ensemble:
       - Compute the L-matrix under the Poisson model formulation.
       - Obtain the latent node mapping using the sequential threshold assignment rule.
       - Reorder the L-matrix according to the inferred latent node order.
    2. Average all reordered L-matrices elementwise to obtain the final latent L estimate.

    @param A: list of adjacency matrices (NumPy arrays) representing the networks
    @param ensemble: list or array of sampled alignments (each of shape [T, N])

    @return sampled_L: array (N x N)
        Averaged latent interaction matrix computed over all sampled alignments.
    """
    T = len(ensemble[0])
    N = len(ensemble[0][0])

    # Compute latent assignments from ensemble
    ensemble_latent = latent_assignment_seqthreshold(ensemble)

    # Store reordered L-matrices
    L_values = []

    for idx, alignment in enumerate(ensemble):
        alignment = np.rint(alignment).astype(int) #ensure int np type
        latent = ensemble_latent[idx]
        L = helper_poisson.get_L(A, alignment, helper_poisson.get_Salpha(A), N, T)

        # Reorder L according to latent mapping
        invperm = np.argsort(latent)
        L_reordered = L[np.ix_(invperm, invperm)]
        L_values.append(L_reordered)

    # Average across all L-matrices
    inferred_blueprint = np.mean(np.stack(L_values), axis=0)
    
    return inferred_blueprint

from collections import OrderedDict, defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as patches
import numpy as np

def show_alignment(alignment, N, T, x, y, labels, title="", groups=None, size_values=12, size_label=14):
    """
    show_alignment:
    Visualizes the alignment matrix across multiple snapshots (connectomes), optionally showing the node group structure at the top and a legend.

    - @param alignment: 2D array (T x N) with binary node alignments
    - @param N: number of nodes
    - @param T: number of snapshots
    - @param x: whether to show x-tick labels
    - @param y: whether to show y-tick labels
    - @param labels: list of connectome names (length T)
    - @param title: title of the figure
    - @param groups: array with the group membership of each node (optional)
    - @param size_values: font size for values/text
    - @param size_label: font size for axes labels
    
    - @returns fig: matplotlib figure handle
    """
    # --- Handle groups and subplot layout ---
    if groups is not None:
        # Build dictGroups (group boundaries) from group array
        unique_groups = np.unique(groups)
        dictGroups = OrderedDict()
        prev_count = 0
        for G in unique_groups:
            count_G = np.sum(groups == G)
            dictGroups[G] = count_G + prev_count
            prev_count += count_G

        # Figure with 3 rows (top = groups, middle = mapping, bottom = legend)
        fig, axs = plt.subplots(
            nrows=3,
            gridspec_kw={'height_ratios': [0.7, 6, 3]},  # top vs alignment vs legend
            figsize=(20, 6)
        )

        # --- Plot 1: Node groups ---
        category_array = []
        init = 0
        key_int = 0
        for key, value in dictGroups.items():
            category_array.extend([key_int for _ in range(value - init)])
            init = value
            key_int += 1
        category_array = np.array(category_array)
        custom_cmap_groups = ListedColormap(
            ['#cdcdcd' if i % 2 == 0 else '#e7e7e7' for i in range(len(dictGroups))]
        )
        sns.heatmap(
            [category_array], cmap=custom_cmap_groups, cbar=False,
            xticklabels=False, yticklabels=False, ax=axs[0]
        )
        init = 0
        for key, value in dictGroups.items():
            axs[0].text(init + 0.5, 0.5, key, ha='left', va='center', fontsize=size_values, color="#333333")
            init = value
        axs[0].set_title(title, loc="left", fontsize=18, pad=20, color="#333333")
        ax_alignment = axs[1]
        ax_legend = axs[2]
    else:
        # Only the alignment plot if no groups are provided
        fig, (ax_alignment, ax_legend) = plt.subplots(nrows=2, figsize=(20, 6), gridspec_kw={'height_ratios':[6,2]})
        ax_alignment.set_title(title, loc="left", fontsize=18, pad=20, color="#333333")

    # --- Plot 2: Alignment matrix ---
    if np.sum(alignment) == N * T:  # all 1s
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#76b5c5", "#76b5c5"])
    else:
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#e28743", "#ffffff", "#76b5c5"])

    sns.heatmap(np.flipud(alignment), cmap=custom_cmap, cbar=False,
                linewidth=0.5, linecolor="#eeeee4", ax=ax_alignment,
                xticklabels=x, yticklabels=y)

    if y:
        ax_alignment.set_yticks(np.arange(T) + 0.5)
        ax_alignment.set_yticklabels(np.flipud(labels), size=size_values, va="center", rotation=0, color="#333333")
    if x:
        col_indices = np.linspace(0, N, 10, dtype=int)
        ax_alignment.set_xticks(col_indices)
        ax_alignment.set_xticklabels(col_indices, size=size_values, rotation=45, color="#333333")

    ax_alignment.set_xlabel("Node", size=size_label)
    ax_alignment.set_ylabel("Snapshot", size=size_label)

    # --- Plot 3: Legend ---
    ax_legend.axis('off')
    rect1 = patches.Rectangle((0, 0.3), 0.02, 0.2, linewidth=1, edgecolor='#c4c4c4', facecolor='#76b5c5')
    rect2 = patches.Rectangle((0.12, 0.3), 0.02, 0.2, linewidth=1, edgecolor='#c4c4c4', facecolor='#e28743')
    rect3 = patches.Rectangle((0.24, 0.3), 0.02, 0.2, linewidth=2, edgecolor='#c4c4c4', facecolor='#ffffff')
    ax_legend.add_patch(rect1)
    ax_legend.add_patch(rect2)
    ax_legend.add_patch(rect3)
    ax_legend.text(0.03, 0.37, 'Correctly aligned', fontsize=size_values)
    ax_legend.text(0.15, 0.37, 'Incorrectly aligned', fontsize=size_values)
    ax_legend.text(0.27, 0.37, 'Non-alignable', fontsize=size_values)

    plt.tight_layout(); plt.subplots_adjust(hspace=0.)
    return fig

def compute_counts_vs_model_df(A, blueprint, snapshot_labels=None, bins=10):
    """
    compute_counts_vs_model_df:
    Compute a binned DataFrame of real adjacency counts vs model rates
    using the inferred blueprint and automatically computed alphas.

    @param A: list/array of adjacency matrices for each network (T x N x N), integer counts
    @param blueprint: inferred blueprint matrix (N x N)
    @param snapshot_labels: list of labels for each network (default: [""]*T)
    @param bins: number of bins for model rate binning (default: 10)

    @return: pandas DataFrame with columns:
             ["Network", "mean_RealCounts", "std_RealCounts", "mean_ModelRates", "std_ModelRates", "err_ModelRates"]
    """
    T, N = len(A), A[0].shape[0]
    
    if snapshot_labels is None:
        snapshot_labels = [t for t in range(T)]

    # Compute alpha values automatically
    alphas = helper_poisson.get_alpha(A, helper_poisson.get_ST(A))

    # Compute lambda matrices for each alpha
    lambda_mats = np.array([blueprint * alpha for alpha in alphas])

    # Flatten data for DataFrame
    realCounts, modelRates, alphaNetworks, networkID = [], [], [], []
    
    for t in range(T):
        for i in range(N):
            for j in range(N):
                realCounts.append(A[t][i, j])
                modelRates.append(lambda_mats[t][i, j])
                alphaNetworks.append(alphas[t])
                networkID.append(t + 1)

    countsDF = pd.DataFrame({
        "RealCounts": realCounts,
        "ModelRates": modelRates,
        "Alpha": alphaNetworks,
        "Network": networkID
    })

    # Bin data by model rate for each network
    binned_data = []
    for net in sorted(countsDF["Network"].unique()):
        netDF = countsDF[countsDF["Network"] == net]
        max_rate = netDF["ModelRates"].max()
        bin_edges = np.linspace(0, max_rate, bins + 1)

        for b in range(bins):
            binnedDF = netDF[(netDF["ModelRates"] > bin_edges[b]) & 
                             (netDF["ModelRates"] <= bin_edges[b+1])]
            sample_size = len(binnedDF)

            binned_data.append({
                "Network": f"{snapshot_labels[net-1]}",
                "mean_RealCounts": binnedDF["RealCounts"].mean(),
                "std_RealCounts": binnedDF["RealCounts"].std(),
                "mean_ModelRates": binnedDF["ModelRates"].mean(),
                "std_ModelRates": binnedDF["ModelRates"].std(),
                "err_ModelRates": binnedDF["ModelRates"].std() / np.sqrt(sample_size) if sample_size > 0 else np.nan
            })

    binnedDF = pd.DataFrame(binned_data)
    return binnedDF

def plot_counts_vs_model(binnedDF):
    """
    plot_counts_vs_model:
    Plot errorbar of real counts vs model rates from a precomputed binned DataFrame.
    Each subplot is squared for consistent axes.

    @param binnedDF: pandas DataFrame obtained from compute_counts_vs_model_df
    """
    grouped = binnedDF.groupby("Network", sort=False)
    fig, axs = plt.subplots(len(grouped), 1, figsize=(4, 4 * len(grouped)))

    if len(grouped) == 1:
        axs = [axs]

    for i, (network, data) in enumerate(grouped):
        ax = axs[i]
        ax.errorbar(data["mean_ModelRates"], data["mean_RealCounts"],
                    xerr=data["std_ModelRates"], yerr=data["std_RealCounts"],
                    fmt="o", color="#5e9ab5", markersize=5)
        ax.set_xlabel("Model Rates")
        ax.set_ylabel("Real Counts")
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", color="#6c6c6c")
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{network}", fontsize=12)
            
        max_int = int(np.ceil(max_val))
        ax.set_xticks(np.linspace(0, max_int, 5, dtype=int))
        ax.set_yticks(np.linspace(0, max_int, 5, dtype=int))

    plt.tight_layout()
    plt.show()