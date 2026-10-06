import numpy as np
from numba import njit
from tqdm import tqdm  # For progress bar
from aligner.poissonAlignment import helper as helper_poisson
from aligner.gaussianAlignment import helper as helper_gaussian
from time import time
import pickle

def initial_permutation(A):
    """
    initial_permutation:
    Generates the initial permutation for MCMC alignment. Nodes are ordered by descending strength 
    in each snapshot rather than randomly.

    - @param A: list of adjacency matrices of the networks to be aligned

    - @returns P: list of permutation arrays, one for each snapshot
    """
    T, N, N = A.shape
    
    P = np.empty((T, N), dtype="int") 
    
    for t in range(T):
        nodeStrength = A[t].sum(axis=0) + A[t].sum(axis=1)
        P[t] = np.argsort(-nodeStrength) #by strength (descending)

    return P

def group_boundaries(groups):
    """
    group_boundaries:
    Generates the array of group boundaries so as to preserve the structure of the groups in the permutations during the group-guided alignment.

    @param groups: array with the groups of each node (nodes ordered as in the adj matrices of the networks)

    @return gloc: array with the boundaries of the groups 
    """
    unique_groups, counts = np.unique(groups, return_counts=True)
    gloc = np.concatenate(([0], np.cumsum(counts)))  # group boundaries

    return gloc

def initial_permutation_groups(A, groups):
    """
    initial_permutation_groups:
    Generates the initial permutation for group-guided MCMC alignment. 
    Nodes are ordered by descending strength in each snapshot, preserving the group structure.

    - @param A: list of adjacency matrices of the networks to be aligned
    - @param groups: array with the group of each node (nodes ordered as in A)
    
    - @returns P: list of permutation arrays, one for each snapshot
    """
    P_ng = initial_permutation(A)  # not considering groups

    unique_groups, counts = np.unique(groups, return_counts=True)
    gloc = np.concatenate(([0], np.cumsum(counts)))  # group boundaries
    print(f"Groups: {unique_groups}")
    print(f"Group boundary array: {gloc}")

    # group-structured permutation
    P = []
    for t in range(len(A)):
        P_t = []
        for group in np.unique(groups):
            P_t.extend(P_ng[t][np.where(groups[P_ng[t]] == group)[0]])
        P.append(P_t)
    P = np.array(P)
    
    return P

@njit 
def generateHypoPerm_t(P, t, nodes, N): #poisson algorithm
    """
    generateHypoPerm_t:
    Generates an hypothetical permutation of the nodes of one network (t-time) by considering the swap between some nodes.
    
    @param P: permutation arrays (for a given beta)
    @param t: time index 
    @param nodes: list of the nodes to be swapped
    @param N: number of nodes of the networks
    
    @return hypothetical permutation array with the swap (for index t)
    """
    hypo_Pt = P[t].copy()
    i, j = nodes
    hypo_Pt[i], hypo_Pt[j] = hypo_Pt[j], hypo_Pt[i]
        
    return hypo_Pt

@njit 
def generateHypoPerm(P, t, nodes, N): #gaussian algorithm
    """
    generateHypoPerm:
    Generates an hypothetical permutation of the nodes of all networks (for one replica).
    
    @param P: permutation arrays (for a given beta)
    @param t: time index 
    @param nodes: list of the nodes to be swapped
    @param N: number of nodes of the networks
    
    @return hypothetical permutation array with the swap (for index t)
    """
    hypo_P = P.copy()
    i, j = nodes
    hypo_P[t][i], hypo_P[t][j] = hypo_P[t][j], hypo_P[t][i]
        
    return hypo_P


@njit
def metropolis_poisson(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r):
    """
    metropolis_poisson:
    Evaluates a proposed swap between nodes in network A^t for a given beta using the Poisson model.

    - @param t: index of the network to be modified
    - @param A_r: adjacency matrices of the networks
    - @param P_r: permutation arrays (one per network)
    - @param betas: list of beta values (inverse temperatures)
    - @param Esteps_r: array storing energies over MC sweeps for each beta
    - @param Salpha: Salpha parameter for the Poisson model
    - @param b: index of the beta (replica) being updated
    - @param N: number of nodes in the networks
    - @param T: number of networks
    - @param step_idx_r: array with the current step index for each replica

    - @returns: boolean indicating whether the swap was accepted; updates P_r and Esteps_r in place
    """
    #t = np.random.randint(0, T)
    nodes = np.random.permutation(N)[:2]
    
    hypo_Pt = generateHypoPerm_t(P_r[b], t, nodes, N)
    dE = helper_poisson.calculateDifEnergy_partial(A_r[b], P_r[b], hypo_Pt, Salpha, t, nodes, N, T)
    
    r = np.random.rand()
    acceptswap = r < np.exp(-betas[b] * dE)

    if acceptswap:
        P_r[b][t] = hypo_Pt
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]] + dE
        
    else:
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]]  # Retain the last

    step_idx_r[b] += 1
    
    return acceptswap

@njit
def metropolis_poisson_anchors(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r, possible_nodes):
    """
    metropolis_poisson_anchors:
    Evaluates a proposed swap between nodes in network A^t for a given beta, excluding anchor nodes, using the Poisson model.

    - @param t: index of the network to be modified
    - @param A_r: adjacency matrices of the networks
    - @param P_r: permutation arrays (one per network)
    - @param betas: list of beta values (inverse temperatures)
    - @param Esteps_r: array storing energies over MC sweeps for each beta
    - @param Salpha: Salpha parameter for the Poisson model
    - @param b: index of the beta (replica) being updated
    - @param N: number of nodes in the networks
    - @param T: number of networks
    - @param step_idx_r: array with the current step index for each replica
    - @param possible_nodes: array of nodes eligible for swapping (excluding anchors)

    - @returns: boolean indicating whether the swap was accepted; updates P_r and Esteps_r in place
    """
    #t = np.random.randint(0, T)
    nodes = np.random.permutation(possible_nodes)[:2]
    
    hypo_Pt = generateHypoPerm_t(P_r[b], t, nodes, N)
    dE = helper_poisson.calculateDifEnergy_partial(A_r[b], P_r[b], hypo_Pt, Salpha, t, nodes, N, T)
    
    r = np.random.rand()
    acceptswap = r < np.exp(-betas[b] * dE)

    if acceptswap:
        P_r[b][t] = hypo_Pt
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]] + dE
        
    else:
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]]  # Retain the last

    step_idx_r[b] += 1
    
    return acceptswap

@njit
def metropolis_poisson_groups(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r, gloc):
    """
    metropolis_poisson_groups:
    Evaluates a proposed swap between nodes in network A^t for a given beta, while preserving the group structure.

    - @param t: index of the network to be modified
    - @param A_r: adjacency matrices of the networks
    - @param P_r: permutation arrays (one per network)
    - @param betas: list of beta values (inverse temperatures)
    - @param Esteps_r: array storing energies over MC sweeps for each beta
    - @param Salpha: Salpha parameter for the Poisson model
    - @param b: index of the beta (replica) being updated
    - @param N: number of nodes in the networks
    - @param T: number of networks
    - @param step_idx_r: array with the current step index for each replica
    - @param gloc: array with the group boundaries

    - @returns: boolean indicating whether the swap was accepted; updates P_r and Esteps_r in place
    """
    #t = np.random.randint(0, T) 
    i = np.random.randint(0, N)
    j = np.random.randint(gloc[np.where(gloc<=i)[0][-1]], gloc[np.where(gloc>i)[0][0]]) #lower, upper limits within the same group
    nodes = np.array([i, j])
    
    hypo_Pt = generateHypoPerm_t(P_r[b], t, nodes, N)
    dE = helper_poisson.calculateDifEnergy_partial(A_r[b], P_r[b], hypo_Pt, Salpha, t, nodes, N, T)
    
    r = np.random.rand()
    acceptswap = r < np.exp(-betas[b] * dE)

    if acceptswap:
        P_r[b][t] = hypo_Pt
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]] + dE
        
    else:
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]]  # Retain the last

    step_idx_r[b] += 1
    
    return acceptswap
    

@njit
def metropolis_gaussian(t, A_r, P_r, betas, Esteps_r, L_r, alpha_r, a2_r, b, N, T, convSteps, convDiff, step_idx_r):
    """
    metropolis_gaussian:
    Evaluates a proposed swap between nodes in network A^t for a given beta under the Gaussian model.

    - @param t: index of the network to be modified
    - @param A_r: replica adjacency matrices of the networks
    - @param P_r: replica permutation arrays (one per network)
    - @param betas: list of beta values (inverse temperatures)
    - @param Esteps_r: array storing energies over MC sweeps for each beta
    - @param L_r: replica array of current L parameters
    - @param alpha_r: replica array of current alpha parameters
    - @param a2_r: replica array of current a2 parameters
    - @param b: index of the beta (replica) being updated
    - @param N: number of nodes in the networks
    - @param T: number of networks
    - @param convSteps: number of steps for parameter convergence
    - @param convDiff: threshold for change in parameters to consider convergence
    - @param step_idx_r: array tracking the number of MC steps performed for each replica

    - @returns: boolean indicating whether the swap was accepted; updates A_r, P_r, L_r, alpha_r, a2_r, and Esteps_r in place
    """
    #t = np.random.randint(0, T)
    nodes = np.random.permutation(N)[:2]
    
    hypo_P = generateHypoPerm(P_r[b], t, nodes, N)

    L, alpha, a2, stop = helper_gaussian.get_convergedParameters(
        A_r[b], hypo_P, N, T, L_r[b], alpha_r[b], a2_r[b], convSteps, convDiff 
    ) 

    Ef = helper_gaussian.calculate_energy(A_r[b], hypo_P, L, alpha, a2, N, T)

    dE = Ef - Esteps_r[b, step_idx_r[b]]
    r = np.random.rand()
    acceptswap = r < np.exp(-betas[b] * dE)

    if acceptswap:
        P_r[b] = hypo_P
        
        L_r[b] = L
        alpha_r[b] = alpha
        a2_r[b] = a2
        
        Esteps_r[b, step_idx_r[b]+1] = Ef
        
    else:
        Esteps_r[b, step_idx_r[b]+1] = Esteps_r[b, step_idx_r[b]]  # Retain the last

    step_idx_r[b] += 1
    
    return acceptswap, stop


@njit
def parallelTemperingExchange_poisson(P_r, Esteps_r, betas, odd, step_idx_r):
    """
    parallelTemperingExchange_poisson:
    Performs one Parallel Tempering step for the Poisson model. 
    Evaluates swaps of energy/configurations between consecutive replicas (at different betas/temperatures). 
    Only odd or even pairs of replicas are considered depending on the 'odd' flag. 
    If accepted, permutation matrices and energies are updated in place.

    - @param P_r: replica permutation arrays of the networks (list of NumPy arrays)
    - @param Esteps_r: array storing replica energies over MC steps
    - @param betas: list of beta values (inverse temperatures)
    - @param odd: boolean, whether to swap odd-indexed pairs (True) or even-indexed pairs (False)
    - @param step_idx_r: array tracking the number of MC steps performed for each replica

    - @returns: number of accepted replica swaps (updates P_r and Esteps_r globally)
    """
    B = len(betas)
    totalAccepted = 0

    # Decide the beta indices based on whether odd or even swaps are to be performed
    b1_list = range(0, B - 1, 2) if odd else range(1, B - 1, 2)

    for b1 in b1_list:
        b2 = b1 + 1
        dE = (betas[b1] - betas[b2]) * (Esteps_r[b2, step_idx_r[b2]] - Esteps_r[b1, step_idx_r[b1]])
        r = np.random.rand()
        accepted_swap = r < np.exp(-dE)

        if accepted_swap:
            totalAccepted += 1

            # Swap permutation arrays
            P_r[b1], P_r[b2] = np.copy(P_r[b2]), np.copy(P_r[b1])
              
            # Swap energies
            Esteps_r[b1, step_idx_r[b1]], Esteps_r[b2, step_idx_r[b2]] = Esteps_r[b2, step_idx_r[b2]], Esteps_r[b1, step_idx_r[b1]]
            step_idx_r[b1], step_idx_r[b2] = step_idx_r[b2], step_idx_r[b1]

    return totalAccepted


@njit
def parallelTemperingExchange_gaussian(P_r, L_r, alpha_r, a2_r, Esteps_r, betas, odd, step_idx_r):
    """
    parallelTemperingExchange_gaussian:
    Performs one Parallel Tempering step for the Gaussian model. 
    Evaluates swaps of energy/configurations between consecutive replicas at different betas/temperatures. 
    Only odd or even pairs of replicas are considered depending on the 'odd' flag. 
    If accepted, permutation matrices, energies, and model parameters (L, alpha, a2) are updated in place.

    - @param P_r: replica permutation arrays of the networks (list of NumPy arrays)
    - @param L_r: replica array of the last L parameter values
    - @param alpha_r: replica array of the last alpha parameter values
    - @param a2_r: replica array of the last a2 parameter values
    - @param Esteps_r: replica array of energies over MC steps
    - @param betas: list of beta values (inverse temperatures)
    - @param odd: boolean, whether to swap odd-indexed pairs (True) or even-indexed pairs (False)
    - @param step_idx_r: array tracking the number of MC steps performed for each replica

    - @returns: number of accepted replica swaps (updates P_r, L_r, alpha_r, a2_r, and Esteps_r globally)
    """
    B = len(betas)
    totalAccepted = 0

    # Decide the beta indices based on whether odd or even swaps are to be performed
    b1_list = range(0, B - 1, 2) if odd else range(1, B - 1, 2)

    for b1 in b1_list:
        b2 = b1 + 1
        dE = (betas[b1] - betas[b2]) * (Esteps_r[b2, step_idx_r[b2]] - Esteps_r[b1, step_idx_r[b1]])
        r = np.random.rand()
        accepted_swap = r < np.exp(-dE)
        
        if accepted_swap:
            totalAccepted += 1

            # Swap permutation matrices
            P_r[b1], P_r[b2] = P_r[b2], P_r[b1]

            # Swap model parameters
            L_r[b1], L_r[b2] = L_r[b2], L_r[b1]
            alpha_r[b1], alpha_r[b2] = alpha_r[b2], alpha_r[b1]
            a2_r[b1], a2_r[b2] = a2_r[b2], a2_r[b1]
              
            # Swap energies
            Esteps_r[b1, step_idx_r[b1]], Esteps_r[b2, step_idx_r[b2]] = Esteps_r[b2, step_idx_r[b2]], Esteps_r[b1, step_idx_r[b1]]
            step_idx_r[b1], step_idx_r[b2] = step_idx_r[b2], step_idx_r[b1]

    return totalAccepted


@njit
def getNodesOrder(P, N, T):
    """
    getNodesOrder:
    Given the permutation matrices, it extracts the order of the nodes for each network with respect to the version of the networks from which the Monte Carlo simulation started.

    - @param P: list of permutation matrices (NumPy arrays)
    - @param N: number of nodes
    - @param T: number of networks

    - @returns: list of node orders for all networks
    """
    nodesOrder = np.zeros((T, N))  
    for t in range(T):  
        for col in range(N): 
            for row in range(N):  # find the row where the `1` is
                if P[t, row, col] == 1:
                    nodesOrder[t, col] = row
                    break
    return nodesOrder

def sampling_poisson_groups(A, P, betas, S, sampleFrom, sampleFreq, groups, filesave=None):
    """
    sampling_poisson_groups:
    Performs a group-guided Monte Carlo simulation with Parallel Tempering to align networks under a Poisson formulation. Samples node alignments with lowest energy at T=1.

    - @param A: list of adjacency matrices (NumPy arrays)
    - @param P: list of initial permutation matrices
    - @param betas: array of inverse temperatures
    - @param S: number of Monte Carlo sweeps
    - @param sampleFrom: sweep number to start sampling from
    - @param sampleFreq: number of sweeps between consecutive samples
    - @param groups: array indicating the group membership of each node
    - @param filesave: file path for saving sampled configurations

    - @returns:
        * Esweeps: list of energies over sweeps for each replica
        * samples: list of sampled node orders
        * groundState: node order corresponding to the lowest energy configuration
        * MCacceptance: Monte Carlo acceptance rate during sampling
    """
    N = A[0].shape[0]  # Number of nodes
    T = len(A)         # Number of networks
    B = len(betas)     # Number of beta (temperature) replicas

    MCsteps = T * N
    b_beta1 = np.where(betas == 1)[0][0]  # Index of beta = 1 replica

    # Group boundary array 
    gloc = np.array(group_boundaries(groups))

    # Initial energy and parameters
    ST = helper_poisson.get_ST(A)
    Salpha = helper_poisson.get_Salpha(A)

    # Energy calculation
    Ei = helper_poisson.calculateEnergy_total(A, P, Salpha, N, T)

    # Configuration replicas (one for each beta)
    A_r = np.empty((B, *A.shape), A.dtype)
    P_r = np.empty((B, *P.shape), P.dtype)
    Esweeps_r = np.zeros((B, S+1), float)
    Esteps_r = np.zeros((B, S*MCsteps+1), float)
    for b in range(B):
        A_r[b] = A.copy() 
        P_r[b] = P.copy()
        Esweeps_r[b][0] = Ei
        Esteps_r[b][0] = Ei
    step_idx_r = np.zeros((B), int) #for each beta, to store the id of the MC step (for storing the E_step)

    # Variables to store alignment information
    MCacceptance = 0
    stepsSampling = 0
    
    numSamples = int((S - sampleFrom) / sampleFreq)
    samples = np.empty((numSamples, T, N), dtype=np.int64) 
    sample_idx = 0

    # Monte Carlo sampling
    for sweep in tqdm(range(1, S + 1), desc="Sweeps"):
        for step in range(MCsteps):
            t = np.random.randint(0, T)
            for b in range(B):
                MCexchange = metropolis_poisson_groups(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r, gloc)

                if sweep > sampleFrom and sweep % sampleFreq == 0 and b == b_beta1:
                    stepsSampling += 1
                    MCacceptance += MCexchange

        if (sweep % 1000 == 0) and filesave is not None:
            with open(f"{filesave}_sweep{sweep}.pkl", 'wb') as f:
                pickle.dump([Esweeps_r, P_r], f)
            print(f"Data saved for sweep {sweep}.")

        # Parallel tempering exchanges
        oddPT = (sweep % 2 != 0)
        PTexchange = parallelTemperingExchange_poisson(P_r, Esteps_r, betas, oddPT, step_idx_r)

        # Update energy sweeps
        for b in range(B):
            Esweeps_r[b, sweep] = Esteps_r[b][step_idx_r[b]]

        # Sampling
        if sweep > sampleFrom and sweep % sampleFreq == 0:
            samples[sample_idx] = P_r[b_beta1]
            sample_idx += 1

    # Ground state configuration
    b_minE = np.argmin([Esweeps_r[b][-1] for b in range(B)])
    print(f"Replica with minimum energy: {b_minE}")
    groundState = P_r[b_minE]

    # Monte Carlo acceptance rate
    MCacceptance_rate = MCacceptance / stepsSampling if stepsSampling > 0 else 0

    return Esweeps_r, samples, groundState, MCacceptance_rate

def sampling_poisson(A, P, betas, S, sampleFrom, sampleFreq, filesave=None):
    """
    sampling_poisson:
    Performs a group-guided Monte Carlo simulation with Parallel Tempering to align networks under a Poisson formulation. Samples node alignments with lowest energy at T=1.

    - @param A: list of adjacency matrices (NumPy arrays)
    - @param P: list of initial permutation matrices
    - @param betas: array of inverse temperatures
    - @param S: number of Monte Carlo sweeps
    - @param sampleFrom: sweep number to start sampling from
    - @param sampleFreq: number of sweeps between consecutive samples
    - @param filesave: file path for saving sampled configurations

    - @returns:
        * Esweeps: list of energies over sweeps for each replica
        * samples: list of sampled node orders
        * groundState: node order corresponding to the lowest energy configuration
        * MCacceptance: Monte Carlo acceptance rate during sampling
    """
    N = A[0].shape[0]  # Number of nodes
    T = len(A)         # Number of networks
    B = len(betas)     # Number of beta (temperature) replicas

    MCsteps = T * N
    b_beta1 = np.where(betas == 1)[0][0]  # Index of beta = 1 replica

    # Initial energy and parameters
    ST = helper_poisson.get_ST(A)
    Salpha = helper_poisson.get_Salpha(A)

    # Energy calculation
    Ei = helper_poisson.calculateEnergy_total(A, P, Salpha, N, T)

    # Configuration replicas (one for each beta)
    A_r = np.empty((B, *A.shape), A.dtype)
    P_r = np.empty((B, *P.shape), P.dtype)
    Esweeps_r = np.zeros((B, S+1), float)
    Esteps_r = np.zeros((B, S*MCsteps+1), float)
    for b in range(B):
        A_r[b] = A.copy() 
        P_r[b] = P.copy()
        Esweeps_r[b][0] = Ei
        Esteps_r[b][0] = Ei
    step_idx_r = np.zeros((B), int) #for each beta, to store the id of the MC step (for storing the E_step)

    # Variables to store alignment information
    MCacceptance = 0
    stepsSampling = 0
    
    numSamples = int((S - sampleFrom) / sampleFreq)
    samples = np.empty((numSamples, T, N), dtype=np.int64) 
    sample_idx = 0

    # Monte Carlo sampling
    for sweep in tqdm(range(1, S + 1), desc="Sweeps"):
        for step in range(MCsteps):
            t = np.random.randint(0, T)
            for b in range(B):
                MCexchange = metropolis_poisson(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r)

                if sweep > sampleFrom and sweep % sampleFreq == 0 and b == b_beta1:
                    stepsSampling += 1
                    MCacceptance += MCexchange

        if (sweep % 1000 == 0) and filesave is not None:
            with open(f"{filesave}_sweep{sweep}.pkl", 'wb') as f:
                pickle.dump([Esweeps_r, P_r], f)
            print(f"Data saved for sweep {sweep}.")

        # Parallel tempering exchanges
        oddPT = (sweep % 2 != 0)
        PTexchange = parallelTemperingExchange_poisson(P_r, Esteps_r, betas, oddPT, step_idx_r)

        # Update energy sweeps
        for b in range(B):
            Esweeps_r[b, sweep] = Esteps_r[b][step_idx_r[b]]

        # Sampling
        if sweep > sampleFrom and sweep % sampleFreq == 0:
            samples[sample_idx] = P_r[b_beta1]
            sample_idx += 1

    # Ground state configuration
    b_minE = np.argmin([Esweeps_r[b][-1] for b in range(B)])
    print(f"Replica with minimum energy: {b_minE}")
    groundState = P_r[b_minE]

    # Monte Carlo acceptance rate
    MCacceptance_rate = MCacceptance / stepsSampling if stepsSampling > 0 else 0

    return Esweeps_r, samples, groundState, MCacceptance_rate

def sampling_poisson_anchors(A, P, betas, S, sampleFrom, sampleFreq, anchors, filesave):
    """
    sampling_poisson_anchors:
    Performs a Monte Carlo simulation with Parallel Tempering to align networks under a Poisson formulation. There are some anchors nodes that must remain mapped together. Samples node alignments with lowest energy at T=1.

    - @param A: list of adjacency matrices (NumPy arrays)
    - @param P: list of initial permutation matrices
    - @param betas: array of inverse temperatures
    - @param S: number of Monte Carlo sweeps
    - @param sampleFrom: sweep number to start sampling from
    - @param sampleFreq: number of sweeps between consecutive samples
    - @param anchors: array with the node positions ns that can not be swapped
    - @param filesave: file path for saving sampled configurations

    - @returns:
        * Esweeps: list of energies over sweeps for each replica
        * samples: list of sampled node orders
        * groundState: node order corresponding to the lowest energy configuration
        * MCacceptance: Monte Carlo acceptance rate during sampling
    """
    N = A[0].shape[0]  # Number of nodes
    T = len(A)         # Number of networks
    B = len(betas)     # Number of beta (temperature) replicas

    possible_nodes = np.array([n for n in range(N) if n not in anchors]) #list with the possible nodes to be swapped, all but anchors

    MCsteps = T * N
    b_beta1 = np.where(betas == 1)[0][0]  # Index of beta = 1 replica

    # Initial energy and parameters
    ST = helper_poisson.get_ST(A)
    Salpha = helper_poisson.get_Salpha(A)

    # Energy calculation
    Ei = helper_poisson.calculateEnergy_total(A, P, Salpha, N, T)

    # Configuration replicas (one for each beta)
    A_r = np.empty((B, *A.shape), A.dtype)
    P_r = np.empty((B, *P.shape), P.dtype)
    Esweeps_r = np.zeros((B, S+1), float)
    Esteps_r = np.zeros((B, S*MCsteps+1), float)
    for b in range(B):
        A_r[b] = A.copy() 
        P_r[b] = P.copy()
        Esweeps_r[b][0] = Ei
        Esteps_r[b][0] = Ei
    step_idx_r = np.zeros((B), int) #for each beta, to store the id of the MC step (for storing the E_step)

    # Variables to store alignment information
    MCacceptance = 0
    stepsSampling = 0
    
    numSamples = int((S - sampleFrom) / sampleFreq)
    samples = np.empty((numSamples, T, N), dtype=np.int64) 
    sample_idx = 0

    # Monte Carlo sampling
    for sweep in tqdm(range(1, S + 1), desc="Sweeps"):
        for step in range(MCsteps):
            t = np.random.randint(0, T)
            for b in range(B):
                MCexchange = metropolis_poisson_anchors(t, A_r, P_r, betas, Esteps_r, Salpha, b, N, T, step_idx_r, possible_nodes)

                if sweep > sampleFrom and sweep % sampleFreq == 0 and b == b_beta1:
                    stepsSampling += 1
                    MCacceptance += MCexchange

        if (sweep % 1000 == 0):
            with open(f"{filesave}_sweep{sweep}.pkl", 'wb') as f:
                pickle.dump([Esweeps_r, P_r], f)
            print(f"Data saved for sweep {sweep}.")

        # Parallel tempering exchanges
        oddPT = (sweep % 2 != 0)
        PTexchange = parallelTemperingExchange_poisson(P_r, Esteps_r, betas, oddPT, step_idx_r)

        # Update energy sweeps
        for b in range(B):
            Esweeps_r[b, sweep] = Esteps_r[b][step_idx_r[b]]

        # Sampling
        if sweep > sampleFrom and sweep % sampleFreq == 0:
            samples[sample_idx] = P_r[b_beta1]
            sample_idx += 1

    # Ground state configuration
    b_minE = np.argmin([Esweeps_r[b][-1] for b in range(B)])
    print(f"Replica with minimum energy: {b_minE}")
    groundState = P_r[b_minE]

    # Monte Carlo acceptance rate
    MCacceptance_rate = MCacceptance / stepsSampling if stepsSampling > 0 else 0

    return Esweeps_r, samples, groundState, MCacceptance_rate


def sampling_gaussian(A, P, betas, S, sampleFrom, sampleFreq, convSteps, convDiff, internalFileName=None):
    """
    sampling_gaussian:
    Performs a Monte Carlo simulation with Parallel Tempering to align networks
    under a Gaussian formulation. Samples node alignments with the lowest energy at T=1.

    - @param A: list of adjacency matrices (NumPy arrays)
    - @param P: list of initial permutation matrices
    - @param betas: array of inverse temperatures
    - @param S: number of Monte Carlo sweeps
    - @param sampleFrom: sweep number to start sampling from
    - @param sampleFreq: number of sweeps between consecutive samples
    - @param convSteps: number of steps for parameter convergence (Gaussian model only)
    - @param convDiff: convergence tolerance for Gaussian parameters
    - @param internalFileName: path to save internal sampling data

    - @returns:
        * Esweeps: list of energies over sweeps for each replica
        * samples: list of sampled node orders
        * groundState: node order corresponding to the lowest energy configuration
        * MCacceptance: Monte Carlo acceptance rate during sampling
    """
    N = A[0].shape[0]  # Number of nodes
    T = len(A)         # Number of networks
    B = len(betas)     # Number of beta (temperature) replicas

    MCsteps = T * N
    b_beta1 = np.where(betas == 1)[0][0]  # Index of beta = 1 replica

    # Initial energy and parameters
    ST = helper_poisson.get_ST(A)
    Salpha = helper_poisson.get_Salpha(A)

    # Poisson initialization
    L_i = helper_poisson.get_L(A, P, Salpha, N, T)
    alpha_i = helper_poisson.get_alpha(A, ST)
    a2_i = helper_gaussian.get_a2(A, P, alpha_i, L_i, N, T) 

    # Parameter convergence
    L, alpha, a2, stop = helper_gaussian.get_convergedParameters(A, P, N, T, L_i, alpha_i, a2_i, convSteps, -1) #intial params: all conv steps (-1 diff so it doesn't stop)

    # Energy calculation
    Ei = helper_gaussian.calculate_energy(A, P, L, alpha, a2, N, T) #helper_gaussian.

    # Configuration replicas (one for each beta)
    A_r = np.empty((B, *A.shape), A.dtype)
    P_r = np.empty((B, *P.shape), P.dtype)
    Esweeps_r = np.zeros((B, S+1), float)
    Esteps_r = np.zeros((B, S*MCsteps+1), float)
    L_r = np.empty((B, *L.shape), L.dtype)
    alpha_r = np.empty((B, *alpha.shape), alpha.dtype)
    a2_r = np.empty((B), float)

    for b in range(B):
        A_r[b] = A.copy() 
        P_r[b] = P.copy()
        Esweeps_r[b][0] = Ei
        Esteps_r[b][0] = Ei
        L_r[b] = L.copy()
        alpha_r[b] = alpha.copy()
        a2_r[b] = a2

    step_idx_r = np.zeros((B), int) #for each beta, to store the id of the MC step (for storing the E_step)
    stop_list = []

    # Variables to store alignment information
    MCacceptance = 0
    stepsSampling = 0
    numSamples = int((S - sampleFrom) / sampleFreq)
    samples = np.empty((numSamples, T, N), dtype=np.int64) 
    sample_idx = 0

    # Monte Carlo sampling
    for sweep in tqdm(range(1, S + 1), desc="Sweeps"):
        for step in range(MCsteps):
            t = np.random.randint(0, T)
            for b in range(B):
                MCexchange, stop = metropolis_gaussian(t, A_r, P_r, betas, Esteps_r, L_r, alpha_r, a2_r, b, N, T, 
                                                       convSteps, convDiff, step_idx_r)
                stop_list.append(stop)

                if sweep > sampleFrom and sweep % sampleFreq == 0 and b == b_beta1:
                    stepsSampling += 1
                    MCacceptance += MCexchange

        # Parallel tempering exchanges
        oddPT = (sweep % 2 != 0)
        PTexchange = parallelTemperingExchange_gaussian(P_r, L_r, alpha_r, a2_r, Esteps_r, betas, oddPT,step_idx_r)

        # Update energy sweeps
        for b in range(B):
            Esweeps_r[b, sweep] = Esteps_r[b][step_idx_r[b]]

        #storing internal results every 10 sweeps:
        if sweep % 10 == 0 and internalFileName is not None:
            int_res = dict({"E": Esweeps_r, "P": P_r, "L": L_r, 
                           "alpha": alpha_r, "a2": a2_r})
            with open(f"../results/wtn_gaussian/{internalFileName}_internal_results_sweep_{sweep}.pkl", "wb") as file: 
                pickle.dump(int_res, file)
            
        # Sampling
        if sweep > sampleFrom and sweep % sampleFreq == 0:
            samples[sample_idx] = P_r[b_beta1]
            sample_idx += 1

    # Ground state configuration
    b_minE = np.argmin([Esweeps_r[b][-1] for b in range(B)])
    print(f"Replica with minimum energy: {b_minE}")
    groundState = P_r[b_minE]

    # Monte Carlo acceptance rate
    MCacceptance_rate = MCacceptance / stepsSampling if stepsSampling > 0 else 0

    return Esweeps_r, samples, groundState, MCacceptance_rate, stop_list