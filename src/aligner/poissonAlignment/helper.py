import numpy as np
from numba import njit


@njit
def synthesize_networks(blueprint, alpha, seed=None):
    """
    synthesizeNetworks(blueprint, alphas, seed=nothing):
    It generates synthetic networks from a blueprint (i.e., scaffold network). It generates networks based on the weights of the blueprint and the parameter alpha (i.e., desired relative densities), following a Poisson distribution. 
    
    - @param blueprint: network from which it generates the synthetic networks
    - @param alpha: list of alpha_t (relative densities) of the syntetic networks
    - @seed: seed to control for randomness, nothing by default
    
    - @returns list of synthetic networks
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = blueprint.shape[0]  
    T = len(alpha)         
    synthetic_nets = np.zeros((T, N, N))  
    
    for t in range(T):
        alpha_t = alpha[t]  
        for i in range(N):
            for j in range(N):
                poisson_rate = blueprint[i, j] * alpha_t
                synthetic_nets[t, i, j] = np.random.poisson(poisson_rate)
    
    return synthetic_nets


@njit
def get_St(A, t):
    """
    get_St(A, t):
    It calculates the sum of weights in the network indexed with t.
    - @param A: adjacency matrices of all the networks 
    - @param t: index of the network under consideration 
    
    - @returns St
    """
    return np.sum(A[t])


@njit
def get_ST(A):
    """
    get_ST(A):
    It calculates maximum sum of weights in the networks (i.e., sum of weights in the densest network).
    - @param A: adjacency matrices of all the networks 
    
    - @returns St
    """
    T = len(A)
    allSt = np.zeros(T)
    
    for t in range(T):
        allSt[t] = np.sum(A[t])
        
    ST = np.max(allSt)
    
    return ST


@njit
def get_alpha(A, ST):
    """
    get_alpha(A, ST)
    It calculates alpha for all networks(i.e., relative density).
    - @param A: adjacency matrices of all the networks 
    - @param ST: sum of weights in the densest network

    - @returns alpha array
    """
    alpha = np.zeros(len(A))
    
    for t in range(len(A)):
        St = np.sum(A[t])
        alpha_t = St / ST
        alpha[t] = alpha_t
        
    return alpha


@njit
def get_Salpha(A):
    """
    get_Salpha(A):
    It calculates Salpha, the sum of all alpha_t.
    - @param A: adjacency matrices of all the networks

    - returns Salpha
    """
    T = len(A)
    ST = get_ST(A)
    alpha = get_alpha(A, ST)
    Salpha = np.sum(alpha)
    
    return Salpha


@njit
def get_Omatrix(A, P, N, T):
    """
    It generates the matrix O with the overlaps across networks (i.e., sum of weights on the same edge).  
    
    - @param A: adjacency matrices of all the networks 
    - @param P: permutation arrays (one for each network)

    - @returns matrix O
    """
    
    O = np.zeros((N, N), dtype=np.float64) 
    
    for i in range(N):
        for j in range(N):
            for t in range(T):
                O[i, j] += A[t][P[t][i], P[t][j]]
                
    return O



    
@njit
def get_nodeCoordinates(nodes, N):
    """
    Get all the coordinates that account for specific nodes (all the elements from their rows and columns).
    @param nodes: list of the nodes of interest (pair)
    @param N: number of nodes of the network
    
    @return coordinates array
    """
    i, j = nodes
    coords = np.zeros((N * 4 - 4, 2), dtype=np.int32)
    c = 0

    for n in range(N):
        coords[c, 0] = i
        coords[c, 1] = n
        c += 1
        coords[c, 0] = j
        coords[c, 1] = n
        c += 1
        if n != i and n != j:
            coords[c, 0] = n
            coords[c, 1] = i
            c += 1
            coords[c, 0] = n
            coords[c, 1] = j
            c += 1

    return coords
    

@njit
def get_O_partial_BeforeAfter(A, P_old, P_new_tmod, t_mod, nodes, N, T):
    """
    It generates arrays with the overlaps given the last permutation and the new hypothetical permutation, by considering only those rows and columns that account for the nodes to be swapped (2 at least).
   
    @param A: adjacency matrices of the nets to be aligned (specific beta)
    @param P_old: last permutation array (for all networks, t)
    @param P_new_tmod: hypothetical permutation array (for specific t)
    @param t_mod: t index of the network that we're modifying
    @param nodes: array with the node indices to be swapped
    @param N: number of nodes of the networks
    @param T: number of networks
    
    @return array with [O_lastP, O_newP] 
    """
    coords = get_nodeCoordinates(nodes, N)
    nCoords = len(coords)

    parO_old = np.zeros(nCoords, dtype=np.float64)
    parO_new = np.zeros(nCoords, dtype=np.float64)

    for c in range(nCoords):
        i, j = coords[c]
        for t in range(T):
            this_Aij = A[t][P_old[t][i], P_old[t][j]]
            parO_old[c] += this_Aij
            
            if t != t_mod:
                parO_new[c] += this_Aij
            else:
                this_Aij = A[t][P_new_tmod[i], P_new_tmod[j]]
                parO_new[c] += this_Aij

    return parO_old, parO_new


@njit
def get_L(A, P, Salpha, N, T):
    """
    It generates the blueprint by considering the overlaps within the networks (As), given their permutations and the parameters ST and Salpha.
    
    @param A: adjacency matrices of the nets to be aligned
    @param P: permutation arrays (for each A)
    @param Salpha: sum of all alphas (i.e., net densities)
    @param N: number of nodes of the networks
    @param T: number of networks
    
    @return blueprint, L
    """
    
    O = get_Omatrix(A, P, N, T)
    L = O/Salpha
    
    return L    


@njit
def calculateEnergy_total(A, P, Salpha, N, T):
    """
    It calculates the energy H (H = -SUM_ij(O_ij*log(L_ij)) of the given configuration (i.e., networks As with their permutations and the blueprint).
    
    @param A: adjacency matrices of the nets to be aligned
    @param P: permutation arrays (for each A)
    @param Salpha: sum of all alpha[t]
    @param N: number of nodes of the networks
    
    @return H "energy"
    """
    O = get_Omatrix(A, P, N, T)
    L = O / Salpha
    logL = np.log(L)
    O_and_logL = O*logL
    sum_ij = np.nansum(O_and_logL)
    
    return -sum_ij


@njit
def calculateDifEnergy_partial(A, P_old, P_new_tmod, Salpha, t_mod, nodes, N, T):
    """
    It calculates the energy difference given by the change in permutations (last permutation vs new hypothetical permutation) on network t. It only considers the rows and columns that account for the nodes to be swapped (2 at least).
    
    @param A: adjacency matrices of the nets to be aligned
    @param P_old: last permutation array
    @param P_new_tmod: hypothetical permutation array (for specific t)
    @param Salpha: sum of all the alphas (i.e., sum of densities)
    @param t: index of the considered network
    @param nodes: array with the node indices to be swapped
    @param N: number of nodes of the networks
    @param T: number of networks
    
    @return Ediff
    """
    O_before, O_after = get_O_partial_BeforeAfter(A, P_old, P_new_tmod, t_mod, nodes, N, T) #T arrays
    
    logL_before = np.log(O_before / Salpha)
    partialE_before = np.nansum(O_before * logL_before)

    logL_after = np.log(O_after / Salpha)
    partialE_after = np.nansum(O_after * logL_after)
        
    Edif = partialE_before - partialE_after #deltaE = Ei_mod - Ef_mod (deltaE = Ef-Ei = (-(Ef_ctt+Ef_mod)--(Ei_ctt+Ei_mod)) where Ei_ctt=Ef_ctt
    
    return Edif






