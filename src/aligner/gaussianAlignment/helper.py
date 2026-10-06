import numpy as np
from numba import njit
from aligner.poissonAlignment import helper as helper_poisson
from math import lgamma


def synthesizeNetworks(blueprint, a, alpha, seed=None):
    """
    synthesizeNetworks(blueprint, a, alphas, seed=nothing):
    It generates synthetic networks from a blueprint (i.e., scaffold network). It generates networks based on the weights of the blueprint and the parameter alpha (i.e., desired relative densities) and a (network scale), following a Gaussian distribution. 
    
    - @param blueprint: network from which it generates the synthetic networks
    - @param alpha: list of alpha_t (relative densities) of the syntetic networks
    - @param a: value of a (related to the scale of the data)
    - @seed: seed to control for randomness, nothing by default
    
    - @returns list of synthetic networks
    """
    np.random.seed(seed)
    
    N = blueprint.shape[0]
    syntheticNets = []
    
    for alpha_t in alpha:
        thisSyntheticNet = np.zeros((N, N))  
        
        for i in range(N):
            for j in range(N):
                mean = blueprint[i, j] * alpha_t
                stdev = a * alpha_t * blueprint[i, j]
                thisSyntheticNet[i, j] = np.random.normal(mean, stdev)
        
        syntheticNets.append(thisSyntheticNet)
    
    return np.array(syntheticNets)


@njit
def get_L(A, P, alpha, a2, N, T):
    """
    get_L(A, alpha, a2, N, T)
    It calculates the blueprint matrix, L.
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param alpha: developmental parameters of all the networks
    - @param a2: scale parameter
    - @param N: number of nodes of the networks
    - @param T: number of networks

    @returns L  
    """
    L = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            sum_Aijt_alphat = 0.0
            sum_Aijt2_alphat2 = 0.0

            for t in range(T):
                Aijt_alphat = A[t][P[t][i], P[t][j]] / alpha[t]
                sum_Aijt_alphat += Aijt_alphat
                sum_Aijt2_alphat2 += Aijt_alphat ** 2

            L[i, j] = (-sum_Aijt_alphat + np.sqrt(sum_Aijt_alphat ** 2 + 4 * T * a2 * sum_Aijt2_alphat2)) / (2 * a2 * T)

    return L

@njit
def get_alpha(A, P, a2, L, N, T):
    """
    It calculates the parameter alpha for all the networks, {alpha_t}.
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param a2: scale parameter
    - @param L: blueprint matrix
    - @param N: number of nodes of the networks
    - @param T: number of networks
    
    - @returns alpha list
    """
    alpha = np.zeros(T)
    n1 = np.count_nonzero(L)
    
    for t in range(T - 1):
        sum_Aijt_Lij = 0
        sum_Aijt2_Lij2 = 0
        
        for i in range(N):
            for j in range(N):
                if L[i, j] != 0:  # Check if L[i,j] is non-zero
                    Aijt_Lij = A[t][P[t][i], P[t][j]] / L[i, j]
                    sum_Aijt_Lij += Aijt_Lij
                    sum_Aijt2_Lij2 += Aijt_Lij ** 2
        
        alpha_t = (-sum_Aijt_Lij + np.sqrt(sum_Aijt_Lij**2 + 4 * n1 * a2 * sum_Aijt2_Lij2)) / (2 * a2 * n1)
        alpha[t] = alpha_t
    
    alpha[-1] = 1.0
    
    return alpha


@njit
def get_a2(A, P, alpha, L, N, T):
    """
    get_a2(A, alpha, L, N, T):
    It calculates the parameter "a" squared (which relates to the scale of the data).
    
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param alpha: developmental parameters of all the networks
    - @param L: blueprint matrix
    - @param N: number of nodes of the networks
    - @param T: number of networks
    
    - @returns a^2
    """
    total_sum = 0.0
    n1 = np.count_nonzero(L)
    
    for t in range(T):
        for i in range(N):
            for j in range(N):
                if L[i,j] != 0:
                    total_sum +=  (A[t][P[t][i],P[t][j]]/(alpha[t]*L[i,j])-1)**2    

    return total_sum / (n1*T)


@njit
def get_convergenceProcess(A, P, N, T, L_i, alpha_i, a2_i, convergence_steps, diffstop):
    """
    get_convergenceProcess(A, N, T, L_i, alpha_i, a2_i, convergence_steps)
    It initializes the parameters of the model (a, alpha and L) following the Poisson model, but then it updates the parameters with the Gaussian formulation until convergence. This function returns the list with all parameters along the process.
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param N: number of nodes of the networks
    - @param T: number of networks
    - @param L_i: initial blueprint
    - @param alpha_i: initial alpha vector
    - @param a2_i: initial a^2
    - @param convergence_steps: number of steps to reach convergence
    
    - @returns L list,  alpha list, a2 list, stop step
    """
    L_list = [L_i]
    alpha_list = [alpha_i]
    a2_list = [a2_i]

    stop = convergence_steps

    for n in range(convergence_steps):
        # L_n(alpha_n-1, a2_n-1)
        L_n = get_L(A, P, alpha_i, a2_i, N, T)
        L_list.append(L_n)

        # alpha_n(L_n-1, a2_n-1)
        alpha_n = get_alpha(A, P, a2_i, L_n, N, T) #L_n!
        alpha_list.append(alpha_n)

        # a2_n(L_n, alpha_n)
        a2_n = get_a2(A, P, alpha_n, L_n, N, T)
        a2_list.append(a2_n)

        if np.abs(alpha_n[0] - alpha_i[0]) / alpha_i[0] <= diffstop:
            stop = n
            break

        L_i, alpha_i, a2_i = L_n, alpha_n, a2_n

    return L_list, alpha_list, a2_list, stop


@njit
def get_convergedParameters(A, P, N, T, L_i, alpha_i, a2_i, convergence_steps, diffstop):
    """
    get_convergedParameters(A, N, T, L_i, alpha_i, a2_i, convergence_steps)
    It updates the parameters with the Gaussian formulation until convergence given an initial set of parameters. This function only returns the values of the converged parameters.
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param N: number of nodes of the networks
    - @param T: number of networks
    - @param L_i: initial blueprint
    - @param alpha_i: initial alpha vector
    - @param a2_i: initial a^2
    - @param convergence_steps: number of steps to reach convergence
    
    - @returns L, alpha, a2, stop step
    """

    stop = convergence_steps

    for n in range(convergence_steps):
        # L_n(alpha_n-1, a2_n-1)
        L_n = get_L(A, P, alpha_i, a2_i, N, T)

        # alpha_n(L_n-1, a2_n-1)
        alpha_n = get_alpha(A, P, a2_i, L_n, N, T)  # L_n

        # a2_n(L_n, alpha_n)
        a2_n = get_a2(A, P, alpha_n, L_n, N, T)

        if np.abs(alpha_n[0] - alpha_i[0]) / alpha_i[0] <= diffstop: 
            stop = n
            break

        # Update parameters
        L_i = L_n
        alpha_i = alpha_n
        a2_i = a2_n

    return L_n, alpha_n, a2_n, stop


@njit
def calculate_energy(A, P, L, alpha, a2, N, T):
    """
    calculate_energy(A, L, alpha, a2, N, T)
    It calculates the energy (hamiltonian) of a specific alignment.
    - @param A: adjacency matrices of all the networks
    - @param P: permutation arrays
    - @param L: blueprin matrix
    - @param alpha: alpha parameter, list
    - @param a2: a squared parameter
    - @param N: number of nodes
    - @param T: number of networks
    
    - @returns energy (hamiltonian)
    """
    a = np.sqrt(a2)
    
    n1 = np.count_nonzero(L) 
    
    term_1 = n1 * T * np.log(a * np.sqrt(2 * np.pi))
    term_2 = 0.0
    
    for t in range(T):
        alpha_t = alpha[t]
        for i in range(N):
            for j in range(N):
                L_ij = L[i][j]
                if L_ij != 0:
                    term_2 += np.log(alpha_t*L_ij) + ((A[t][P[t][i], P[t][j]] - alpha_t*L_ij)**2)/(2*a2*(alpha_t**2)*(L_ij)**2)

    n0 = N*N-n1
    term_3 = lgamma(n0+1) + lgamma(n1+1) - lgamma(n0+n1+2)
    
    log_likeli = - term_1 - term_2 + term_3
    
    return -log_likeli