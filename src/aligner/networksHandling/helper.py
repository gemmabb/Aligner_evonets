import numpy as np
import numba
import os
import pandas as pd 

#regular python, not numba:

def load_networks(data_folder, net_ids, file_prefix=""):
    """
    loadNetworks(dataFolder, netIds, filePrefix = ""):
    It loads the networks (as adjacency matrices in independent csv files) stored at data_folder (file names should be: "(filePrefix_)adjMat_[netId].csv").
    
    - @param data_folder: name of the folder with the networks
    - @param ids: list with the networks ids (integer format)
    - @param file_prefix: prefix of the files (empty by default)
    
    - @returns list of connectomes as numpy arrays
    """
    num_networks = len(net_ids)
    networks = []

    for net_id in net_ids:
        file_name = f"{file_prefix}adjMat_{net_id}.csv"
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path, index_col=0)
        
        networks.append(df.values) 

    return np.array(networks).astype(np.float64)


def invperm(perm):
    """
    invperm(perm):
    It returns the inverse permutation array.
    
    - @param perm: permutation array
    """
    N = len(perm)
    inv_perm = np.empty(N, dtype=int)
    for i in range(N):
        inv_perm[perm[i]] = i
    return inv_perm


def generatePermArray(N, T, seed=None):
    """ 
    generatePermArray(N, T, seed):
    It generates a permutation array. The correspondence between the original sequence of nodes and the permuted/shuffled version of it can be seen as the position being the original and this permArray[pos] being the new id.
    - @param N: number of nodes of the networks
    - @param T: number of networks
    - @param seed: seed to control for randomness, nothing by default
    
    - @returns array of T permutation arrays
    """            
    if seed is not None:
        np.random.seed(seed)
    
    permArray = np.empty((T, N), dtype="int")
    
    for t in range(T):
        permArray[t] = np.random.permutation(N)
    
    return permArray


def generatePermArray_withGroups(N, T, nodesGroup, seed=None):
    """ 
    generatePermArray_withGroups(N, T, nodeGroups, seed=nothing):
    It generates a permutation array. The correspondence between the original sequence of nodes and the permuted/shuffled version of it can be seen as the position being the original and this permArray[pos] being the new id. Node groups are conserved here.
    
    - @param N: number of nodes of the networks
    - @param T: number of networks
    - @param nodeGroups: annotation of the group of each node (list of strings) 
    - @param seed: seed to control for randomness, nothing by default
    
    - @return array of T permutation arrays
    """    
    if seed is not None:
        np.random.seed(seed)
    
    permArray = np.zeros((T, N), dtype="int")
    groups = np.unique(nodesGroup)
    
    for t in range(T):
        this_perm = np.zeros(N, dtype="int") 
        idx = 0  
        
        for group in groups:
            print(np.where(nodesGroup == group))
            this_group_nodes = np.where(nodesGroup == group)[0]
            np.random.shuffle(this_group_nodes)
            
            this_perm[idx:idx+len(this_group_nodes)] = this_group_nodes
            idx += len(this_group_nodes)
        
        permArray[t] = invperm(this_perm)
        
    return permArray


def permuteNetworks_withPermArrays(networks, permArrays):
    """
    It permutes a list of networks given a list of dictionaries where there's the correspondence between the original order of the nodes and a permuted/shuffled version of it.
    
    - @param networks: list of networks to be permuted
    - @param permutationDicts: list of dictionaries with original-permuted correspondence of node order
    
    - @returns permuted networks  
    """         
    T = len(networks)
    N = networks[0].shape[0]

    permutedNetworks = []
    
    for t in range(T):
        permMat = np.zeros((N, N))
        for n in range(N):
            permMat[permArrays[t][n], n] = 1
        
        thisPermutedNetwork = permMat @ networks[t] @ permMat.T
        permutedNetworks.append(thisPermutedNetwork.astype(np.float64))
    
    return np.array(permutedNetworks)