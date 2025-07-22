import torch
import numpy as np
import scipy.sparse as sp
import random as random
from torch.nn.functional import normalize

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:',seed)


"""
Calculation of distance matrix in the formula
"""
def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * torch.bmm(y, x) # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()


"""
KNN adjacency matrix calculation
"""
def create_sparse(I):

    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])

    assert len(similar) == len(index)
    indices = torch.tensor([index, similar]).cuda()
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)),
                                     [I.shape[0], I.shape[0]])

    return result


"""
AGC module for multi-view data
"""
def node_filter_adaptive_i(dataset, view, K, reg_zero, beta, alpha, verbose=False):
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)

    for v in range(view):

        # Calculate matrix A^v through closed form solution
        D = pairwise_distance(xs[v])  # Calculate distance matrix D
        D = normalize(D)
        res = torch.mm(xs[v], torch.transpose(xs[v], 0, 1))  # Calculate H*H.T
        inv = torch.inverse(res + beta * torch.eye(xs[v].shape[0]))  # # Inverse matrix in A^v
        front = res - alpha/2 * D  # The first part of A
        S = torch.mm(front, inv)  # obtain A^v

        S = torch.where(S > 0, torch.sign(S), torch.tensor(0.0))  # Eq.(7) If S > 0, the value is 1
        adj.append(S)

        n, f = xs[v].shape

        adj[v] = adj[v].cpu().numpy()

        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]

        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()

        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)

        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E

        feats_props = np.transpose(feats_props, (2, 1, 0))

        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])),
                                                           np.append(feats_props[:, node_idx, 0], np.zeros(1)),
                                                           rcond=None)  # Solve the w by Eq.(9)

            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)


            if verbose:
                print("Finished node %i of %i." % (node_idx, n))

        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')


"""
AGC module for multi-graph data
"""
def node_filter_adaptive_g(dataset, view, K, reg_zero, verbose=False):
    print('graph')
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    graph = dataset.get_graph(index)
    for v in range(view):

        adj.append(graph[v])

        n, f = xs[v].shape

        adj[v] = adj[v].cpu().numpy()

        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]

        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()

        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)

        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E

        feats_props = np.transpose(feats_props, (2, 1, 0))

        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])),
                                                           np.append(feats_props[:, node_idx, 0], np.zeros(1)),
                                                           rcond=None)  # Solve the w by Eq.(9)
            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)

            if verbose:
                print("Finished node %i of %i." % (node_idx, n))

        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')

