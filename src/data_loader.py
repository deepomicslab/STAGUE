import os
import scanpy as sc
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse
import torch
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
from torch_sparse import SparseTensor
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def get_coord(adata):
    obs_columns = adata.obs.columns
    if 'spatial' in adata.obsm.keys():
        cell_coords = adata.obsm['spatial']
    elif 'x' in obs_columns and 'y' in obs_columns:
        cell_coords = adata.obs[['x', 'y']].values
    elif 'st_x' in obs_columns and 'st_y' in obs_columns:
        cell_coords = adata.obs[['st_x', 'st_y']].values
    else:
        raise Exception('Can not find coordinates in adata')
    return cell_coords


def get_sketched_cci_fast(adata, num_neighbors=5, is_undirected=True, n=1000):
    coords = get_coord(adata)
    num_cells = adata.n_obs

    k = num_neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords)
    k_nearest_idx = nbrs.kneighbors(coords, return_distance=False)

    #
    rows = np.repeat(np.arange(num_cells), k)
    cols = k_nearest_idx[:, 1:].flatten()  # excluding self
    data = np.ones(k * num_cells)
    adj = csr_matrix((data, (rows, cols)), shape=(num_cells, num_cells))

    #
    if is_undirected:
        adj = adj + adj.T
        adj.data = np.ones_like(adj.data)

    adata.obsp['knn_adj'] = adj

    # +1 to retain n-nearest neighbors after excluding self
    dist_n = min(num_cells, n + 1)
    dist_sort, dist_sort_idx = nbrs.kneighbors(coords, n_neighbors=dist_n)

    return adata, dist_sort, dist_sort_idx


def get_sketched_cci(adata, num_neighbors=5, is_undirected=True):
    adjacency = np.zeros(shape=(adata.n_obs, adata.n_obs), dtype=int)
    coords = get_coord(adata)
    dist = distance_matrix(coords, coords)

    dist_sort_idx = np.argsort(dist, axis=1)
    dist_sort = np.take_along_axis(dist, dist_sort_idx, axis=1)

    for i, n_idx in enumerate(dist_sort_idx):
        n_idx = n_idx[n_idx != i][:num_neighbors]
        adjacency[i, n_idx] = 1
        assert adjacency[i, i] != 1

    if is_undirected:
        print('Use undirected cell-cell communication graph')
        adjacency = ((adjacency + adjacency.T) > 0).astype(int)

    adata.obsp['knn_adj'] = csr_matrix(adjacency)
    return adata, dist_sort, dist_sort_idx


def load_data_from_raw(args):
    adata = sc.read_h5ad(args.adata_file)
    print('Raw adata:', adata, sep='\n')

    if args.hvg:
        sc.pp.filter_genes(adata, min_cells=args.filter_cell)
        print('After flitering: ', adata.shape)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

    args.norm_target = float(args.norm_target) if args.norm_target is not None else None
    sc.pp.normalize_total(adata, target_sum=args.norm_target)
    sc.pp.log1p(adata)

    if 'highly_variable' in adata.var.keys():
        adata = adata[:, adata.var['highly_variable']].copy()

    print('Processed adata:', adata, sep='\n')

    if scipy.sparse.issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = adata.X

    gene_exp = torch.tensor(counts, dtype=torch.float32)

    if args.n_clusters is not None:
        labels = None
        nclasses = args.n_clusters
    else:
        if 'cluster' in adata.obs.columns:
            key = 'cluster'
        elif 'domain' in adata.obs.columns:
            key = 'domain'
        elif 'ground_truth' in adata.obs.columns:
            key = 'ground_truth'
        else:
            raise Exception('Cluster annotations not found.')
        cat = adata.obs[key].astype('category').values
        labels = torch.tensor(cat.codes, dtype=torch.long)
        nclasses = len(cat.categories)
    print(f'Clustering with {nclasses} centers')

    #
    if args.sparse_learner:
        adata, dist_sort, dist_sort_idx = get_sketched_cci_fast(adata, num_neighbors=args.a_k)
    else:
        adata, dist_sort, dist_sort_idx = get_sketched_cci(adata, num_neighbors=args.a_k)

    (row, col), val = from_scipy_sparse_matrix(adata.obsp['knn_adj'])
    num_nodes = adata.obsp['knn_adj'].shape[0]
    adj_knn = SparseTensor(row=row, col=col, value=val.to(torch.float32), sparse_sizes=(num_nodes, num_nodes))

    cell_coords = get_coord(adata)
    cell_coords = torch.tensor(cell_coords, dtype=torch.float32)

    return adata, gene_exp, labels, nclasses, adj_knn, cell_coords, dist_sort, dist_sort_idx
