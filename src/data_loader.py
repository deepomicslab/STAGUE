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


def get_sketched_cci(adata, num_neighbors=5, is_undirected=True):
    adjacency = np.zeros(shape=(adata.n_obs, adata.n_obs), dtype=int)
    coords = get_coord(adata)
    dis = distance_matrix(coords, coords)

    neighbors_idx = np.argsort(dis, axis=1)

    for i, n_idx in enumerate(neighbors_idx):
        n_idx = n_idx[n_idx != i][:num_neighbors]
        adjacency[i, n_idx] = 1
        assert adjacency[i, i] != 1

    if is_undirected:
        print('Use undirected cell-cell communication graph')
        adjacency = ((adjacency + adjacency.T) > 0).astype(int)

    adata.obsp['knn_adj'] = csr_matrix(adjacency)
    return adata


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
        else:
            raise Exception('Cluster annotations not found.')
        cat = adata.obs[key].astype('category').values
        labels = torch.tensor(cat.codes, dtype=torch.long)
        nclasses = len(cat.categories)
    print(f'Clustering with {nclasses} centers')

    #
    adata = get_sketched_cci(adata, num_neighbors=args.a_k)
    (row, col), val = from_scipy_sparse_matrix(adata.obsp['knn_adj'])
    num_nodes = adata.obsp['knn_adj'].shape[0]
    adj_knn = SparseTensor(row=row, col=col, value=val.to(torch.float32), sparse_sizes=(num_nodes, num_nodes))

    cell_coords = get_coord(adata)
    cell_coords = torch.tensor(cell_coords, dtype=torch.float32)

    return adata, gene_exp, labels, nclasses, adj_knn, cell_coords
