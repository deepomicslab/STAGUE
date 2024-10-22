import os
import numpy as np
import torch
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import dense_to_sparse
from sklearn import metrics
import scanpy as sc
from scipy.stats import mode
import anndata


def get_feat_mask(features, rate):
    feat_size = features.shape[1]
    mask = torch.ones(features.shape, device=features.device)
    samples = np.random.choice(feat_size, size=int(feat_size * rate), replace=False)

    mask[:, samples] = 0
    return mask


def dense2sparse(adj):
    (row, col), val = dense_to_sparse(adj)
    num_nodes = adj.size(0)
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))


def normalize_adj_symm(adj):
    assert adj.size(0) == adj.size(1)
    if not isinstance(adj, SparseTensor):
        adj = dense2sparse(adj)

    if not adj.has_value():
        adj = adj.fill_value(1., dtype=torch.float32)

    deg = torch_sparse.sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))
    adj = torch_sparse.mul(adj, deg_inv_sqrt.view(1, -1))

    return adj


class ClusteringMetrics:
    def __init__(self, true_label, predict_label, X=None):
        self.true_label = true_label
        self.pred_label = predict_label
        self.X = X

    def evaluationClusterModelFromLabel(self):
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        ami = metrics.adjusted_mutual_info_score(self.true_label, self.pred_label)
        return ari, ami


def refine_labels(raw_labels, dist_sort_idx, n_neigh):
    """
    from https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/utils.py
    """
    n_cell = len(raw_labels)
    raw_labels = np.tile(raw_labels, (n_cell, 1))
    idx = dist_sort_idx[:, 1:n_neigh + 1]
    new_labels = raw_labels[np.arange(n_cell)[:, None], idx]
    new_labels = mode(new_labels, axis=1).mode

    return new_labels


def mclust_R(embedding, n_clusters, random_state, modelNames='EEE'):
    np.random.seed(random_state)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_state)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), n_clusters, modelNames)
    if not isinstance(res, rpy2.rinterface_lib.sexp.NULLType):
        clusters = np.array(res[-2])
    else:
        clusters = np.ones(len(embedding))

    return clusters


def run_leiden(embedding, n_clusters, range_min=0, range_max=5, max_steps=100):
    adata = anndata.AnnData(X=embedding)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)

    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max - this_min) / 2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_clusters:
            this_max = this_resolution
        elif this_clusters < n_clusters:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_clusters, this_resolution))
            return adata.obs["leiden"].values
        this_step += 1

    return


def run_louvain(embedding, n_clusters, range_min=0, range_max=5, max_steps=100):
    adata = anndata.AnnData(X=embedding)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)

    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs['louvain'].nunique()

        if this_clusters > n_clusters:
            this_max = this_resolution
        elif this_clusters < n_clusters:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f" % (n_clusters, this_resolution))
            return adata.obs["louvain"].values
        this_step += 1

    return


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list
