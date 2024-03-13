import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import scanpy as sc


# https://github.com/jianhuupenn/SpaGCN
clusterCmap = [
    "#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E",
    "#3A84E6", "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C",
    "#F9BD3F", "#DAB370", "#877F6C", "#268785",
    "#43BCCD", "#E17C05", "#4F6228", "#C15050", "#5F8A8B", "#B565A7",
    "#D9A641", "#5A7382"
]

# from https://github.com/digitalcytometry/cytospace
celltypeCmap = ["#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1", "#BE0032", "#C2B280",
                "#848482", "#008856", "#E68FAC", "#0067A5", "#F99379", "#604E97", "#F6A600", "#B3446C",
                "#DCD300", "#882D17", "#8DB600", "#654522", "#E25822", "#2B3D26", "#5A5156", "#E4E1E3",
                "#F6222E", "#FE00FA", "#16FF32", "#3283FE", "#FEAF16", "#B00068", "#1CFFCE", "#90AD1C",
                "#2ED9FF", "#DEA0FD", "#AA0DFE", "#F8A19F", "#325A9B", "#C4451C", "#1C8356", "#85660D",
                "#B10DA1", "#FBE426", "#1CBE4F", "#FA0087", "#FC1CBF", "#F7E1A0", "#C075A6", "#782AB6",
                "#AAF400", "#BDCDFF", "#822E1C", "#B5EFB5", "#7ED7D1", "#1C7F93", "#D85FF7", "#683B79",
                "#66B0FF", "#3B00FB"]


def plot_cci(adata, ax, adj=None, color=None, legend=True, palette=None):
    x, y = adata.obsm['spatial'][:, 0], adata.obsm['spatial'][:, 1]
    y = -y
    if color is None:
        if 'celltype' in adata.obs_keys():
            categories = adata.obs['celltype'].values.categories
            celltypes = adata.obs['celltype']
        elif 'cluster' in adata.obs_keys():
            categories = adata.obs['cluster'].values.categories
            celltypes = adata.obs['cluster']
        else:
            adata.obs['celltype'] = '-'
            adata.obs['celltype'] = adata.obs['celltype'].astype('category')
            categories = adata.obs['celltype'].values.categories
            celltypes = adata.obs['celltype']
    else:
        categories = adata.obs[color].values.categories
        celltypes = adata.obs[color]

    if palette is not None:
        cmap = palette[:len(categories)]
    else:
        cmap = celltypeCmap[:len(categories)]

    ct2color = {ct: c for ct, c in zip(categories, cmap)}

    ax.scatter(x, y, c=[ct2color[ct] for ct in celltypes])

    if adj is not None:
        if not isinstance(adj, np.ndarray):
            adj = adj.toarray()
        row, col = np.where(adj)
        for i, j in zip(row, col):
            start = [x[i], y[i]]
            end = [x[j], y[j]]
            ax.annotate('', xy=(end[0], end[1]), xytext=(start[0], start[1]), zorder=-10,
                        arrowprops=dict(color='grey', arrowstyle='-', linewidth=0.5))

    for ct, color in ct2color.items():
        ax.scatter([], [], color=color, label=ct)
    if legend:
        ax.legend(
            frameon=False,
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(ct2color) <= 15 else 2 if len(ct2color) <= 30 else 3),
        )


def my_scatter(ax, coord, labels, params, title=None, highlight=None):
    x, y = coord[:, 0], coord[:, 1]
    y = -y
    labels = pd.Categorical(labels)
    clusters = labels.categories

    clu2color = {}
    for clu, c in zip(clusters, clusterCmap[:len(clusters)]):
        if highlight is not None and clu != highlight:
            c = 'grey'
        clu2color[clu] = c

    ax.scatter(x, y, c=[clu2color[clu] for clu in labels], s=params['s'], linewidths=params.get('linewidths', None))

    for clu, color in clu2color.items():
        ax.scatter([], [], color=color, label=clu)

    if title is not None:
        ax.set_title(label=title, fontsize=params['title_size'], pad=params.get('pad', None))


def plot_cluster_pred(coord, ax, title, params, labels_true, labels_pred=None, refine=None,
                      highlight=None, show_performance=True):
    if labels_pred is not None:
        ari = metrics.adjusted_rand_score(labels_true, labels_pred)
        ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        my_scatter(coord=coord, ax=ax, labels=labels_pred, params=params, title=title, highlight=highlight)

        if show_performance:
            text = 'ARI: {:.3f} AMI: {:.3f}'.format(ari, ami)
            ax.text(0.5, 0., text, fontsize=params['text_size'],
                    transform=ax.transAxes, ha='center', va='top')
    else:
        my_scatter(coord=coord, ax=ax, labels=labels_true, params=params, title=title, highlight=highlight)


