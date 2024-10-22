import copy
import os
import argparse
from src import GCL, GraphLearner, GraphLearnerSparse
from src.data_loader import load_data_from_raw
from src.utils import *
from sklearn.cluster import KMeans
import random
import time
import torch.nn.functional as F
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import pandas as pd


class STAGUE:
    def __init__(self, args):
        self.args = args
        adata, gene_exp, labels, nclasses, adj_knn, cell_coords, dist_sort, dist_sort_idx = load_data_from_raw(args)
        gene_exp = gene_exp.to(args.device)

        self.adata = adata
        self.gene_exp = gene_exp
        self.labels = labels
        self.nclasses = nclasses
        self.adj_knn = adj_knn
        self.cell_coords = cell_coords
        self.dist_sort = dist_sort
        self.dist_sort_idx = dist_sort_idx

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

    def predict(self, model, cell_features, learned_adj):
        model.eval()
        with torch.no_grad():
            embedding, _ = model(cell_features.detach(), learned_adj)
        embedding = embedding.cpu().detach().numpy()

        #
        clu_model = self.args.clu_model
        if clu_model == 'kmeans':
            kmeans = KMeans(n_clusters=self.nclasses, random_state=0, n_init="auto").fit(embedding)
            labels_pred = kmeans.predict(embedding)
        elif clu_model == 'mclust':
            labels_pred = mclust_R(embedding, n_clusters=self.nclasses, random_state=0)
        elif clu_model == 'leiden':
            labels_pred = run_leiden(embedding, n_clusters=self.nclasses)
        elif clu_model == 'louvain':
            labels_pred = run_louvain(embedding, n_clusters=self.nclasses)
        else:
            raise Exception(f'Unknown cluster model {clu_model}')

        if self.args.refine != 0:
            labels_pred = refine_labels(labels_pred, self.dist_sort_idx, self.args.refine)

        return embedding, labels_pred

    def train(self):
        args = self.args

        job_dir = args.output_dir
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        with open(os.path.join(job_dir, 'args.txt'), 'w') as f:
            print(args, file=f)

        trial = self.args.seed
        self.setup_seed(trial)

        # prepare model
        anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)
        bn = not args.no_bn
        model = GCL(nlayers=args.nlayers, cell_feature_dim=self.gene_exp.size(1), in_dim=args.exp_out,
                    hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                    dropout=args.dropout, dropout_adj=args.dropedge_rate, margin=args.margin, bn=bn)

        if args.sparse_learner:
            model.graph_lerner = GraphLearnerSparse(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.k,
                                                    gamma=args.gamma, adj=anchor_adj, d_sorted=self.dist_sort,
                                                    d_indices=self.dist_sort_idx, device=args.device,
                                                    omega=args.adj_weight)
        else:
            model.graph_lerner = GraphLearner(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.k,
                                              gamma=args.gamma, adj=anchor_adj, coords=self.cell_coords,
                                              device=args.device, omega=args.adj_weight)

        model = model.to(args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        print(model)

        # train
        identity = dense2sparse(torch.eye(self.gene_exp.shape[0])).to(args.device)
        for epoch in range(1, 1 + args.epochs):
            optimizer.zero_grad()
            model.train()

            cell_features = model.get_cell_features(self.gene_exp)

            _, z1 = model(cell_features, anchor_adj, args.maskfeat_rate_anchor)

            learned_adj, learned_adj_raw = model.get_learned_adj(cell_features)
            _, z2 = model(cell_features, learned_adj, args.maskfeat_rate_learner)

            idx = torch.randperm(self.gene_exp.shape[0])
            _, z1_neg = model(cell_features[idx], identity, args.maskfeat_rate_anchor, training=False)

            d_pos = F.pairwise_distance(z2, z1)
            d_neg = F.pairwise_distance(z2, z1_neg)
            margin_label = -1 * torch.ones_like(d_pos)

            # loss_nt
            if args.sim_batch_size == 0:
                loss_nt = model.sim_loss(z1, z2, args.temperature)
            else:
                node_indices = list(range(self.gene_exp.shape[0]))
                batches = split_batch(node_indices, args.sim_batch_size)
                loss_nt = 0
                for batch in batches:
                    weight = len(batch) / self.gene_exp.shape[0]
                    loss_nt += model.sim_loss(z1[batch], z2[batch], args.temperature) * weight

            loss_triplet = model.margin_loss(d_pos, d_neg, margin_label) * args.margin_weight
            loss = loss_nt + loss_triplet
            loss.backward()
            optimizer.step()

            # Structure Bootstrapping
            anchor_adj = anchor_adj.mul_nnz(torch.tensor(args.tau, dtype=torch.float32), layout='coo')
            learned_ = learned_adj.detach().mul_nnz(
                torch.tensor(1 - args.tau, dtype=torch.float32), layout='coo')
            anchor_adj = anchor_adj.add(learned_)

            print("Epoch {:05d} | NT-Xent Loss {:.5f} | Triplet Loss {:.5f}".format(epoch, loss_nt.item(), loss_triplet.item()))

        # save adj, embedding, and labels
        embedding, labels_pred = self.predict(model, cell_features, learned_adj)
        self.adata.obsm['embedding'] = embedding
        self.adata.obs['cluster_pred'] = pd.Categorical(labels_pred)

        self.adata.obsp['learned_adj_normalized'] = csr_matrix(learned_adj.detach().cpu().to_dense().numpy())
        self.adata.obsp['learned_adj_raw'] = csr_matrix(learned_adj_raw.detach().cpu().to_dense().numpy())

        adata_path = os.path.join(job_dir, 'adata_processed.h5ad'.format(trial))
        self.adata.write(adata_path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--adata_file', type=str,
                        help='Path to the input AnnData file.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory path where outputs will be saved.')
    parser.add_argument('--n_clusters', type=int,
                        help='Number of clusters to identify.')

    # important optional arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computing device to use; {"cpu", "cuda"}.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model training.')
    parser.add_argument('--clu_model', type=str, default='kmeans',
                        help='Clustering algorithm; {"kmeans", "mclust", "louvain", "leiden"}.')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--a_k', type=int, default=5,
                        help='Number of neighbors for constructing the raw cell graph adjacency matrix.')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of neighbors for determining the cutoff distance when '
                             'inferring the learner view\'s cell graph adjacency matrix. '
                             'Larger values (>= 15) are preferred.')
    parser.add_argument('--adj_weight', type=float, default=0.5,
                        help='Weight of the cosine similarity term in the learned adjacency matrix. '
                             'Lower values (<= 0.5) are preferred for data with low gene coverage.')
    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.6,
                        help='Dropout rate for augmenting learner view\' feature matrix. '
                             'Moderate values (~ 0.5) are preferred for data with low gene coverage.')
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.9,
                        help='Dropout rate for augmenting positive and negative views\' feature matrices. '
                             'Larger values (>= 0.8) are preferred.')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Non-negative margin of the triplet loss. Lower values (<= 1) are generally preferred.')
    parser.add_argument('--margin_weight', type=float, default=2,
                        help='Weight of the triplet loss.')
    # other optional arguments
    parser.add_argument('--sparse_learner', action='store_true',
                        help='Use the sparsification in the graph learner module.')
    parser.add_argument('--sim_batch_size', type=int, default=0,
                        help='Compute NT-Xent loss for a mini-batch or set to 0 to disable')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature parameter for the NT-Xent loss.')
    parser.add_argument('--exp_out', type=int, default=512,
                        help='Feature dimension of the processed gene expression matrix.')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of layers of the GCN encoder.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Output dimension of the GCN encoder\' first layer.')
    parser.add_argument('--rep_dim', type=int, default=64,
                        help='Feature dimension of the learned cell embeddings.')
    parser.add_argument('--proj_dim', type=int, default=64,
                        help='Output dimension of the projection head.')
    parser.add_argument('--gamma', type=float, default=2,
                        help='Decay rate of the spatial decay term in the learned adjacency matrix.')
    parser.add_argument('--hvg', action='store_true', help='Select highly variable genes.')
    parser.add_argument('--filter_cell', type=int, default=100,
                        help='Minimum number of cells required to keep a gene during filtering, i.e., '
                             '\'min_cells\' in scanpy.pp.filter_genes. Used only for --hvg is enabled.')
    parser.add_argument('--norm_target', default=None,
                        help='Normalization target sum, i.e., \'target_sum\' in scanpy.pp.normalize_total.')
    parser.add_argument('--refine', type=int, default=0,
                        help='Number of neighbors used to refine the predicted cluster labels. 0 for non-refinement.')
    parser.add_argument('--w_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for the input feature matrix of the GCN encoder.')
    parser.add_argument('--dropedge_rate', type=float, default=0.5,
                        help='Dropout rate for augmenting the adjacency matrix.')
    parser.add_argument('--tau', type=float, default=0.999,
                        help='Conservation rate of the raw cell graph adjacency matrix '
                             'when updating it with the learned one.')
    parser.add_argument('--no_bn', action='store_true',
                        help='Disable Batch Normalization when processing the input gene expression matrix.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    experiment = STAGUE(args)
    experiment.train()
