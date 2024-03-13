import copy
import os
import argparse
from src import GCL, GraphLearner
from src.utils import *
from sklearn.cluster import KMeans
import random
import torch.nn.functional as F
import pandas as pd
from main import STAGUE


class Experiment(STAGUE):
    def __init__(self, args):
        super().__init__(args)

    def predict(self, model, cell_features, learned_adj):
        model.eval()
        with torch.no_grad():
            embedding, _ = model(cell_features.detach(), learned_adj)
        embedding = embedding.cpu().detach().numpy()
        ari_ls, ami_ls = [], []
        for clu_trial in range(5):
            kmeans = KMeans(n_clusters=self.nclasses, random_state=clu_trial, n_init="auto").fit(embedding)
            predict_labels = kmeans.predict(embedding)
            if self.args.refine != 0:
                predict_labels = refine_labels(predict_labels, self.dist_sort_idx, self.args.refine)
            cm_all = ClusteringMetrics(self.labels.cpu().numpy(), predict_labels)
            ari, ami = cm_all.evaluationClusterModelFromLabel()
            ari_ls.append(ari)
            ami_ls.append(ami)
        ari, ami = np.mean(ari_ls), np.mean(ami_ls)
        return embedding, ari, ami

    def train(self):
        args = self.args

        job_dir = args.output_dir
        if not os.path.exists(job_dir):
            os.mkdir(job_dir)
        with open(os.path.join(job_dir, 'args.txt'), 'w') as f:
            print(args, file=f)

        record_ls = []
        for trial in range(args.ntrials):
            self.setup_seed(trial)

            # prepare model
            anchor_adj = normalize_adj_symm(self.adj_knn).to(args.device)
            bn = not args.no_bn
            model = GCL(nlayers=args.nlayers, cell_feature_dim=self.gene_exp.size(1), in_dim=args.exp_out,
                        hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                        dropout=args.dropout, dropout_adj=args.dropedge_rate, margin=args.margin, bn=bn)

            model.graph_lerner = GraphLearner(nlayers=args.nlayers, isize=args.exp_out, neighbor=args.k,
                                              gamma=args.gamma, adj=anchor_adj, coords=self.cell_coords,
                                              device=args.device, omega=args.adj_weight)

            model = model.to(args.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            print(model)

            # train
            best_ari = -np.inf
            best_ami = None
            best_embedding = None
            best_model = None
            ari_records = []
            ami_records = []
            identity = dense2sparse(torch.eye(self.gene_exp.shape[0])).to(args.device)
            for epoch in range(1, 1 + args.epochs):
                optimizer.zero_grad()
                model.train()

                cell_features = model.get_cell_features(self.gene_exp)

                _, z1 = model(cell_features, anchor_adj, args.maskfeat_rate_anchor)

                learned_adj, _ = model.get_learned_adj(cell_features)
                _, z2 = model(cell_features, learned_adj, args.maskfeat_rate_learner)

                idx = torch.randperm(self.gene_exp.shape[0])
                _, z1_neg = model(cell_features[idx], identity, args.maskfeat_rate_anchor, training=False)

                d_pos = F.pairwise_distance(z2, z1)
                d_neg = F.pairwise_distance(z2, z1_neg)
                margin_label = -1 * torch.ones_like(d_pos)

                loss_nt = model.sim_loss(z1, z2, args.temperature)
                loss_triplet = model.margin_loss(d_pos, d_neg, margin_label) * args.margin_weight
                loss = loss_nt + loss_triplet
                loss.backward()
                optimizer.step()

                # Structure Bootstrapping
                anchor_adj = dense2sparse(
                    anchor_adj.to_dense() * args.tau + learned_adj.detach().to_dense() * (1 - args.tau)
                )

                embedding, ari, ami = self.predict(model, cell_features, learned_adj)
                ari_records.append(ari)
                ami_records.append(ami)
                print("Epoch {:05d} | NT-Xent Loss {:.5f} | Triplet Loss {:.5f} | ARI {:5f}| AMI {:5f}".format(
                    epoch, loss_nt.item(), loss_triplet.item(), ari, ami))

                if ari > best_ari:
                    best_ari = ari
                    best_ami = ami
                    best_embedding = embedding
                    best_model = copy.deepcopy(model)

            # save best embedding and model for each trial
            emb_path = os.path.join(job_dir, 'trial{}_ARI{:.5f}_AMI{:.5f}.npy'.format(trial, best_ari, best_ami))
            np.save(emb_path, best_embedding)

            # model_path = os.path.join(job_dir, 'trial{}_model.pt'.format(trial))
            # torch.save(best_model.state_dict(), model_path)

            # save records for each trial
            df_record = pd.DataFrame({'trial': trial, 'ARI': ari_records, 'AMI': ami_records})
            record_ls.append(df_record)

        # save job args
        best_ari_ls = []
        best_ami_ls = []
        for df_ in record_ls:
            idx = df_['ARI'].argmax()
            best_ari_ls.append(df_['ARI'][idx])
            best_ami_ls.append(df_['AMI'][idx])
        all_record = pd.concat(record_ls, ignore_index=True)
        all_record.to_csv(os.path.join(job_dir, 'training_record.csv'), index=False)
        metric_path = os.path.join(job_dir,
                                   'metric_result_meanARI{:.5f}_stdARI{:.5f}_meanAMI{:.5f}_stdAMI{:.5f}'.format(
                                       np.mean(best_ari_ls), np.std(best_ari_ls),
                                       np.mean(best_ami_ls), np.std(best_ami_ls)))
        open(metric_path, 'a').close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    #
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--adata_file', type=str)
    parser.add_argument('--hvg', action='store_true')

    #
    parser.add_argument('--lr', type=float)
    parser.add_argument('--k', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--exp_out', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--rep_dim', type=int)
    parser.add_argument('--proj_dim', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--margin_weight', type=float, default=1)
    parser.add_argument('--adj_weight', type=float, default=0.5)
    parser.add_argument('--n_clusters', type=int, default=None)
    parser.add_argument('--a_k', type=int, default=5)
    parser.add_argument('--filter_cell', type=int, default=100)
    parser.add_argument('--norm_target', default=None)
    parser.add_argument('--refine', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ntrials', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--w_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--dropedge_rate', type=float, default=0.5)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--maskfeat_rate_learner', type=float, default=0.6)
    parser.add_argument('--maskfeat_rate_anchor', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.999)
    parser.add_argument('--no_bn', action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    experiment = Experiment(args)
    experiment.train()
