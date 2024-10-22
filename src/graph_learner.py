import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.spatial import distance_matrix
from .utils import *


class GraphLearner(nn.Module):
    def __init__(self, nlayers, isize, neighbor, gamma, adj, coords, device, omega):
        super().__init__()

        self.adj = adj.to(device)

        d_matrix = torch.tensor(distance_matrix(coords, coords), dtype=torch.float32, device=device)
        d_sorted, _ = d_matrix.sort()

        # mask
        c1 = d_matrix > 0
        d_cut = torch.median(d_sorted[:, neighbor])
        c2 = d_matrix <= d_cut
        self.adj_mask = torch.logical_and(c1, c2)  # 1-k neighbor, no self-loop

        d_matrix = torch.where(self.adj_mask, d_matrix, torch.inf) / d_cut
        self.s_d = 1 / torch.exp(gamma * torch.pow(d_matrix, 2))

        self.convs = nn.ModuleList()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=isize, out_channels=isize)] +
            [GCNConv(in_channels=isize, out_channels=isize) for _ in range(nlayers-1)]
        )

        self.input_dim = isize
        self.omega = omega

    def internal_forward(self, h):
        for i, conv in enumerate(self.convs):
            h = conv(h, self.adj.t())
            if i != (len(self.convs) - 1):
                h = F.relu(h, inplace=True)
        return h

    def forward(self, features, eps=1e-8):
        h = self.internal_forward(features)
        h_norm = torch.linalg.vector_norm(h, ord=2, dim=1, keepdim=True)
        s1 = (h @ h.t()) / (h_norm @ h_norm.t() + eps)
        s1 = torch.where(s1 >= 0, s1, 0)  # symmetric, [0,1]

        s2 = self.s_d
        s = self.omega * s1 + (1 - self.omega) * s2
        s = torch.where(self.adj_mask, s, 0)
        s_norm = normalize_adj_symm(s)
        return s_norm, s


class GraphLearnerSparse(nn.Module):
    def __init__(self, nlayers, isize, neighbor, gamma, adj, d_sorted, d_indices, device, omega):
        """
        :param nlayers:
        :param isize:
        :param neighbor:
        :param gamma:
        :param adj:
        :param d_sorted: sorted distances to the n-nearest neighbors, including self, n for large number
        :param d_indices: indices corresponding to d_sorted
        :param device:
        :param omega:
        """
        super().__init__()

        self.adj = adj.to(device)
        d_sorted = d_sorted[:, 1:]  # excluding self
        d_indices = d_indices[:, 1:]  # excluding self

        # mask
        d_cut = np.median(d_sorted[:, neighbor - 1])  # cutoff distance
        mask = d_sorted.flatten() <= d_cut
        i_indices = np.repeat(np.arange(adj.size(0)), d_indices.shape[1])  # central node
        j_indices = d_indices.flatten()  # neighboring node
        self.i_indices = torch.tensor(i_indices[mask], dtype=torch.long, device=device)
        self.j_indices = torch.tensor(j_indices[mask], dtype=torch.long, device=device)
        self.ij_dist = torch.tensor(d_sorted.flatten()[mask], dtype=torch.float32)

        self.s_d = (1 / torch.exp(gamma * torch.pow(self.ij_dist/d_cut, 2))).to(device)

        self.convs = nn.ModuleList()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=isize, out_channels=isize)] +
            [GCNConv(in_channels=isize, out_channels=isize) for _ in range(nlayers-1)]
        )

        self.omega = omega

    def internal_forward(self, h):
        for i, conv in enumerate(self.convs):
            h = conv(h, self.adj.t())
            if i != (len(self.convs) - 1):
                h = F.relu(h, inplace=True)
        return h

    def forward(self, features):
        h = self.internal_forward(features)
        h = F.normalize(h, p=2, dim=-1)
        s_val = (h[self.i_indices] * h[self.j_indices]).sum(dim=-1)
        s_val = torch.where(s_val >= 0, s_val, 0)
        s_val = self.omega * s_val + (1 - self.omega) * self.s_d

        #
        s = SparseTensor(row=self.i_indices, col=self.j_indices, value=s_val,
                         sparse_sizes=(self.adj.size(0), self.adj.size(1)))
        s = s.to_symmetric(reduce='mean')
        s_norm = normalize_adj_symm(s)
        return s_norm, s

