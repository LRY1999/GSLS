import argparse
from scipy.sparse import diags
import numpy as np
import torch
from termcolor import cprint
from sklearn.neighbors import kneighbors_graph
import torch.nn.functional as F

def get_args(dataset_class, dataset_name, key="") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Graph')
    parser.add_argument("--key", default=key)
    parser.add_argument("--seed", default=5)
    parser.add_argument('--data-root', default="~/graph-data")
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate', dest='lr')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--lr2', default=0.001, type=float, help='learning rate2', dest='lr2')
    parser.add_argument('--w_loss1', type=float, default=10, help='')
    parser.add_argument('--w_loss2', type=float, default=1, help='')
    parser.add_argument('--margin1', type=float, default=0.8, help='')
    parser.add_argument('--margin2', type=float, default=0.2, help='')
    parser.add_argument('--dim', type=float, default=128, help='')
    parser.add_argument('--cfg', default=[512, 128], help='')
    parser.add_argument('--NegNum', default=5)
    parser.add_argument('--wd1', type=float, default=0.00005, help='weight_decay1')
    parser.add_argument('--wd2', type=float, default=0.005, help='weight_decay2')
    parser.add_argument('--dropout', type=float, default=0.3, help='')
    parser.add_argument('--lamda', type=float, default=0.9999)
    parser.add_argument('--k', default=25, type=int, help='k in pruning')
    args = parser.parse_args()
    return args

# graph
def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def graph_normalize(A):
    eps = 1e-16
    deg_inv_sqrt = (A.sum(dim=-1).clamp(min=0.) + eps).pow(-0.5)
    if A.size()[0] != A.size()[1]:
        A = deg_inv_sqrt.unsqueeze(-1) * (deg_inv_sqrt.unsqueeze(-1) * A)
    else:
        A = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A


def nearest_nei(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def symmetrize(adj):
    return (adj + adj.T) / 2


def normalize(adj, mode, sparse=False):
    esp = 1e-16
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + esp)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + esp)
            return inv_degree[:, None] * adj
        else:
            print("error")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            a = torch.sparse.sum(adj, dim=1)
            b = a.values()
            inv_degree = 1. / (b + esp)
            D_value = inv_degree[adj.indices()[0]]
        else:
            print("error")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())

def dglgraph_to_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx

def knn_f(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def cal_similarity_graph(emb):
    similarity_graph = torch.mm(emb, emb.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph