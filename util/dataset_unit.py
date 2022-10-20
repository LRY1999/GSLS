import torch
from torch_geometric.utils import to_undirected
import os.path as osp
import numpy as np
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from scipy.sparse import coo_matrix, csr_matrix
from util.utils import row_normalize, sparse_mx_to_torch_sparse_tensor,graph_normalize
import os


def get_dataset(args):
    eps = 2.2204e-16
    if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers']:
        if args.dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            path = osp.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'dataset/')
            dataset = Planetoid(path, args.dataset_name)
        elif args.dataset_name in ['Photo', 'Computers']:
            path = osp.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'dataset/')
            dataset = Amazon(path, args.dataset_name, pre_transform=None)
        data = dataset[0]
        print(f'Number of node features: {data.num_node_features}')
        print(f'Number of node features: {data.num_features}')
        print(f'Number of edge features: {data.num_edge_features}')
        # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        # print(f'if edge indices are ordered and do not contain duplicate entries.: {data.is_coalesced()}')
        # print(f'Number of training nodes: {data.train_mask.sum()}')
        # print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        # print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        # print(f'Contains self-loops: {data.contains_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        print(len(data.edge_index[0]))
        print(len(data.edge_index[1]))
        print(data.x.shape)
        v = torch.FloatTensor(torch.ones([data.num_edges]))
        i = torch.LongTensor(np.array([data.edge_index[0].numpy(), data.edge_index[1].numpy()]))
        #coo
        A_sparse = torch.sparse.FloatTensor(i, v, torch.Size([data.num_nodes, data.num_nodes]))
        #dense
        A = A_sparse.to_dense()
        I = torch.eye(A.shape[1]).to(A.device)
        AI = A + I
        A_nomal = graph_normalize(AI)
        A_nomal = A_nomal.to_sparse()
        data.x = torch.FloatTensor(data.x)
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        # A_csr = csr_matrix(
        #     (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
        #     shape=(data.num_nodes, data.num_nodes))

        return [data, data.x], A_nomal

    elif args.dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=args.dataset_name)  # ,transform=T.ToSparseTensor()
        data = dataset[0]
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
        data.x = torch.FloatTensor(data.x)
        norm = data.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
        data.x = data.x.div(norm.expand_as(data.x))
        adj = coo_matrix(
            (np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
            shape=(data.num_nodes, data.num_nodes))
        nb_nodes = data.num_nodes
        I = coo_matrix((np.ones(nb_nodes), (np.arange(0, nb_nodes, 1), np.arange(0, nb_nodes, 1))),
                       shape=(nb_nodes, nb_nodes))
        adj_I = adj + I  # coo_matrix(sp.eye(adj.shape[0]))
        adj_I = row_normalize(adj_I)
        A_nomal = sparse_mx_to_torch_sparse_tensor(adj_I).float()
        return [data, data.x], A_nomal

    else:
        raise NotImplementedError



