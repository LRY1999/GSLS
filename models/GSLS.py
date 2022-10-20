import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
from util.utils import nearest_nei, symmetrize, normalize, dglgraph_to_sparse, knn_f, cal_similarity_graph, top_k

class my_MLP(nn.Module):
    def __init__(self, in_channel, cfg, batch_norm=False, out_layer=None):
        super(my_MLP, self).__init__()
        in_channels = in_channel
        layer_num = len(cfg)
        self.fc = nn.ModuleList()
        for i, v in enumerate(cfg):
            out_channels = v
            mlp = nn.Linear(in_channels, out_channels)
            if batch_norm:
                self.fc.extend([mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()])
            elif i != (layer_num - 1):
                self.fc.extend([mlp, nn.ReLU()])
            else:
                self.fc.append(mlp)
            in_channels = out_channels
        if out_layer != None:
            mlp = nn.Linear(in_channels, out_layer)
            self.fc.append(mlp)

    def forward(self, x):
        for i,layer in enumerate(self.fc):
                x=layer(x)
        return x



class USLonG(nn.Module):
    def __init__(self, n_in ,cfg = None, dropout = 0.2):
        super(USLonG, self).__init__()
        self.GCN = my_MLP(n_in, cfg)
        self.act = nn.ReLU()
        self.dropout = dropout
        self.sparse = True
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feature, adj, t, k):
        feature = F.dropout(feature, self.dropout, training=self.training)
        #anchor
        h_s = self.GCN(feature)
        h_a = F.dropout(h_s, 0.2, training=self.training)
        if self.sparse:
            h_a = torch.spmm(adj, h_a)
            rows, cols, values = knn_f(h_a, k, b=1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = F.relu(values_)
            Adj = dgl.graph((rows_, cols_), num_nodes=feature.shape[0], device='cuda')
            Adj.edata['w'] = values_
            Adj=dglgraph_to_sparse(Adj)
            learn_adj = (1-t) * Adj.cuda() + t * adj.cuda()
            h_p = torch.spmm(learn_adj.cuda(), h_a.cuda())

        else:
            h_a = torch.mm(adj, h_a)
            h_a = F.normalize(h_a, dim=1, p=2)
            similarities = cal_similarity_graph(h_a)
            similarities = top_k(similarities, k + 1)
            similarities = F.relu(similarities)
            Adj = symmetrize(similarities)
            Adj = normalize(Adj, 'sym', self.sparse)
            learn_adj = torch.add((1-t) * Adj.cuda(), t * adj.cuda()).to(running_device)
            h_p = torch.mm(learn_adj.cuda(), h_a.cuda())
        return h_s, h_p ,learn_adj


    def embed(self, feature, adj,t, k):
        h_s = self.GCN(feature)
        if self.sparse:
            h_a = torch.spmm(adj, h_s)
            rows, cols, values = knn_f(h_a, k, b=1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = F.relu(values_)
            Adj = dgl.graph((rows_, cols_), num_nodes=feature.shape[0], device='cuda')
            Adj.edata['w'] = values_
            Adj = dglgraph_to_sparse(Adj)
            learn_adj = (1-t) * Adj.cuda() + t * adj.cuda()
            h_p = torch.spmm(learn_adj.cuda(), h_a.cuda())
        else:
            h_a = torch.mm(adj, h_s)
            h_a = F.normalize(h_a, dim=1, p=2)
            similarities = cal_similarity_graph(h_a)
            similarities = top_k(similarities, k + 1)
            similarities = F.relu(similarities)
            Adj = symmetrize(similarities)
            Adj = normalize(Adj, 'sym', self.sparse)
            learn_adj = torch.add((1-t) * Adj.cuda(), t * adj.cuda()).to(running_device)
            h_p = torch.mm(learn_adj.cuda(), h_a.cuda())
        return h_a.detach(), h_p.detach()




