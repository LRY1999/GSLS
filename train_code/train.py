from time import perf_counter as t
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from util.utils import get_args
from util.dataset_unit import get_dataset
from eval.evaluate import clustering_metrics
from torch_geometric.utils import degree
from termcolor import cprint
from models.logreg import LogReg
from models.GSLS import USLonG
from eval.plot_embed import plot_embeddings
from sklearn.cluster import KMeans

#homology
def get_homo(label, edge_index):
    homo = (label[edge_index[0]] == label[edge_index[1]])
    homo = np.array(homo)
    return 1-np.mean(homo)



def train_model(args):
    # ======================================================================#
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ======================================================================#
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # ==============Loading Dataset=======================================#
    cprint("=====Loading Dataset======", "red")
    data_list, A_nomal = get_dataset(args)
    data=data_list[0]
    lable = data.y
    nb_feature = data.num_features
    nb_classes = int(lable.max() - lable.min()) + 1
    nb_nodes = data.num_nodes
    feature_X = data_list[1].to(device)
    A_nomal = A_nomal.to(device)
    if args.dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag']:
        feature_X = F.normalize(feature_X)
        feature_X = torch.spmm(A_nomal, feature_X)
    ylabelsx = []
    for i in range(0, nb_nodes):
        ylabelsx.append(i)
    ## ======================================================================#
    random_split = np.random.permutation(nb_nodes)
    train_index = random_split[:int(nb_nodes * 0.1)]
    test_index = random_split[int(nb_nodes * 0.1):]
    train_lbls = lable[train_index].squeeze().to(device)
    test_lbls = lable[test_index].squeeze().to(device)
    cprint("=====Loading Dataset Done=====", "red")


    ## ======================================================================#
    model = USLonG(nb_feature, cfg=args.cfg,
                   dropout=args.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd1)
    model.to(device)
    lable = lable.to(device)

    ## ======================================================================#
    A_degree = degree(data.edge_index[0], nb_nodes, dtype=int).tolist()
    #loop
    A_degree1 = degree(A_nomal._indices()[0], nb_nodes, dtype=int).tolist()
    edge=A_nomal._indices()
    d_s = []
    d_s.append(0)
    for i in range(nb_nodes):
        d_s.append(d_s[-1] + A_degree[i])
    n_list=[]

    dst = data.edge_index[1]
    n_f=[]
    for i in range(nb_nodes):
        s=d_s[i]
        d_e=s+A_degree1[i] - 1
        n = dst[s:d_e]
        n=torch.tensor(n)
        n_list.append(n)
        node = feature_X[i]
        nodes = torch.repeat_interleave(node, repeats=len(n), dim=0)
        nodes = nodes.reshape(len(n), len(feature_X[i]))
        n_f_1= F.pairwise_distance(nodes, feature_X[n])
        dis = n_f_1.argsort()
        n_k=n[dis]
        if len(n_k)==0:
            n_f_1 = np.array([i]*5)
            n_f_1=torch.tensor(n_f_1)
        elif (len(n_k) < 5) & (len(n_k)!=0):
            r_l =[j % len(n_k) for j in range(0,5)]
            n_f_1=n_k[r_l]
        else:
            n_f_1=n_k[0:5]
        n_f.append(n_f_1)
    n_f = torch.stack(n_f, dim=0)
    n_index_list = torch.t(n_f)

    #======================================================================#
    margin1 = args.margin1
    margin2 = margin1 + args.margin2
    triplet_loss = torch.nn.MarginRankingLoss(margin=margin1, reduce=False)

    #======================================================================#
    l_u = torch.tensor([0.]).to(device)

    start = t()
    for epoch in tqdm(range(0, args.epochs + 1)):
        model.train()
        optimiser.zero_grad()

        h_a, h_p ,aadj= model(feature_X.cuda(), A_nomal, args.lamda, args.k)
        A_nomal= aadj

        kmeans = KMeans(n_clusters=nb_classes).fit(h_a[ylabelsx].cpu().detach().numpy())
        predict_labels = kmeans.predict(h_a[ylabelsx].cpu().detach().numpy())
        predict_labels = torch.Tensor(predict_labels)
        loss_p = get_homo(predict_labels, edge.cpu().numpy())


        #neiborhood positive embeding
        h_a_1 = (h_a+h_a[n_index_list[0]] + h_a[n_index_list[1]] + h_a[n_index_list[2]] + h_a[n_index_list[3]] + h_a[n_index_list[4]])/6

        # negative embeding
        neg_emb_list = []
        for i in range(args.NegNum):
            idx_0 = np.random.permutation(nb_nodes)
            h_n = h_a[idx_0].to(device)
            if i % args.NegNum==0 & i!=0:
                seed = torch.rand(h_n.shape[0], h_n.shape[1]).to(device)
                h_n = seed * h_n + (1 - seed) * h_a_1
            neg_emb_list.append(h_n)

        #distance
        s_n_list = []
        for h_n in neg_emb_list:
            s_n = F.pairwise_distance(h_a, h_n)
            s_n_list.append(s_n)
        s_p = F.pairwise_distance(h_a.cuda(), h_p.cuda())
        s_p_1 = F.pairwise_distance(h_a, h_a_1)
        l_s = -1 * torch.ones_like(s_p)
        loss_s = 0
        loss_n = 0
        loss_u = 0
        for s_n in s_n_list:
            loss_s += (triplet_loss(s_p, s_n, l_s)).mean()
            loss_n += (triplet_loss(s_p_1, s_n, l_s)).mean()
            loss_u += torch.max((s_n - s_p.detach() - margin2), l_u).sum()
        loss_u = loss_u / args.NegNum
        loss = args.w_loss2*loss_p + loss_s * args.w_loss1 + loss_n * args.w_loss1 + loss_u
        loss.requires_grad_(True)
        loss.backward()
        optimiser.step()
        string_loss = "epoch:{}##### loss_1: {:.3f}||loss_2: {:.3f}||loss_3: {:.3f}||loss_4: {:.3f}".format(epoch+1, loss_s.item(), loss_n.item(), loss_u.item(),loss_p)
        cprint(string_loss,"yellow")
        torch.save(model.state_dict(), 'E:/GSLS/save_model/' + args.dataset_name + '.pth')
        if epoch % args.epochs == 0 and epoch != 0:
            model.eval()
            h_a, h_p = model.embed(feature_X, A_nomal.cuda(),args.lamda, args.k)
            embs = h_p
            embs = embs / embs.norm(dim=1)[:, None]
            embs_np= embs.cpu().numpy()
            if args.key == 'classification':
                train_embs = embs[train_index]
                test_embs = embs[test_index]
                accs = []
                cla_loss = nn.CrossEntropyLoss()
                for _ in range(2):
                    log = LogReg(args.dim, nb_classes)
                    opt = torch.optim.Adam(log.parameters(), lr=args.lr2, weight_decay=args.wd2)
                    log.to(device)
                    for _ in range(500):
                        log.train()
                        opt.zero_grad()
                        tra_cla_lb = log(train_embs)
                        loss = cla_loss(tra_cla_lb.cuda(), train_lbls.cuda())
                        loss.backward()
                        opt.step()
                    tra_cla_lb = log(test_embs)
                    preds = torch.argmax(tra_cla_lb, dim=1)
                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    accs.append(acc * 100)
                accs = torch.stack(accs)
                string_2 = "accs: {:.1f},std: {:.2f} ".format(accs.mean().item(), accs.std().item())
                print(string_2)
                noe = t()
                print('total time', noe - start)
                plot_embeddings(embs_np, lable)

            elif args.key == 'clusting':
                kmeans = KMeans(n_clusters=nb_classes).fit(embs[ylabelsx].cpu().detach().numpy())
                predict_labels = kmeans.predict(embs[ylabelsx].cpu().detach().numpy())

                cm = clustering_metrics(lable.cpu().detach().numpy(), predict_labels)
                acc, nmi, ari = cm.evaluationClusterModelFromLabel()

                print('Acc, nmi, ari:', acc, nmi, ari)
            else:
                print('error')





if __name__ == '__main__':
    my_args = get_args(dataset_class="Planetoid", dataset_name="Cora", key="classification")
    print(my_args)
    train_model(my_args)
    torch.cuda.empty_cache()
    cprint("Done",'yellow')